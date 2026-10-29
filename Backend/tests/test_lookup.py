"""Drug-name resolution: exact names, brands, partial input, suggestions."""
import json

import pytest


@pytest.fixture(scope="module")
def api(flask_app):
    import api as api_module
    return api_module


def resolved_name(api, query):
    row = api.find_drug(query)
    return None if row is None else row["name"]


@pytest.mark.parametrize("query, expected", [
    ("Ibuprofen", "Ibuprofen"),
    ("  ibuprofen ", "Ibuprofen"),          # case and whitespace
    ("Dexibuprofen", "Dexibuprofen"),       # exact name beats a synonym match
    ("Advil", "Ibuprofen"),                 # DrugBank brand
    ("Tylenol", "Acetaminophen"),
    ("Paracetamol", "Acetaminophen"),       # synonym
    ("Metformin", "Metformin"),
])
def test_exact_names(api, query, expected):
    assert resolved_name(api, query) == expected


@pytest.mark.parametrize("query, expected", [
    ("Crocin", "Acetaminophen"),   # DrugBank has a saffron compound named Crocin
    ("Dolo 650", "Acetaminophen"),
    ("Calpol", "Acetaminophen"),
    ("Brufen", "Ibuprofen"),
    ("Pan 40", "Pantoprazole"),
    ("Glycomet", "Metformin"),
])
def test_indian_brands(api, query, expected):
    assert resolved_name(api, query) == expected


def test_every_curated_brand_points_at_a_real_drug(api):
    with open(api.BRAND_ALIAS_FILE, encoding="utf-8") as f:
        brands = json.load(f)["brands"]
    for brand, target in brands.items():
        assert resolved_name(api, brand) == target, brand


def test_short_partial_is_not_guessed(api):
    # "para" used to resolve to Paraoxon, a pesticide metabolite.
    assert api.find_drug("para") is None
    labels = [s["label"].lower() for s in api.suggest("para")]
    assert "paracetamol" in labels


def test_longer_prefix_resolves(api):
    assert resolved_name(api, "ibupro") == "Ibuprofen"
    assert resolved_name(api, "metfor") == "Metformin"


def test_interior_substring_is_not_used_to_guess(api):
    # "profen" appears inside many names but starts none of the right ones.
    assert api.find_drug("xyzzyprofen") is None


def test_unknown_name(api):
    assert api.find_drug("notarealdrug") is None
    assert api.find_drug("") is None
    assert api.find_drug("   ") is None


def test_suggest_ranks_approved_name_first(api):
    results = api.suggest("ibu")
    assert results, "expected suggestions for 'ibu'"
    assert results[0]["name"] == "Ibuprofen"
    # One entry per drug.
    ids = [r["drug_id"] for r in results]
    assert len(ids) == len(set(ids))


def test_suggest_ignores_tiny_input(api):
    assert api.suggest("") == []
    assert api.suggest("a") == []
