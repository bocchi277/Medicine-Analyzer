"""
Search-quality regression test. ATC codes are never used to build the
vectors, so they are a fair answer key (see evaluate.py). If a change to the
embedding text or model lowers these numbers, this test fails.
"""
import pytest


@pytest.fixture(scope="module")
def scores():
    import evaluate
    index, id_map = evaluate.load_index()
    return evaluate.atc_scores(index, id_map, evaluate.load_atc(), pool="all")


def test_all_atc_drugs_are_evaluated(scores):
    assert scores["drugs_evaluated"] == 3541


def test_therapeutic_group_accuracy(scores):
    # Old index (name-first text with "Not Available" filler): 36.8%.
    # Rebuilt index (categories and targets first): see evaluate.py output.
    assert scores["levels"]["L2 therapeutic group"]["top1"] >= 48.0


def test_organ_system_accuracy(scores):
    assert scores["levels"]["L1 organ system"]["top1"] >= 55.0
