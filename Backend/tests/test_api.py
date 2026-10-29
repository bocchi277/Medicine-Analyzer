"""HTTP behaviour of the API routes, with Gemini replaced by a fake."""


def test_healthz(client):
    assert client.get("/healthz").status_code == 200


def test_missing_name_is_400(client):
    assert client.get("/api/get_alternative").status_code == 400
    assert client.get("/api/drug_details").status_code == 400
    assert client.get("/api/compare_medicines?name1=Ibuprofen").status_code == 400


def test_not_found_returns_suggestions(client):
    r = client.get("/api/get_alternative?name=para&llm=0")
    assert r.status_code == 404
    body = r.get_json()
    assert "not found" in body["error"]
    assert any(s["label"].lower() == "paracetamol" for s in body["suggestions"])


def test_alternatives_are_approved_by_default(client):
    r = client.get("/api/get_alternative?name=Ibuprofen&llm=0")
    assert r.status_code == 200
    body = r.get_json()
    assert body["target"]["name"] == "Ibuprofen"
    assert len(body["similar"]) == 10
    assert all(d["approved"] for d in body["similar"])
    assert body["filters"]["approved_only"] is True
    assert body["recommendation"] is None          # llm=0 skips Gemini
    assert "doctor" in body["disclaimer"].lower()
    scores = [d["similarity_percentage"] for d in body["similar"]]
    assert scores == sorted(scores, reverse=True)
    assert all(d["name"] != "Ibuprofen" for d in body["similar"])


def test_approved_filter_can_be_turned_off(client):
    on = client.get("/api/get_alternative?name=Ibuprofen&llm=0").get_json()
    off = client.get("/api/get_alternative?name=Ibuprofen&llm=0&approved_only=0").get_json()
    assert off["filters"]["approved_only"] is False
    assert any(not d["approved"] for d in off["similar"])
    assert on["similar"] != off["similar"]


def test_recommendation_uses_llm_and_is_cached(client, fake_gemini):
    first = client.get("/api/get_alternative?name=Ibuprofen").get_json()
    second = client.get("/api/get_alternative?name=Advil").get_json()  # same drug
    assert first["recommendation"].startswith("Best Alternative:")
    assert second["recommendation"] == first["recommendation"]
    assert len(fake_gemini.prompts) == 1, "second request should be served from the cache"


def test_prompt_forbids_invented_brand_names(client, fake_gemini):
    client.get("/api/get_alternative?name=Metformin")
    prompt = fake_gemini.prompts[0]
    assert "even if they are not present" not in prompt
    assert "Never invent brand names" in prompt
    assert "Metformin" in prompt
    # Long fields are clipped so the prompt stays a reasonable size.
    assert len(prompt) < 20000


def test_llm_errors_are_not_cached(client, monkeypatch):
    import llm_service
    llm_service.clear_cache()
    monkeypatch.setattr(llm_service, "_get_client", lambda: (None, "not configured"))
    body = client.get("/api/get_alternative?name=Ibuprofen").get_json()
    assert body["recommendation"].startswith("Error")
    assert len(body["similar"]) == 10      # the list still works without the AI pick
    assert not llm_service._cache


def test_drug_details(client):
    r = client.get("/api/drug_details?name=Advil")
    assert r.status_code == 200
    body = r.get_json()
    assert body["name"] == "Ibuprofen"
    assert body["approved"] is True
    assert "combined_embedding_text" not in body


def test_drug_details_not_found(client):
    r = client.get("/api/drug_details?name=para")
    assert r.status_code == 404
    assert r.get_json()["suggestions"]


def test_compare_is_symmetric(client):
    a = client.get("/api/compare_medicines?name1=Ibuprofen&name2=Naproxen").get_json()
    b = client.get("/api/compare_medicines?name1=Naproxen&name2=Ibuprofen").get_json()
    assert a["similarity_percentage"] == b["similarity_percentage"]
    assert 0 < a["similarity_percentage"] < 100


def test_compare_not_found_names_the_missing_field(client):
    r = client.get("/api/compare_medicines?name1=Ibuprofen&name2=notarealdrug")
    assert r.status_code == 404
    assert r.get_json()["missing"] == "name2"


def test_suggest_route(client):
    r = client.get("/api/suggest?q=metf")
    assert r.status_code == 200
    assert r.get_json()["suggestions"][0]["name"] == "Metformin"


def test_cors_allows_only_known_origins(client):
    ok = client.get("/api/suggest?q=ibu", headers={"Origin": "https://medsphere707.netlify.app"})
    bad = client.get("/api/suggest?q=ibu", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "https://medsphere707.netlify.app"
    assert "Access-Control-Allow-Origin" not in bad.headers
