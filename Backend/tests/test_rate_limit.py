"""Rate limiting, including attempts to dodge it with forged headers."""


def hit(client, n, headers=None):
    return [client.get("/api/get_alternative?name=Ibuprofen&llm=0", headers=headers or {}).status_code
            for _ in range(n)]


def test_thirteenth_call_in_a_minute_is_429(client):
    codes = hit(client, 13)
    assert codes[:12] == [200] * 12
    assert codes[12] == 429


def test_429_has_retry_after(client):
    hit(client, 12)
    r = client.get("/api/get_alternative?name=Ibuprofen&llm=0")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_forged_x_forwarded_for_does_not_reset_the_limit(client):
    # Behind Render the client cannot change CF-Connecting-IP (Cloudflare sets
    # it). Rotating the first X-Forwarded-For entry used to give a fresh bucket.
    codes = [
        client.get("/api/get_alternative?name=Ibuprofen&llm=0",
                   headers={"CF-Connecting-IP": "203.0.113.7",
                            "X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}).status_code
        for i in range(13)
    ]
    assert codes[12] == 429


def test_without_cloudflare_the_last_forwarded_entry_is_used(client):
    codes = [
        client.get("/api/get_alternative?name=Ibuprofen&llm=0",
                   headers={"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.9"}).status_code
        for i in range(13)
    ]
    assert codes[12] == 429


def test_different_clients_have_separate_limits(client):
    hit(client, 12, {"CF-Connecting-IP": "203.0.113.1"})
    r = client.get("/api/get_alternative?name=Ibuprofen&llm=0",
                   headers={"CF-Connecting-IP": "203.0.113.2"})
    assert r.status_code == 200
