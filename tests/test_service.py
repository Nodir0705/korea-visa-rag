"""Service behaviour without models: the app is driven with a fake answerer and the
lifespan (which loads the models) is never entered."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from visarag import api
from visarag.refresh import diff, page_set


class FakeAnswerer:
    def ask(self, question, mode):
        return SimpleNamespace(
            refused=False, citations=[1], flags=[], timings={"total": 0.01},
            to_dict=lambda: {"question": question, "answer": "ok [1]"},
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(api.state, "answerer", FakeAnswerer())
    monkeypatch.setattr(api, "limiter", api.RateLimiter(1000))
    return TestClient(api.app)  # not used as a context manager: no lifespan, no models


def test_ask_returns_answer_and_request_id(client):
    r = client.post("/v1/ask", json={"question": "What is a D-2 visa?"}, headers={"X-Request-ID": "abc"})
    assert r.status_code == 200 and r.json()["answer"] == "ok [1]"
    assert r.headers["X-Request-ID"] == "abc"


def test_api_key_required_when_set(client, monkeypatch):
    monkeypatch.setattr(api.settings, "api_key", "s3cret")
    assert client.post("/v1/ask", json={"question": "What is a D-2 visa?"}).status_code == 401
    r = client.post("/v1/ask", json={"question": "What is a D-2 visa?"}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(api, "limiter", api.RateLimiter(2))
    codes = [client.post("/v1/ask", json={"question": "What is a D-2 visa?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limiter_window_slides():
    rl = api.RateLimiter(1)
    assert rl.retry_after("a", now=0) == 0
    assert rl.retry_after("a", now=30) > 0
    assert rl.retry_after("b", now=30) == 0
    assert rl.retry_after("a", now=61) == 0


def test_queue_refuses_beyond_limit():
    q = api.GpuQueue(1)
    with q:
        with pytest.raises(HTTPException) as exc:
            q.__enter__()
        assert exc.value.status_code == 503
    with q:  # released again
        pass


def test_health_is_503_when_a_dependency_is_down(client, monkeypatch):
    monkeypatch.setattr(api, "check_health", lambda: (False, {"ollama": {"ok": False}}))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


def test_refresh_diff_and_page_hash_ignore_front_matter(tmp_path):
    (tmp_path / "D-2.md").write_text("---\nfetched_at: 2026-01-01\n---\n\nbody\n", encoding="utf-8")
    first = page_set(tmp_path)
    (tmp_path / "D-2.md").write_text("---\nfetched_at: 2026-02-02\n---\n\nbody\n", encoding="utf-8")
    assert page_set(tmp_path) == first
    assert diff({"a": "1", "b": "1"}, {"b": "2", "c": "1"}) == {"added": ["c"], "removed": ["a"], "changed": ["b"]}
