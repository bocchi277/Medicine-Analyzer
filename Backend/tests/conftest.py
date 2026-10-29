"""
Shared fixtures. The tests run against the real data and index, so they check
what the deployed API will actually return, not a toy copy.

    cd Backend
    pip install -r requirements.txt -r requirements-dev.txt
    pytest -q
"""
import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BACKEND_DIR)  # config.py uses paths relative to Backend/
sys.path.insert(0, BACKEND_DIR)
# Never call the real Gemini API from tests.
os.environ["GEMINI_API_KEY"] = ""


@pytest.fixture(scope="session")
def flask_app():
    from app import app
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(flask_app):
    import api
    with api._rate_lock:
        api._rate_state.clear()
    return flask_app.test_client()


class FakeGemini:
    """Stands in for google.genai.Client; records every prompt it receives."""

    def __init__(self, reply="Best Alternative: Naproxen\nReasoning: Same use.\nBrand names: Aleve"):
        self.prompts = []
        self.reply = reply
        self.models = self

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)

        class Response:
            text = self.reply
            candidates = None
            prompt_feedback = None

        return Response()


@pytest.fixture()
def fake_gemini(monkeypatch):
    import llm_service
    fake = FakeGemini()
    llm_service.clear_cache()
    monkeypatch.setattr(llm_service, "_get_client", lambda: (fake, None))
    yield fake
    llm_service.clear_cache()
