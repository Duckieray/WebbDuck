from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import provider_credentials as credentials
from server.provider_credentials_api import router


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("WEBBDUCK_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CIVITAI_TOKEN",
        "CIVITAI_API_TOKEN",
        "CIVITAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    credentials._MANAGED_ENV.clear()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_provider_credential_api_never_echoes_saved_secret(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    secret = "hf_do_not_echo_this"

    response = client.put("/settings/provider-credentials/huggingface", json={"token": secret})
    assert response.status_code == 200
    assert secret not in response.text
    assert response.json()["providers"]["huggingface"]["configured"] is True

    response = client.get("/settings/provider-credentials")
    assert response.status_code == 200
    assert secret not in response.text
    assert "token" not in json.dumps(response.json()).lower()

    response = client.delete("/settings/provider-credentials/huggingface")
    assert response.status_code == 200
    assert response.json()["providers"]["huggingface"]["configured"] is False


def test_provider_credential_api_rejects_unknown_provider(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    response = client.put("/settings/provider-credentials/not-a-provider", json={"token": "secret"})
    assert response.status_code == 400
