"""GatewayClient tests — requests fully mocked, no network, no OpenAI."""

from __future__ import annotations

import pytest
import requests

from src import service_client
from src.service_client import GatewayClient, GatewayError


class FakeResp:
    def __init__(self, *, status=200, json_data=None, ok=True):
        self.status_code = status
        self._json = json_data if json_data is not None else {}
        self.ok = ok

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError("bad status")


def test_health_success(monkeypatch):
    monkeypatch.setattr(service_client.requests, "get",
                        lambda *a, **k: FakeResp(json_data={"gateway": "ok", "pythonService": "up"}))
    assert GatewayClient("http://gw:8080").health()["pythonService"] == "up"


def test_health_unavailable(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(service_client.requests, "get", boom)
    with pytest.raises(GatewayError):
        GatewayClient("http://gw:8080").health()


def test_prepare_document_success(monkeypatch):
    captured = {}

    def fake_post(url, files=None, timeout=None, **k):
        captured["url"] = url
        captured["has_file"] = files is not None and "file" in files
        return FakeResp(json_data={"document_id": "doc_1", "filename": "a.pdf",
                                   "pages": 6, "chunks": 6, "skipped_pages": [], "status": "ready"})
    monkeypatch.setattr(service_client.requests, "post", fake_post)
    out = GatewayClient("http://gw:8080").prepare_document(b"%PDF-1.4", "a.pdf")
    assert out["document_id"] == "doc_1" and out["pages"] == 6
    assert captured["url"].endswith("/api/documents") and captured["has_file"]


def test_prepare_document_error_detail(monkeypatch):
    monkeypatch.setattr(service_client.requests, "post",
                        lambda *a, **k: FakeResp(status=422, ok=False,
                                                 json_data={"detail": "Only PDF files are supported."}))
    with pytest.raises(GatewayError, match="Only PDF files"):
        GatewayClient("http://gw:8080").prepare_document(b"x", "a.txt")


def test_ask_success(monkeypatch):
    def fake_post(url, json=None, timeout=None, **k):
        assert json == {"document_id": "doc_1", "question": "What is a variable?"}
        return FakeResp(json_data={"answer": "A symbol [S1].", "abstained": False,
                                   "citations": [{"source_id": "S1", "filename": "a.pdf",
                                                  "page": 1, "passage": "..."}]})
    monkeypatch.setattr(service_client.requests, "post", fake_post)
    out = GatewayClient("http://gw:8080").ask("doc_1", "What is a variable?")
    assert out["abstained"] is False and out["citations"][0]["page"] == 1


def test_ask_unreachable(monkeypatch):
    def boom(*a, **k):
        raise requests.Timeout("timeout")
    monkeypatch.setattr(service_client.requests, "post", boom)
    with pytest.raises(GatewayError):
        GatewayClient("http://gw:8080").ask("doc_1", "q")
