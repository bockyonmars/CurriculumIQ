"""HTTP client for the Spring gateway (used by Streamlit in ``gateway`` mode).

Keeps the same student-facing data shape the UI already renders, and converts
transport failures into short, safe messages. No OpenAI access here — the
gateway proxies the Python AI service, which owns all AI operations.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# (connect timeout, read timeout) in seconds.
DEFAULT_TIMEOUT: Tuple[float, float] = (3.0, 90.0)


class GatewayError(Exception):
    """Safe, user-facing failure talking to the gateway."""


class GatewayClient:
    def __init__(self, base_url: str, timeout: Tuple[float, float] = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _detail(self, resp: requests.Response) -> str:
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("detail"):
                return str(data["detail"])
        except Exception:
            pass
        return "The service could not process this request."

    def health(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Gateway health failed: %s", type(exc).__name__)
            raise GatewayError("The service is currently unavailable.") from exc

    def prepare_document(self, file_bytes: bytes, filename: str) -> dict:
        try:
            resp = requests.post(
                f"{self.base_url}/api/documents",
                files={"file": (filename, file_bytes, "application/pdf")},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Gateway prepare failed: %s", type(exc).__name__)
            raise GatewayError("The service is currently unavailable. Please try again shortly.") from exc
        if not resp.ok:
            raise GatewayError(self._detail(resp))
        return resp.json()

    def ask(self, document_id: str, question: str) -> dict:
        try:
            resp = requests.post(
                f"{self.base_url}/api/questions",
                json={"document_id": document_id, "question": question},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Gateway ask failed: %s", type(exc).__name__)
            raise GatewayError("The service is currently unavailable. Please try again shortly.") from exc
        if not resp.ok:
            raise GatewayError(self._detail(resp))
        return resp.json()
