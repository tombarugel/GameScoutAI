"""HTTP client for the GameScout AI Cloudflare Worker."""

from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_WORKER_URL = "https://cloudflare-worker.tombarugel.workers.dev"
REQUEST_TIMEOUT_SECONDS = 90


def get_worker_url() -> str:
    """Resolve the AI backend URL."""
    configured = os.getenv("GAMESCOUT_WORKER_URL")

    if configured:
        return configured.rstrip("/")

    return DEFAULT_WORKER_URL.rstrip("/")


def check_worker_health(worker_url: str | None = None) -> tuple[bool, str]:
    base_url = (worker_url or get_worker_url()).rstrip("/")
    try:
        response = requests.get(f"{base_url}/health", timeout=8)
        response.raise_for_status()
        payload = response.json()
        return True, str(payload.get("model", "Workers AI"))
    except Exception as error:  # UI helper: return a human-readable status instead of crashing.
        return False, str(error)


def generate_game_concept(
    payload: dict[str, Any],
    worker_url: str | None = None,
) -> dict[str, Any]:
    """Generate one game concept through the Cloudflare Worker.

    Raises a RuntimeError with a concise user-facing message when the backend fails.
    """
    base_url = (worker_url or get_worker_url()).rstrip("/")

    try:
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not reach the AI service. Check that the Cloudflare Worker is deployed "
            "or set GAMESCOUT_WORKER_URL to a running local Worker."
        ) from error

    try:
        body = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"AI service returned a non-JSON response (HTTP {response.status_code})."
        ) from error

    if not response.ok:
        detail = body.get("details") or body.get("error") or f"HTTP {response.status_code}"
        raise RuntimeError(f"AI generation failed: {detail}")

    if not body.get("success") or not isinstance(body.get("concept"), dict):
        raise RuntimeError(body.get("error", "AI service returned an unexpected response."))

    return body["concept"]
