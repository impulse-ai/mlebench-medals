"""Thin HTTP wrapper around the Impulse gateway for bench-driven runs.

The gateway flow we exercise (per recon report):
  1. POST /api/datasets/upload  (multipart)            → dataset_id
  2. GET  /api/datasets/{id}    (poll until "ready")
  3. POST /api/chat             (keyword-routes to dataset_training tool)
  4. GET  /api/jobs/{job_id}    (poll until "succeeded")
  5. POST /api/chat             (predict on a different dataset)
  6. GET  download_url          (predictions.csv)

Auth: in local-dev mode, the gateway accepts `x-guest-id: <uuid>` and skips
Auth0. In prod we'll need a real bearer token.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


@dataclass
class ImpulseGatewayClient:
    base_url: str = "http://localhost:3001"
    guest_id: str | None = None
    auth_token: str | None = None
    request_timeout_s: int = 120
    session: requests.Session = field(default_factory=requests.Session, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.guest_id and not self.auth_token:
            # Default to a stable guest id per client instance
            self.guest_id = str(uuid.uuid4())
        if self.guest_id:
            self.session.headers["x-guest-id"] = self.guest_id
        if self.auth_token:
            self.session.headers["Authorization"] = f"Bearer {self.auth_token}"

    # ---- datasets -----------------------------------------------------------

    def upload_dataset(self, csv_path: Path, name: str | None = None) -> str:
        """Upload a CSV via multipart and return the dataset_id."""
        url = f"{self.base_url}/api/datasets/upload"
        # Don't send default JSON Content-Type for multipart
        headers = {
            k: v for k, v in self.session.headers.items()
            if k.lower() != "content-type"
        }
        with csv_path.open("rb") as f:
            files = {"file": (csv_path.name, f, "text/csv")}
            data = {"name": name or csv_path.stem}
            resp = requests.post(
                url, files=files, data=data, headers=headers,
                timeout=max(300, self.request_timeout_s),
            )
        resp.raise_for_status()
        body = resp.json()
        if "id" not in body:
            raise RuntimeError(f"upload response missing 'id': {body}")
        return body["id"]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.base_url}/api/datasets/{dataset_id}",
            timeout=self.request_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_profiling(
        self, dataset_id: str, timeout_s: int = 300, poll_s: float = 2.0,
    ) -> dict[str, Any]:
        """Block until the gateway has profiled the dataset (status == 'ready')."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            info = self.get_dataset(dataset_id)
            status = (info.get("status") or "").lower()
            if status == "ready":
                return info
            if status in {"error", "failed"}:
                raise RuntimeError(f"dataset {dataset_id} profiling failed: {info}")
            time.sleep(poll_s)
        raise TimeoutError(
            f"dataset {dataset_id} did not become 'ready' within {timeout_s}s"
        )

    # ---- chat / training ----------------------------------------------------

    def chat(
        self,
        message: str,
        dataset_id: str | None = None,
        conversation_id: str | None = None,
        budget_preset: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat turn; gateway keyword-routes to dataset_training.

        Body schema is what `gateway/src/schemas/chat.ts:ChatRequest` accepts:
          - messages: array of {role, content}  (not a single 'message' string)
          - datasetId: camelCase
          - contextId: camelCase (the prior 'conversation_id' equivalent)
          - stream: false (default is true; we don't speak SSE here)

        Gateway's /api/chat 307-redirects to /api/v2/chat (agent-api proxy)
        unless USE_AGENT_V2=false. We disable redirect-following and surface
        the redirect as a clear error.
        """
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }
        if dataset_id:
            body["datasetId"] = dataset_id
        if conversation_id:
            body["contextId"] = conversation_id
        if budget_preset:
            body["budget"] = {"preset": budget_preset}

        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json=body,
            timeout=self.request_timeout_s,
            allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location", "")
            raise RuntimeError(
                f"gateway redirected /api/chat → {loc!r}. "
                f"Set USE_AGENT_V2=false on the gateway, or run the agent-api "
                f"service (port 8005) and point ImpulseAgent at it."
            )
        if not resp.ok:
            # Surface the response body verbatim — `raise_for_status()` only
            # shows the URL + status, which hides Zod validation errors.
            raise RuntimeError(
                f"gateway /api/chat {resp.status_code}: "
                f"{resp.text[:800] or '(empty body)'}"
            )
        return resp.json()

    # ---- jobs ---------------------------------------------------------------

    def get_job(self, job_id: str) -> dict[str, Any]:
        resp = self.session.get(
            f"{self.base_url}/api/jobs/{job_id}", timeout=self.request_timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("job", body)

    def wait_for_job(
        self, job_id: str, timeout_s: int, poll_s: float = 5.0,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            job = self.get_job(job_id)
            status = (job.get("status") or "").lower()
            if status in {"succeeded", "success", "completed"}:
                return job
            if status in {"failed", "error"}:
                raise RuntimeError(
                    f"job {job_id} failed: {job.get('error_message') or job}"
                )
            time.sleep(poll_s)
        raise TimeoutError(f"job {job_id} did not complete within {timeout_s}s")

    # ---- downloads ----------------------------------------------------------

    def download(self, url: str, out_path: Path) -> Path:
        """Download a (possibly signed) URL through the session."""
        # If the URL is relative, prepend base_url
        if url.startswith("/"):
            url = f"{self.base_url}{url}"
        resp = self.session.get(url, timeout=max(300, self.request_timeout_s))
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return out_path
