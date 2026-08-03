"""Agent adapter protocol for MLE-bench runs.

Anything that produces a `submission.csv` given a data directory can be
benchmarked. Two concrete adapters today:
  - ScriptAgent  — wraps a CLI experiment script (used for harness testing).
  - ImpulseAgent — drives the Impulse gateway end-to-end (production target).
"""
from __future__ import annotations

import json
import os
import base64
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class AgentResult:
    success: bool
    submission_path: Path | None
    wall_time_s: float
    stdout: str
    stderr: str
    error: str | None = None
    # Free-form run metadata (tree milestone, budget, run record, …). Defaults
    # empty so ScriptAgent/ImpulseAgent construction sites are untouched.
    meta: dict = field(default_factory=dict)


class Agent(Protocol):
    name: str

    def run(self, data_dir: Path, work_dir: Path, timeout_s: int) -> AgentResult: ...


@dataclass
class ScriptAgent:
    """Run a Python script that accepts --data-dir and --output.

    Convention enforced for every script under bench/experiments/:
      python <script> --data-dir <task-data> --output <submission.csv>
    """

    script_path: Path
    extra_args: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"script:{self.script_path.stem}"

    def run(self, data_dir: Path, work_dir: Path, timeout_s: int) -> AgentResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        submission_path = work_dir / "submission.csv"

        cmd = [
            sys.executable,
            str(self.script_path),
            "--data-dir", str(data_dir),
            "--output", str(submission_path),
            *self.extra_args,
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return AgentResult(
                success=False, submission_path=None, wall_time_s=float(timeout_s),
                stdout="", stderr="", error=f"agent timed out after {timeout_s}s",
            )
        wall = time.time() - t0

        if proc.returncode != 0:
            return AgentResult(
                success=False, submission_path=None, wall_time_s=wall,
                stdout=proc.stdout, stderr=proc.stderr,
                error=f"agent exited with code {proc.returncode}",
            )
        if not submission_path.exists():
            return AgentResult(
                success=False, submission_path=None, wall_time_s=wall,
                stdout=proc.stdout, stderr=proc.stderr,
                error="agent did not produce submission.csv",
            )
        return AgentResult(
            success=True, submission_path=submission_path, wall_time_s=wall,
            stdout=proc.stdout, stderr=proc.stderr,
        )


@dataclass
class EESStandaloneAgent:
    """Run the shared benchmark-first EES core locally."""

    grade_inside_core: bool = False

    @property
    def name(self) -> str:
        return "ees:standalone"

    def run(self, data_dir: Path, work_dir: Path, timeout_s: int) -> AgentResult:
        from .ees_core.controller import run_ees_task

        t0 = time.time()
        old_evolution_enabled = os.environ.get("EES_EVOLUTION_ENABLED")
        if old_evolution_enabled is None:
            os.environ["EES_EVOLUTION_ENABLED"] = "1"
        try:
            result = run_ees_task(
                task_id=data_dir.parts[-3] if len(data_dir.parts) >= 3 else data_dir.name,
                data_dir=data_dir,
                work_dir=work_dir / "ees_core",
                grade_submission=self.grade_inside_core,
            )
        finally:
            if old_evolution_enabled is None:
                os.environ.pop("EES_EVOLUTION_ENABLED", None)
            else:
                os.environ["EES_EVOLUTION_ENABLED"] = old_evolution_enabled
        stdout = json.dumps(
            {
                "candidate_id": result.candidate_id,
                "recipe_id": result.recipe_id,
                "score": result.score,
                "medal": result.medal,
                "final_artifact_source": result.final_artifact_source,
                "no_score_reason": result.no_score_reason,
                "artifact_paths": result.artifact_paths,
            },
            indent=2,
        )
        return AgentResult(
            success=result.success and result.submission_path is not None,
            submission_path=result.submission_path,
            wall_time_s=time.time() - t0,
            stdout=stdout,
            stderr="",
            error=result.no_score_reason,
        )


@dataclass
class McpHttpImpulseAgent:
    """Drive Impulse through the HTTP MCP endpoint.

    This adapter intentionally uses only the public MCP tools exposed by
    `/api/mcp-http`: auth status, dataset upload, chat, and chat status. The
    current upload tool accepts single files, so directory-backed MLE-bench
    tasks (image/audio archives) fail clearly instead of pretending that a
    labels-only CSV is a complete dataset.
    """

    mcp_url: str
    api_key: str | None = None
    bearer_token: str | None = None
    guest_id: str = "bench-runner"
    max_upload_bytes: int = 80 * 1024 * 1024

    @property
    def name(self) -> str:
        return "mcp:http"

    def run(self, data_dir: Path, work_dir: Path, timeout_s: int) -> AgentResult:
        import requests

        from .adapters.task_detection import _find_sample_submission, detect_schema

        work_dir.mkdir(parents=True, exist_ok=True)
        transcript: list[dict[str, Any]] = []
        transcript_path = work_dir / "mcp_transcript.json"
        t0 = time.time()

        def _record(kind: str, **payload: Any) -> None:
            transcript.append({"kind": kind, "timestamp": time.time(), **payload})
            transcript_path.write_text(json.dumps(transcript, indent=2) + "\n")

        def _remaining() -> int:
            return max(1, timeout_s - int(time.time() - t0))

        try:
            client = _McpHttpClient(
                self.mcp_url,
                api_key=self.api_key,
                bearer_token=self.bearer_token,
                guest_id=self.guest_id,
            )
            auth_text = client.call_tool("impulse_auth_status", {})
            _record("tool_result", tool="impulse_auth_status", text=auth_text)

            schema = detect_schema(data_dir)
            if schema.modality not in {"tabular", "nlp"}:
                raise RuntimeError(
                    "MCP HTTP upload currently supports single-file tabular/NLP "
                    f"datasets only; detected modality={schema.modality!r}. "
                    "Expose an archive/URL ingestion tool before running this task."
                )

            train_file = data_dir / "train.csv"
            test_file = data_dir / "test.csv"
            if not train_file.exists() and (data_dir / "train.json").exists():
                train_file = data_dir / "train.json"
            if not test_file.exists() and (data_dir / "test.json").exists():
                test_file = data_dir / "test.json"

            for path in (train_file, test_file):
                if not path.exists():
                    raise FileNotFoundError(f"{path.name} not found in {data_dir}")
                if path.stat().st_size > self.max_upload_bytes:
                    mb = self.max_upload_bytes // 1024 // 1024
                    raise RuntimeError(
                        f"{path.name} is too large for MCP base64 upload "
                        f"({path.stat().st_size:,} bytes > {mb}MB cap)"
                    )

            train_dataset_id = self._upload_file(
                client,
                train_file,
                name=f"{data_dir.name}-train",
                description=f"MLE-bench {data_dir.name} training data",
                record=_record,
            )
            test_dataset_id = self._upload_file(
                client,
                test_file,
                name=f"{data_dir.name}-test",
                description=f"MLE-bench {data_dir.name} held-out test data",
                record=_record,
            )

            sample = _find_sample_submission(data_dir)
            sample_header = ""
            sample_rows = "unknown"
            if sample and sample.exists():
                lines = sample.read_text(errors="ignore").splitlines()
                sample_header = lines[0] if lines else ""
                sample_rows = str(max(0, len(lines) - 1))

            message = "\n".join(
                [
                    f"Run this MLE-bench/Kaggle task: {data_dir.name}.",
                    f"Primary metric: {schema.metric or 'use the competition metric if known'}.",
                    f"ID column: {schema.id_col}. Target columns: {', '.join(schema.target_cols)}.",
                    f"Training file: {train_file.name}. Test file: {test_file.name}.",
                    f"The test dataset is attached as test_dataset_id={test_dataset_id}.",
                    "Train a model, predict the test rows, and persist a submission.csv artifact.",
                    f"The submission.csv header must be exactly: {sample_header or '(match sample_submission.csv)'}.",
                    f"It must contain {sample_rows} prediction rows in the same ID order as the test/sample submission.",
                    "Do not return only metrics; save submission.csv as an artifact.",
                ]
            )
            chat_text = client.call_tool(
                "impulse_chat",
                {
                    "message": message,
                    "dataset_id": train_dataset_id,
                    "test_dataset_id": test_dataset_id,
                    "mode": "intelligence",
                    "wait_for_completion": False,
                },
                timeout_s=min(_remaining(), 120),
            )
            _record("tool_result", tool="impulse_chat", text=chat_text)
            session_id = _extract_mcp_text_field(chat_text, "session_id")
            if not session_id:
                raise RuntimeError("impulse_chat did not return session_id")

            final_status = chat_text
            while _remaining() > 0:
                status_text = client.call_tool(
                    "impulse_chat_status",
                    {
                        "session_id": session_id,
                        "include_artifacts": True,
                        "include_diagnostics": True,
                        "wait_for_terminal": False,
                        "max_wait_seconds": 0,
                    },
                    timeout_s=min(60, _remaining()),
                )
                final_status = status_text
                _record("tool_result", tool="impulse_chat_status", text=status_text)
                if _extract_mcp_text_field(status_text, "terminal") == "true":
                    break
                if (
                    _extract_mcp_text_field(status_text, "awaiting_user_input") == "true"
                    or _extract_mcp_text_field(status_text, "recommended_action") == "reply_with_decision"
                ):
                    decision_text = client.call_tool(
                        "impulse_chat",
                        {
                            "session_id": session_id,
                            "message": (
                                f"Yes, proceed. Use {train_file.name} as the labeled training set "
                                f"and {test_file.name} as the held-out Kaggle test set. Train the "
                                "model, predict the test rows, and persist submission.csv "
                                f"with the required header: {sample_header or 'match the sample submission'}."
                            ),
                            "mode": "intelligence",
                            "wait_for_completion": False,
                        },
                        timeout_s=min(60, _remaining()),
                    )
                    _record("tool_result", tool="impulse_chat", text=decision_text)
                time.sleep(min(30, _remaining()))

            submission_url = _extract_submission_url(final_status)
            submission_path = work_dir / "submission.csv"
            if submission_url:
                client.download(submission_url, submission_path)
                _record("download", url=submission_url, path=str(submission_path))

            if submission_path.exists():
                return AgentResult(
                    success=True,
                    submission_path=submission_path,
                    wall_time_s=time.time() - t0,
                    stdout=json.dumps({"session_id": session_id, "transcript": str(transcript_path)}, indent=2),
                    stderr="",
                )
            return AgentResult(
                success=False,
                submission_path=None,
                wall_time_s=time.time() - t0,
                stdout=json.dumps({"session_id": session_id, "transcript": str(transcript_path), "status": final_status}, indent=2),
                stderr="",
                error="MCP run finished without a downloadable submission.csv artifact",
            )
        except Exception as e:
            transcript_path.write_text(json.dumps(transcript, indent=2) + "\n")
            return AgentResult(
                success=False,
                submission_path=None,
                wall_time_s=time.time() - t0,
                stdout=json.dumps({"transcript": str(transcript_path)}, indent=2),
                stderr=traceback.format_exc(),
                error=f"McpHttpImpulseAgent.run raised {type(e).__name__}: {e}",
            )

    def _upload_file(
        self,
        client: "_McpHttpClient",
        path: Path,
        *,
        name: str,
        description: str,
        record: Callable[..., None],
    ) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        text = client.call_tool(
            "impulse_dataset_upload_base64",
            {
                "filename": path.name,
                "content_base64": encoded,
                "name": name,
                "description": description,
            },
            timeout_s=180,
        )
        record("tool_result", tool="impulse_dataset_upload_base64", filename=path.name, text=text)
        dataset_id = _extract_mcp_text_field(text, "dataset_id")
        if not dataset_id:
            raise RuntimeError(f"upload did not return dataset_id for {path.name}: {text[:300]}")
        return dataset_id


class _McpHttpClient:
    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        guest_id: str = "bench-runner",
    ) -> None:
        import requests

        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["content-type"] = "application/json"
        if api_key:
            self.session.headers["x-api-key"] = api_key
        elif bearer_token:
            self.session.headers["Authorization"] = f"Bearer {bearer_token}"
        else:
            self.session.headers["x-guest-id"] = guest_id

    def call_tool(self, name: str, arguments: dict[str, Any], timeout_s: int = 120) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": f"{name}-{uuid.uuid4().hex}",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = self.session.post(self.url, json=payload, timeout=timeout_s)
        if not resp.ok:
            raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:500]}")
        body = resp.json()
        if body.get("error"):
            raise RuntimeError(f"MCP error: {body['error']}")
        result = body.get("result") or {}
        text = ""
        for entry in result.get("content") or []:
            if entry.get("type") == "text":
                text = entry.get("text") or ""
                break
        if result.get("isError"):
            raise RuntimeError(text or f"MCP tool {name} returned isError")
        return text

    def download(self, url: str, destination: Path) -> None:
        if url.startswith("/"):
            from urllib.parse import urlsplit

            root = f"{urlsplit(self.url).scheme}://{urlsplit(self.url).netloc}"
            url = f"{root}{url}"
        resp = self.session.get(url, timeout=300)
        if not resp.ok:
            raise RuntimeError(f"download {url} failed: {resp.status_code} {resp.text[:200]}")
        destination.write_bytes(resp.content)


def _extract_mcp_text_field(text: str, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}=(.*)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    return None if value in {"", "n/a", "null"} else value


def _extract_submission_url(text: str) -> str | None:
    for line in text.splitlines():
        if "submission.csv" not in line:
            continue
        if line.startswith("artifact_"):
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[1].endswith("submission.csv"):
                return parts[2].strip()
        match = re.search(r'"filename"\s*:\s*"submission\.csv".*?"url"\s*:\s*"([^"]+)"', line)
        if match:
            return match.group(1)
    return None


@dataclass
class ImpulseAgent:
    """Drive the FULL Impulse agentic loop against `agent-api`.

    Mirrors the production agentic frontend (port 3002 in start_agentic.sh):
        POST /v2/datasets/upload  (multipart: train.csv + test.csv + sample_submission.csv,
                                   all attached to the same session_id so they
                                   land in the same sandbox workspace)
        POST /v2/chat             (SSE stream — Gemini-driven SDK orchestrator
                                   writes Python that runs in the Docker sandbox)
        GET  /v2/sessions/{id}/artifacts → list with signed URLs
        GET  <signed_url>         → download submission.csv

    No training-mcp, no recipes, no manual feature engineering. The model
    is whatever Gemini decides to write — same path as a real user typing
    "train me a model" into the production UI.

    `version`:
      - "current": today's agentic stack (default)
      - "new":     reserved for future post-replacement agent (INTEL-108)
    """

    agent_api_url: str = "http://localhost:8005"
    sandbox_url: str = "http://localhost:8006"
    version: str = "current"
    user_id: str = "bench-runner"
    # Optional callback invoked for every log line emitted during run().
    # Bench-service wires this to its filesystem log so the studio UI can
    # stream progress live (via the existing 2.5s poll of /runs/{id}/log).
    log_callback: Callable[[str], None] | None = None
    # Multi-turn settings. A real power-user typically iterates:
    #   turn 1 → "build me a model"
    #   turn 2 → "now improve it"
    # We default to 1 refinement turn after the initial build (matching the
    # most common production usage pattern). Set higher for more aggressive
    # benchmarking; set 0 to test the single-shot path only.
    refinement_turns: int = 1

    @property
    def name(self) -> str:
        return f"impulse:{self.version}"

    def run(self, data_dir: Path, work_dir: Path, timeout_s: int) -> AgentResult:
        import requests

        from .adapters.task_detection import detect_schema

        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log: list[str] = []
        errlog: list[str] = []

        def _log(msg: str) -> None:
            log.append(msg)
            print(f"    {msg}")
            if self.log_callback:
                try:
                    self.log_callback(msg)
                except Exception:
                    # Logger failures must NEVER kill the agent loop.
                    pass

        def _remaining() -> int:
            return max(60, timeout_s - int(time.time() - t0))

        url = self.agent_api_url.rstrip("/")
        sandbox = self.sandbox_url.rstrip("/")
        sess = requests.Session()

        try:
            # 1. Detect schema from sample_submission.csv
            _log(f"detecting schema in {data_dir}")
            schema = detect_schema(data_dir)
            _log(
                f"  id_col={schema.id_col}  targets={schema.target_cols}  "
                f"n_train={schema.n_train}  n_test={schema.n_test}  "
                f"metric={schema.metric or '<unknown>'}"
            )

            train_csv = (data_dir / "train.csv").resolve()
            test_csv = (data_dir / "test.csv").resolve()
            for p in (train_csv, test_csv):
                if not p.exists():
                    raise FileNotFoundError(f"{p.name} not in {data_dir}")
            # Sample-submission filename varies across competitions
            # (sample_submission.csv / sampleSubmission.csv / *_null.csv).
            # Locate it tolerantly; it's only used for the byte-identity guard
            # below, so a miss is non-fatal.
            from bench.adapters.task_detection import _find_sample_submission
            _sample = _find_sample_submission(data_dir)
            sample_csv = _sample.resolve() if _sample else None

            # SDK-mode agent-api expects files already staged in the sandbox
            # workspace, identified by session_id. We mint the session_id and
            # stage data according to the task's modality (see _choose_staging):
            #   - tabular/nlp:  PUT train.csv + test.csv directly to the sandbox
            #                   (NLP tasks are text-in-CSV).
            #   - vision/audio: pack the whole task tree as task.tar.gz, upload
            #                   to GCS, and have the sandbox pull it; the agent
            #                   must extract the archive before training.
            #
            # IMPORTANT: we deliberately do NOT upload sample_submission.csv
            # to the sandbox. Empirically the ML Engineer agent has been
            # treating it as a "ready-made template" and writing it back
            # verbatim as submission.csv — even when models trained to 81%
            # val accuracy. The schema info the agent actually needs is
            # already in the prompt (`The file must have these columns in
            # order: ...`), so the file itself is redundant *and* a footgun.
            # sample_csv is still referenced below for the byte-identity
            # defensive check (in case the agent finds it via another path).
            session_id = uuid.uuid4().hex
            _log(f"sandbox session_id={session_id}  (uploading via {sandbox})")
            if _choose_staging(schema.modality) == "archive":
                from bench.staging import build_task_archive, stage_archive_to_sandbox
                BENCH_BUCKET = os.environ.get("BENCH_BUCKET", "impulse-mle-bench-runs")
                archive = work_dir / "task.tar.gz"
                build_task_archive(data_dir, archive, exclude={"sample_submission.csv"})
                _log(f"  staging task tree as {archive.name} ({archive.stat().st_size:,} bytes)")
                stage_archive_to_sandbox(sess, sandbox, session_id, archive, bucket=BENCH_BUCKET)
                _log("  archive staged; agent must extract task.tar.gz before training")
            else:
                _upload_to_sandbox(sess, sandbox, session_id, train_csv)
                _log(f"  uploaded train.csv ({train_csv.stat().st_size:,} bytes)")
                _upload_to_sandbox(sess, sandbox, session_id, test_csv)
                _log(f"  uploaded test.csv ({test_csv.stat().st_size:,} bytes)")
                _log(
                    f"  (sample_submission.csv NOT uploaded — schema is in the prompt; "
                    f"the file caused the agent to write it back verbatim as submission.csv)"
                )

            # 5. Construct messages — initial build + optional refinement turns.
            #    The shape mirrors what the production frontend at
            #    frontend/components/chat/chat-interface.tsx:1269 actually sends:
            #        { message, session_id, dataset_id, mode: "intelligence" }
            #    No description/intent/user_id — production omits them.
            messages = _build_messages(schema, self.refinement_turns)
            _log(f"plan: {len(messages)} chat turn(s) (1 build + {self.refinement_turns} refine)")

            # 6. POST /v2/chat for each turn, sharing session_id so the
            #    orchestrator has accumulated context across turns.
            #    Continue across turn failures — a real user, after seeing an
            #    error, types another message. Refinement turns can recover.
            #    We grade whatever the agent ends up producing (possibly only
            #    a partial submission from a failed turn).
            turns_attempted = 0
            turns_succeeded = 0
            for i, msg in enumerate(messages, 1):
                turns_attempted += 1
                _log(f"chat turn {i}/{len(messages)}: {msg[:80].strip()}…")
                chat_body = {
                    "message": msg,
                    "session_id": session_id,
                    "dataset_id": session_id,
                    "mode": "intelligence",
                    "modality": schema.modality,
                }
                try:
                    sse_resp = sess.post(
                        f"{url}/v2/chat",
                        json=chat_body,
                        stream=True,
                        timeout=(30, _remaining()),
                        headers={"Accept": "text/event-stream"},
                    )
                except Exception as e:
                    _log(f"  turn {i} HTTP error: {e} — trying next turn")
                    continue
                if not sse_resp.ok:
                    _log(
                        f"  turn {i} returned {sse_resp.status_code}: "
                        f"{sse_resp.text[:300]} — trying next turn"
                    )
                    continue
                done_seen, failed_seen, err_msg = _consume_sse(
                    sse_resp, _log, _remaining,
                )
                if failed_seen:
                    _log(f"  turn {i} errored ({err_msg}) — trying next turn")
                    continue
                if not done_seen:
                    _log(f"  turn {i} stream ended without DONE — trying next turn")
                    continue
                turns_succeeded += 1

            _log(f"turns: {turns_succeeded}/{turns_attempted} succeeded")

            # 8. Locate submission.csv. Try agent-api's DB-backed artifact list
            # first; fall back to listing files in the sandbox directly (the
            # agent's save_artifact may write to disk but not always persist a
            # DB row).
            _log(f"fetching artifacts for session {session_id}")
            artifacts = _list_session_artifacts(sess, url, session_id, _log)
            if not artifacts:
                _log("  agent-api DB has no artifacts — listing sandbox files instead")
                artifacts = _list_sandbox_files(sess, sandbox, session_id)
            _log(f"  files: {sorted([a.get('filename') for a in artifacts])}")

            sub_art = _find_submission_artifact(artifacts)
            if sub_art is None:
                if turns_succeeded == 0:
                    raise RuntimeError(
                        f"no chat turn completed and no submission was produced "
                        f"({turns_attempted} turn(s) attempted; see live log for "
                        f"the agent's error trace). Files in session: "
                        f"{[a.get('filename') for a in artifacts]}"
                    )
                raise RuntimeError(
                    f"the agent ran but produced no submission.csv. "
                    f"Files in session: {[a.get('filename') for a in artifacts]}"
                )

            # Resolve download URL (agent-api signed URL → fallback to direct
            # sandbox fetch by filename).
            signed_url = sub_art.get("signed_url") or ""
            filename = sub_art.get("filename")
            if signed_url:
                if signed_url.startswith("/api/"):
                    # /api/v2/... is the gateway-prefixed form; strip /api for
                    # direct agent-api access.
                    signed_url = f"{url}{signed_url[4:]}"
                elif signed_url.startswith("/"):
                    signed_url = f"{url}{signed_url}"
            else:
                signed_url = f"{sandbox}/sessions/{session_id}/files/{filename}"

            _log(f"downloading {filename} from {signed_url[:80]}")
            dl = sess.get(signed_url, timeout=300)
            if not dl.ok:
                raise RuntimeError(
                    f"download {filename} failed {dl.status_code}: {dl.text[:300]}"
                )

            # Tolerantly read the sample-submission bytes for the
            # byte-identity guard; a miss is non-fatal (sample_csv may be
            # None or unreadable).
            try:
                sample_bytes = sample_csv.read_bytes()
            except Exception:
                sample_bytes = b""
            check_not_sample_submission(
                dl.content, sample_bytes, schema.target_cols, filename, _log,
            )

            submission_path = work_dir / "submission.csv"
            submission_path.write_bytes(dl.content)
            _log(f"submission saved: {submission_path} ({len(dl.content):,} bytes)")

            check_target_columns_populated(dl.content, schema.target_cols, _log)

            return AgentResult(
                success=True,
                submission_path=submission_path,
                wall_time_s=time.time() - t0,
                stdout="\n".join(log),
                stderr="\n".join(errlog),
            )

        except Exception as e:
            errlog.append(traceback.format_exc())
            return AgentResult(
                success=False,
                submission_path=None,
                wall_time_s=time.time() - t0,
                stdout="\n".join(log),
                stderr="\n".join(errlog),
                error=f"ImpulseAgent.run raised {type(e).__name__}: {e}",
            )


# ---------- tree-search adapter ----------------------------------------------

#: Auto-answer for ASKING events — keeps the run unattended.
_TREE_ASK_RESPONSE = "Proceed with your best judgment. Do not ask further questions."
#: One-shot steer message when the run record parks in awaiting_user_input.
_TREE_STEER_MESSAGE = "Proceed — train and produce submission.csv."

_RUN_TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "timed_out"}

#: Hard ceiling for any single requests timeout derived from the deadline.
#: A single network call never legitimately needs a multi-hour timeout — the
#: poll/SSE loops re-issue. Bounding it means a host suspend that inflates a
#: naive `deadline - now` can't hand requests an absurd (or negative) value.
_MAX_CALL_TIMEOUT_S = 120.0

#: Dataset uploads are exempt from the 120s clamp above: a large vision/audio
#: archive can legitimately take minutes to stream. But the read bound is still
#: FINITE — it must never outlast the run's hard deadline (the watchdog can't
#: interrupt an in-flight upload), so the tree agent clamps it to the remaining
#: budget and never exceeds this ceiling (the old hardcoded value).
_MAX_UPLOAD_READ_TIMEOUT_S = 600.0
#: Connect phase is quick regardless of payload size; bound it tightly.
_UPLOAD_CONNECT_TIMEOUT_S = 30.0


def _clamp_call_timeout(value: float, lo: float = 1.0,
                        hi: float = _MAX_CALL_TIMEOUT_S) -> float:
    """Clamp a deadline-derived timeout into a sane [lo, hi] window."""
    if value != value:  # NaN guard
        return lo
    return max(lo, min(hi, value))

#: Env knobs snapshotted into meta for reproducibility.
_TREE_META_ENV_KEYS = (
    "BENCH_AGENT_API_URL",
    "BENCH_SANDBOX_URL",
    "TREE_SEARCH_MAX_NODES",
    "TREE_SEARCH_TURNS_PER_NODE",
    "TREE_SEARCH_MAX_DEPTH",
    "ENABLE_DYNAMIC_TREE",
    "ENABLE_EES_TOOLS",
)


class _TreeRunState:
    """Shared state between the SSE consumer thread(s) and the poll loop."""

    def __init__(self) -> None:
        self.done = False                       # terminal SSE event seen
        self.error_msg: str | None = None       # last error-event content
        self.tree_milestone: dict | None = None  # tree_search summary data
        self.asking_responses = 0
        self.sse_reconnects = 0
        self.stop = threading.Event()           # poll loop says: run is over
        # Hard watchdog fired: monotonic elapsed exceeded timeout_s+grace.
        # The poll loop and salvage steps short-circuit when this is set.
        self.hard_deadline_hit = threading.Event()
        # Guards counter increments — the original worker and a steer-turn
        # thread can write concurrently.
        self.lock = threading.Lock()


@dataclass
class ImpulseTreeAgent:
    """Drive agent-api's PRODUCTION intelligence path over HTTP.

    Routes through routes/chat.py → Orchestrator(mode="intelligence") →
    SolutionTreeRunner — the real tree-search brain, not the SDK/legacy lane.
    Requires agent-api started with USE_SDK_MODE unset/false.

    Flow:
      1. POST /v2/datasets/upload train.csv (auto-creates session), then
         test.csv with role="inference" into the same session.
      2. POST /v2/chat with a tree-safe message (see _build_tree_message):
         NO model-family keywords (orchestrator.py flips those runs to
         direct execution), ≥1 explore signal (forces tree search ON).
      3. A background thread consumes the SSE stream: captures the
         tree_search MILESTONE summary, auto-answers ASKING events via
         POST /v2/chat/{id}/respond, and reconnects through
         GET /v2/sessions/{id}/events if the socket drops mid-run.
      4. The main thread long-polls GET /v2/sessions/{id}/run until a
         terminal status. awaiting_user_input triggers one bounded steer
         turn; hitting the deadline POSTs /v2/sessions/{id}/cancel.
      5. Fetch submission.csv via /artifacts/urls signed URLs, falling back
         to the sandbox file listing — even for failed runs (a gradable
         submission is the bench's success criterion).

    Every failure path returns AgentResult(success=False, error=..., meta=…)
    — run() never raises.
    """

    agent_api_url: str = field(
        default_factory=lambda: os.environ.get("BENCH_AGENT_API_URL", "http://localhost:8005")
    )
    sandbox_url: str = field(
        default_factory=lambda: os.environ.get("BENCH_SANDBOX_URL", "http://localhost:8006")
    )
    user_id: str = "bench-tree"
    # Max follow-up /v2/chat turns sent when the run parks in awaiting_user_input.
    max_auto_replies: int = 2
    # Seconds between run-record polls (the poll itself long-polls up to 60s).
    poll_interval_s: float = 5.0
    # Bound on /events reconnect attempts after stream drops.
    max_sse_reconnects: int = 20
    # Hard watchdog grace: a daemon thread forces run() to return once
    # monotonic elapsed exceeds timeout_s + watchdog_grace_s. This is the
    # liveness guarantee against a wedged network call or a host that
    # suspended mid-run (which would otherwise let an in-flight long-poll
    # block indefinitely while the deadline check never runs).
    watchdog_grace_s: int = 900
    # Total wall budget (monotonic) for the post-deadline salvage phase
    # (artifact list + submission download). Bounds the otherwise-blocking
    # requests calls so the watchdog window isn't blown by a wedged fetch.
    salvage_budget_s: int = 120
    log_callback: Callable[[str], None] | None = None

    @property
    def name(self) -> str:
        return "impulse:tree"

    # ---- SSE consumption (worker thread) ----------------------------------

    def _consume_tree_sse(self, resp, state: _TreeRunState, url: str,
                          session_id: str, log_fn, deadline: float,
                          is_reconnect: bool = False) -> None:
        """Drain one SSE response. Updates `state`; never raises.

        `deadline` is a ``time.monotonic()`` value — monotonic does not jump
        when the host suspends, so a suspend can't silently push this past.

        Once the poll loop sets ``state.stop`` (run terminal / cancelled),
        exit instead of riding out heartbeats until the deadline: a live
        stream (chat / steer turn) breaks immediately, while a reconnected
        /events stream first drains its replay prefix — the final
        tree_search milestone may only exist in the replay buffer — and
        breaks once the replay_done marker has been seen.
        """
        import requests

        replay_done_seen = False
        try:
            for raw in resp.iter_lines():
                if time.monotonic() >= deadline or state.hard_deadline_hit.is_set():
                    break
                if state.stop.is_set() and (replay_done_seen or not is_reconnect):
                    break
                if not raw or not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].decode("utf-8", errors="replace").strip()
                if payload == "[DONE]":
                    state.done = True
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                et = (ev.get("type") or "").lower()
                if et == "replay_done":
                    replay_done_seen = True
                    if state.stop.is_set():
                        break  # replay drained and the run is over — exit now
                    continue
                if et == "heartbeat":
                    continue
                content = (ev.get("content") or "").strip()
                if et == "milestone" and ev.get("agent") == "tree_search":
                    data = ev.get("data") or {}
                    # The final summary carries "status"; keep the richest one.
                    if state.tree_milestone is None or "status" in data:
                        state.tree_milestone = dict(data)
                    log_fn(f"  [tree] {content[:140]}")
                    continue
                if et == "asking":
                    with state.lock:
                        state.asking_responses += 1
                    log_fn(f"  [asking] {content[:120]} — auto-responding")
                    try:
                        requests.post(
                            f"{url}/v2/chat/{session_id}/respond",
                            json={"response": _TREE_ASK_RESPONSE},
                            timeout=30,
                        )
                    except Exception as e:
                        log_fn(f"  auto-respond failed: {e}")
                    continue
                if et in ("done", "completed", "cancelled"):
                    state.done = True
                    log_fn(f"  [done] {content[:140]}")
                    break
                if et in ("error", "failed"):
                    # Diagnostic only — the run record is the terminal authority.
                    state.error_msg = content or json.dumps(ev.get("data") or {})[:300]
                    log_fn(f"  [err] {state.error_msg[:200]}")
                    continue
                short = (content[:140] + "…") if len(content) > 140 else content
                log_fn(f"  [{et}] {short}" if short else f"  [{et}]")
        except Exception as e:
            log_fn(f"  SSE stream interrupted: {e}")
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def _sse_worker(self, first_resp, state: _TreeRunState, url: str,
                    session_id: str, log_fn, deadline: float) -> None:
        """Consume the chat stream; on drop, reconnect via /events.

        Always allows at least one reconnect even after the poll loop flags
        the run as over — when the original socket dropped, the final
        tree_search milestone only exists in the replay buffer.
        """
        import requests

        resp = first_resp
        is_reconnect = False
        while True:
            self._consume_tree_sse(resp, state, url, session_id, log_fn, deadline,
                                   is_reconnect=is_reconnect)
            if state.done or time.monotonic() >= deadline or state.hard_deadline_hit.is_set():
                break
            if state.sse_reconnects >= self.max_sse_reconnects:
                log_fn(f"  giving up after {state.sse_reconnects} SSE reconnects")
                break
            if state.stop.is_set() and state.sse_reconnects >= 1:
                break
            try:
                resp = requests.get(
                    f"{url}/v2/sessions/{session_id}/events",
                    stream=True,
                    headers={"Accept": "text/event-stream"},
                    # Clamp the read timeout: a single network call should
                    # never need a multi-hour timeout (the worker re-polls).
                    timeout=(10, _clamp_call_timeout(deadline - time.monotonic())),
                )
            except Exception as e:
                log_fn(f"  /events reconnect failed: {e}")
                break
            if not resp.ok:
                log_fn(f"  /events reconnect returned {resp.status_code}")
                try:
                    resp.close()
                except Exception:
                    pass
                break
            with state.lock:
                state.sse_reconnects += 1
            is_reconnect = True
            log_fn(f"  reconnected via /events (attempt {state.sse_reconnects})")

    # ---- main flow ---------------------------------------------------------

    def run(self, data_dir: Path, work_dir: Path, timeout_s: int) -> AgentResult:
        import requests

        from .adapters.task_detection import _find_sample_submission, detect_schema

        work_dir.mkdir(parents=True, exist_ok=True)
        # MONOTONIC clock for ALL elapsed/deadline math. time.time() advances
        # through a host suspend (it inflated a real run's wall to ~20h);
        # monotonic does not jump forward on resume, so wall_time_s and the
        # deadline stay sane across a suspend. t0_wall is kept only for
        # wall-clock logging — never for elapsed math.
        t0_mono = time.monotonic()
        t0_wall = time.time()
        deadline = t0_mono + timeout_s
        # Hard deadline: even if a network call wedges or the host suspended,
        # the watchdog guarantees run() returns by here (monotonic).
        hard_deadline = t0_mono + timeout_s + self.watchdog_grace_s
        log: list[str] = []
        errlog: list[str] = []
        meta: dict = {"agent": self.name, "created_at": t0_wall}

        def _log(msg: str) -> None:
            log.append(msg)
            print(f"    {msg}")
            if self.log_callback:
                try:
                    self.log_callback(msg)
                except Exception:
                    pass  # logger failures must never kill the agent loop

        url = self.agent_api_url.rstrip("/")
        sandbox = self.sandbox_url.rstrip("/")
        state = _TreeRunState()
        session_id: str | None = None
        run_rec: dict | None = None
        timed_out = False
        auto_replies = 0
        sse_threads: list[threading.Thread] = []

        def _mono_elapsed() -> float:
            return time.monotonic() - t0_mono

        def _call_timeout(extra: float = 30.0) -> float:
            """Per-call read timeout, clamped so no single blocking request can
            run past the hard deadline. As the hard deadline nears, this
            shrinks toward zero — so a wedged socket can't blow the watchdog
            bound (the call itself times out around the same time)."""
            to_hard = hard_deadline - time.monotonic()
            return _clamp_call_timeout(min(to_hard + extra, _MAX_CALL_TIMEOUT_S))

        # ---- hard watchdog (daemon thread) --------------------------------
        # Liveness guarantee: even if a blocking network call is wedged or the
        # host suspended mid-run, this thread fires once monotonic elapsed
        # exceeds timeout_s + grace, flagging the run over so the poll loop and
        # salvage steps short-circuit. Combined with the clamped per-call
        # timeouts above, run() returns within hard_deadline + small slack.
        # Short, fixed poll so the thread reacts to `state.stop` within ~0.25s
        # (it must exit promptly on normal completion — no lingering thread).
        _WATCHDOG_TICK_S = 0.25

        def _watchdog() -> None:
            # state.stop doubles as the wake signal: Event.wait returns
            # immediately once run() sets stop in its finally block.
            while not state.stop.wait(_WATCHDOG_TICK_S):
                if _mono_elapsed() > timeout_s + self.watchdog_grace_s:
                    state.hard_deadline_hit.set()
                    state.stop.set()
                    return

        watchdog_thread = threading.Thread(
            target=_watchdog, daemon=True, name="bench-tree-watchdog",
        )
        watchdog_thread.start()

        def _capture_meta() -> None:
            meta["tree"] = state.tree_milestone
            meta["asking_responses"] = state.asking_responses
            meta["sse_reconnects"] = state.sse_reconnects
            meta["auto_replies_sent"] = auto_replies
            meta["timed_out"] = timed_out
            if run_rec is not None:
                meta["run_status"] = run_rec.get("status")
                meta["run"] = {
                    k: run_rec.get(k)
                    for k in ("status", "error_message", "result_summary",
                              "metadata", "artifact_ids")
                }
            meta["env"] = {k: os.environ.get(k) for k in _TREE_META_ENV_KEYS}
            meta["bench_git_sha"] = _bench_git_sha()
            # When the watchdog has already fired, skip the network meta fetch
            # entirely: these two GETs are unclamped (30s each) and would stretch
            # the force-return by up to ~60s — the very hang the watchdog exists
            # to bound. The in-memory meta above is enough on that path.
            if state.hard_deadline_hit.is_set():
                return
            if session_id:
                for key, endpoint in (
                    ("budget", f"{url}/v2/sessions/{session_id}/budget"),
                    ("api_meta", f"{url}/v2/meta"),
                ):
                    try:
                        r = requests.get(endpoint, timeout=30)
                        if r.ok:
                            meta[key] = r.json()
                    except Exception:
                        pass

        try:
            # 1. Schema detection (same path as ImpulseAgent).
            _log(f"detecting schema in {data_dir}")
            schema = detect_schema(data_dir)
            _log(
                f"  id_col={schema.id_col}  targets={schema.target_cols}  "
                f"metric={schema.metric or '<unknown>'}  modality={schema.modality}"
            )
            meta["schema"] = {
                "id_col": schema.id_col,
                "target_cols": schema.target_cols,
                "metric": schema.metric,
                "modality": schema.modality,
            }

            _sample = _find_sample_submission(data_dir)
            sample_csv = _sample.resolve() if _sample else None

            # The upload read timeout is generous (large vision/audio archives
            # stream slowly) but FINITE — bounded by the remaining hard-deadline
            # budget so a wedged upload can't park run()'s main thread past the
            # watchdog window (the watchdog can't interrupt an in-flight C-level
            # requests call).
            def _upload_read_timeout() -> float:
                rem = hard_deadline - time.monotonic()
                # Floor at a couple seconds so a tiny remaining budget doesn't
                # make a legitimate upload fail instantly; never exceed the old
                # 600s ceiling, and never outlast the hard deadline.
                return max(2.0, min(rem, _MAX_UPLOAD_READ_TIMEOUT_S))

            # 2. Stage data per the task's modality (see _choose_staging):
            #    - tabular/nlp ("csv"): the run REQUIRES train.csv + test.csv;
            #      upload them via /v2/datasets/upload (NLP is text-in-CSV).
            #    - vision/audio ("archive"): there is no usable train.csv/test.csv
            #      (whale ships only zips + a sample submission). Pack the whole
            #      task tree — nested zips pre-extracted into media subdirs — as
            #      task.tar.gz and stage it into the sandbox workspace; the media
            #      preamble extracts it before training.
            if _choose_staging(schema.modality) == "archive":
                session_id = uuid.uuid4().hex
                # The tree agent drives requests at module level (no Session);
                # the requests module itself exposes .put/.post, so pass it as
                # the "session" used by the staging helpers.
                self._stage_archive(
                    sess=requests, requests_mod=requests, url=url, sandbox=sandbox,
                    session_id=session_id, data_dir=data_dir, work_dir=work_dir,
                    log_fn=_log,
                )
                # SDK-mode reads files from the sandbox workspace keyed by
                # session_id; there are no dataset rows for the archive path, so
                # dataset_id/test_dataset_id mirror the session_id (as the
                # non-tree ImpulseAgent does for archive tasks).
                train_dataset_id = session_id
                test_dataset_id = session_id
            else:
                train_csv = (data_dir / "train.csv").resolve()
                test_csv = (data_dir / "test.csv").resolve()
                for p in (train_csv, test_csv):
                    if not p.exists():
                        raise FileNotFoundError(f"{p.name} not in {data_dir}")
                # Upload train (auto-creates the session), then test as the
                # held-out inference table in the SAME session.
                train_upload = self._upload(requests, url, train_csv, _log,
                                            user_id=self.user_id,
                                            read_timeout=_upload_read_timeout())
                session_id = train_upload["session_id"] or uuid.uuid4().hex
                train_dataset_id = train_upload["dataset_id"]
                test_upload = self._upload(requests, url, test_csv, _log,
                                           session_id=session_id, role="inference",
                                           read_timeout=_upload_read_timeout())
                test_dataset_id = test_upload["dataset_id"]

            meta["session_id"] = session_id
            meta["dataset_id"] = train_dataset_id
            meta["test_dataset_id"] = test_dataset_id
            _log(f"session {session_id}: train={train_dataset_id} test={test_dataset_id}")

            # 3. Kick off the tree run.
            message = _build_tree_message(schema)
            chat_body = {
                "message": message,
                "session_id": session_id,
                "dataset_id": train_dataset_id,
                "test_dataset_id": test_dataset_id,
                "mode": "intelligence",
                "user_id": self.user_id,
                "description": (schema.description or "").strip()[:2000] or None,
                "modality": schema.modality,
            }
            _log("POST /v2/chat (mode=intelligence — production tree path)")
            sse_resp = requests.post(
                f"{url}/v2/chat",
                json=chat_body,
                stream=True,
                timeout=(30, _call_timeout()),
                headers={"Accept": "text/event-stream"},
            )
            if not sse_resp.ok:
                err_preview = sse_resp.text[:300]
                sse_resp.close()
                raise RuntimeError(
                    f"/v2/chat returned {sse_resp.status_code}: {err_preview}"
                )
            sse_thread = threading.Thread(
                target=self._sse_worker,
                args=(sse_resp, state, url, session_id, _log, deadline),
                daemon=True,
                name="bench-tree-sse-worker",
            )
            sse_thread.start()
            sse_threads.append(sse_thread)

            # 4. Long-poll the durable run record until terminal or deadline.
            while True:
                if state.hard_deadline_hit.is_set():
                    timed_out = True
                    break
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    break
                remaining = deadline - now
                # Long-poll window and per-call timeout are both clamped so a
                # wedged socket (or a host that suspended mid-poll) can't block
                # past the hard deadline — _call_timeout() shrinks as it nears.
                wait_window = min(60.0, max(1.0, remaining))
                try:
                    rr = requests.get(
                        f"{url}/v2/sessions/{session_id}/run",
                        params={
                            "wait_for_terminal": "true",
                            "max_wait_seconds": wait_window,
                        },
                        timeout=_call_timeout(extra=wait_window),
                    )
                except Exception as e:
                    _log(f"  run poll error: {e}")
                    time.sleep(self.poll_interval_s)
                    continue
                if rr.status_code == 404:
                    # Run record not written yet — keep polling.
                    time.sleep(self.poll_interval_s)
                    continue
                if not rr.ok:
                    _log(f"  run poll returned {rr.status_code}")
                    time.sleep(self.poll_interval_s)
                    continue
                run_rec = rr.json()
                status = (run_rec.get("status") or "").lower()
                if status in _RUN_TERMINAL_STATUSES:
                    _log(f"run terminal: {status}")
                    break
                if status == "awaiting_user_input":
                    if auto_replies < self.max_auto_replies:
                        auto_replies += 1
                        _log(
                            f"run awaiting user input — steer turn "
                            f"{auto_replies}/{self.max_auto_replies}"
                        )
                        steer_body = dict(
                            chat_body,
                            message=_TREE_STEER_MESSAGE,
                            steer=True,
                            description=None,
                        )
                        try:
                            steer_resp = requests.post(
                                f"{url}/v2/chat",
                                json=steer_body,
                                stream=True,
                                timeout=(30, _call_timeout()),
                                headers={"Accept": "text/event-stream"},
                            )
                            if steer_resp.ok:
                                steer_thread = threading.Thread(
                                    target=self._consume_tree_sse,
                                    args=(steer_resp, state, url, session_id,
                                          _log, deadline),
                                    daemon=True,
                                    name="bench-tree-sse-steer",
                                )
                                steer_thread.start()
                                sse_threads.append(steer_thread)
                            else:
                                _log(f"  steer turn returned {steer_resp.status_code}")
                                steer_resp.close()
                        except Exception as e:
                            _log(f"  steer turn failed: {e}")
                    else:
                        _log(
                            "run still awaiting user input and auto-reply budget "
                            "exhausted — proceeding to artifact fetch"
                        )
                        break
                time.sleep(self.poll_interval_s)

            # Hard watchdog fired: a network call was wedged (or the host
            # suspended) past timeout_s+grace. Return promptly with a clear
            # error — do NOT enter the blocking salvage phase, which could
            # extend the hang. This is the load-bearing liveness guarantee.
            if state.hard_deadline_hit.is_set():
                state.stop.set()
                for th in sse_threads:
                    th.join(timeout=2)
                try:
                    _capture_meta()
                except Exception:
                    pass
                elapsed = _mono_elapsed()
                return AgentResult(
                    success=False, submission_path=None,
                    wall_time_s=elapsed,
                    stdout="\n".join(log), stderr="\n".join(errlog),
                    error=(
                        f"hard watchdog fired after {elapsed:.1f}s "
                        f"(timeout_s={timeout_s} + grace={self.watchdog_grace_s}) — "
                        f"a network call wedged or the host suspended mid-run; "
                        f"run() force-returned to protect the sweep"
                    ),
                    meta=meta,
                )

            if timed_out:
                _log(f"deadline reached after {timeout_s}s — cancelling session")
                try:
                    requests.post(
                        f"{url}/v2/sessions/{session_id}/cancel", timeout=30
                    )
                except Exception as e:
                    _log(f"  cancel failed: {e}")

            state.stop.set()
            for th in sse_threads:
                th.join(timeout=10)
            _capture_meta()

            # 5. Locate submission.csv — signed URLs first, sandbox fallback.
            #    Runs even for failed/timed-out runs: salvage beats discard.
            #    Bounded by a monotonic salvage budget so a wedged artifact
            #    list / download can't blow the watchdog window.
            salvage_deadline = time.monotonic() + self.salvage_budget_s
            content, filename, source = self._fetch_submission(
                requests, url, sandbox, session_id, _log,
                salvage_deadline=salvage_deadline,
            )
            if content is None:
                err = self._diagnostic_error(run_rec, state, timed_out, timeout_s)
                return AgentResult(
                    success=False, submission_path=None,
                    wall_time_s=_mono_elapsed(),
                    stdout="\n".join(log), stderr="\n".join(errlog),
                    error=err, meta=meta,
                )
            meta["submission_source"] = source
            if run_rec and (run_rec.get("status") or "") != "succeeded":
                _log(
                    f"salvaged {filename} from a {run_rec.get('status')} run — "
                    f"a gradable submission is the bench's success criterion"
                )

            # 6. Validate + (best-effort) reshape to the sample schema, write.
            try:
                sample_bytes = sample_csv.read_bytes() if sample_csv else b""
            except Exception:
                sample_bytes = b""
            check_not_sample_submission(
                content, sample_bytes, schema.target_cols, filename, _log
            )
            submission_path = _write_submission(
                content, work_dir, sample_csv, schema, _log
            )
            check_target_columns_populated(
                submission_path.read_bytes(), schema.target_cols, _log
            )
            _log(f"submission saved: {submission_path}")

            return AgentResult(
                success=True,
                submission_path=submission_path,
                wall_time_s=_mono_elapsed(),
                stdout="\n".join(log),
                stderr="\n".join(errlog),
                meta=meta,
            )

        except Exception as e:
            errlog.append(traceback.format_exc())
            state.stop.set()
            try:
                _capture_meta()
            except Exception:
                pass
            return AgentResult(
                success=False,
                submission_path=None,
                wall_time_s=_mono_elapsed(),
                stdout="\n".join(log),
                stderr="\n".join(errlog),
                error=f"ImpulseTreeAgent.run raised {type(e).__name__}: {e}",
                meta=meta,
            )
        finally:
            # Always release the watchdog (stop set ⇒ it wakes immediately)
            # and join it so no daemon thread outlives run().
            state.stop.set()
            try:
                watchdog_thread.join(timeout=_WATCHDOG_TICK_S * 8)
            except Exception:
                pass

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _stage_archive(sess, requests_mod, url: str, sandbox: str,
                       session_id: str, data_dir: Path, work_dir: Path,
                       log_fn) -> None:
        """Stage a vision/audio task tree into the sandbox as task.tar.gz.

        Builds the archive with nested zips PRE-EXTRACTED (so the sandbox media
        preamble sees ready image/audio subdirs + label CSVs at root), then
        uploads it.

        Transport choice (documented):
          - DIRECT (default): PUT the tar.gz straight to the sandbox's raw-bytes
            file endpoint. This is the only transport that works locally — GCS
            staging needs a project + bucket + credentials a dev box lacks.
          - GCS: used ONLY when ``BENCH_BUCKET`` is set AND a GCS client
            constructs successfully (try/except). Production streams large
            archives through GCS rather than inter-service HTTP.
        sample_submission.csv is excluded from the archive — empirically the
        agent treats it as a ready-made template and writes it back verbatim.
        """
        from bench.staging import (
            _gcs_client,
            build_task_archive_extracted,
            stage_archive_direct,
            stage_archive_to_sandbox,
        )

        archive = work_dir / "task.tar.gz"
        build_task_archive_extracted(
            data_dir, archive,
            exclude={"sample_submission.csv", "sampleSubmission.csv"},
            work_dir=work_dir,
        )
        log_fn(
            f"  staged task tree (nested zips pre-extracted) as {archive.name} "
            f"({archive.stat().st_size:,} bytes)"
        )

        bucket = os.environ.get("BENCH_BUCKET")
        use_gcs = False
        if bucket:
            try:
                _gcs_client()  # constructs only with a project/credentials
                use_gcs = True
            except Exception as e:
                log_fn(f"  BENCH_BUCKET set but GCS client failed ({e}) — using direct PUT")

        if use_gcs:
            stage_archive_to_sandbox(sess, sandbox, session_id, archive, bucket=bucket)
            log_fn(f"  archive staged via GCS bucket {bucket}; preamble extracts it before training")
        else:
            stage_archive_direct(sess, sandbox, session_id, archive)
            log_fn("  archive staged via direct sandbox PUT; preamble extracts it before training")

    @staticmethod
    def _upload(requests_mod, url: str, file_path: Path, log_fn,
                session_id: str | None = None, user_id: str | None = None,
                role: str | None = None,
                read_timeout: float = _MAX_UPLOAD_READ_TIMEOUT_S) -> dict:
        """POST /v2/datasets/upload. Returns {dataset_id, session_id|None}.

        Uses a (connect, read) timeout tuple: a short connect bound plus a
        FINITE read bound. Uploads run BEFORE the poll loop, so an unbounded
        read here would park run()'s main thread out of the watchdog's reach
        (the watchdog only sets flags — it can't interrupt an in-flight
        C-level requests call). ``read_timeout`` defaults to the legacy 600s
        so other callers keep their old behavior; the tree agent passes a
        value clamped to its remaining hard-deadline budget so the upload can
        never outlast the watchdog.
        """
        data: dict[str, str] = {}
        if session_id:
            data["session_id"] = session_id
        if user_id:
            data["user_id"] = user_id
        if role:
            data["role"] = role
        with file_path.open("rb") as fh:
            resp = requests_mod.post(
                f"{url}/v2/datasets/upload",
                files={"file": (file_path.name, fh, "text/csv")},
                data=data,
                timeout=(_UPLOAD_CONNECT_TIMEOUT_S, read_timeout),
            )
        if not resp.ok:
            raise RuntimeError(
                f"upload {file_path.name} → {resp.status_code}: {resp.text[:300]}"
            )
        body = resp.json()
        log_fn(f"  uploaded {file_path.name} → dataset {body.get('dataset_id')}")
        return {
            "dataset_id": body.get("dataset_id") or body.get("id"),
            "session_id": body.get("session_id"),
        }

    def _fetch_submission(self, requests_mod, url: str, sandbox: str,
                          session_id: str, log_fn,
                          salvage_deadline: float | None = None,
                          ) -> tuple[bytes | None, str | None, str | None]:
        """Locate + download the submission. Returns (bytes, filename, source).

        `salvage_deadline` (a ``time.monotonic()`` value) caps the TOTAL wall
        of this otherwise-blocking phase: each list/download call is clamped to
        the remaining budget, and once it's exhausted we bail rather than start
        another multi-minute call. Without it a single wedged artifact list (3×
        up to 180s) or download (up to 300s) could blow the watchdog window.

        ORDER (matters — a wedged sandbox-dependent call must not starve the
        cheap one that already has the answer):
          a) ``/artifacts`` — the plain DB list. It returns submission.csv with
             a proxy ``signed_url`` and does NOT touch the (possibly wedged)
             sandbox. Try this FIRST.
          b) ``/artifacts/urls`` — signed URLs (proxies the sandbox /signed-urls;
             can hang).
          c) sandbox ``/files`` listing — last resort, fully sandbox-dependent.

        PER-CALL CAP: no single list/download may consume more than
        ``min(25s, salvage_budget/3)`` so a hang in one path can't eat the whole
        budget and starve the others. The NLP run failed exactly that way: the
        submission existed and was DB-registered, but (b) hung and (a)/(c) were
        skipped.
        """
        # Per-call cap derived from the salvage budget. When no deadline is
        # supplied (callers that don't bound salvage), fall back to the legacy
        # 25s ceiling so a single call still can't run away.
        if salvage_deadline is not None:
            budget = max(0.0, salvage_deadline - time.monotonic())
        else:
            budget = float(self.salvage_budget_s)
        per_call_cap = min(25.0, budget / 3.0) if budget > 0 else 1.0

        def _budget(default: float) -> float:
            capped = min(default, per_call_cap)
            if salvage_deadline is None:
                return _clamp_call_timeout(capped)
            rem = salvage_deadline - time.monotonic()
            return _clamp_call_timeout(min(capped, rem))

        def _expired() -> bool:
            return salvage_deadline is not None and time.monotonic() >= salvage_deadline

        # a) /artifacts — plain DB list FIRST (returns submission.csv with a
        #    proxy signed_url; never touches the sandbox).
        artifacts = _list_session_artifacts(
            requests_mod, url, session_id, log_fn,
            deadline=salvage_deadline, per_call_cap=per_call_cap,
        )
        sub = _find_submission_artifact(artifacts)

        # b) /artifacts/urls — signed URLs (may proxy a wedged sandbox).
        if sub is None and not _expired():
            urls: list[dict] = []
            try:
                r = requests_mod.get(
                    f"{url}/v2/sessions/{session_id}/artifacts/urls",
                    timeout=_budget(_LIST_TIMEOUT_S),
                )
                if r.ok:
                    body = r.json()
                    urls = body.get("artifacts") if isinstance(body, dict) else body
                    urls = urls or []
            except Exception as e:
                log_fn(f"  artifacts/urls fetch failed: {e}")
            sub = _find_submission_artifact(urls)

        if sub is not None:
            filename = sub.get("filename")
            dl_url = _resolve_artifact_url(url, sandbox, session_id, sub)
            log_fn(f"  downloading {filename} from {dl_url[:100]}")
            try:
                dl = requests_mod.get(dl_url, timeout=_budget(300))
                if dl.ok:
                    return dl.content, filename, "artifact_url"
                log_fn(f"  download returned {dl.status_code}")
            except Exception as e:
                log_fn(f"  download failed: {e}")

        if _expired():
            log_fn("  salvage budget exhausted — skipping sandbox fallback")
            return None, None, None

        # c) Sandbox file listing fallback.
        log_fn("  no artifact-backed submission — listing sandbox files")
        try:
            files = _list_sandbox_files(requests_mod, sandbox, session_id,
                                        deadline=salvage_deadline,
                                        per_call_cap=per_call_cap)
        except Exception as e:
            log_fn(f"  sandbox listing failed: {e}")
            files = []
        log_fn(f"  sandbox files: {sorted(f.get('filename') for f in files)}")
        sub = _find_submission_artifact(files)
        if sub is None:
            return None, None, None
        filename = sub.get("filename")
        try:
            dl = requests_mod.get(
                f"{sandbox}/sessions/{session_id}/files/{filename}",
                timeout=_budget(300),
            )
            if dl.ok:
                return dl.content, filename, "sandbox"
            log_fn(f"  sandbox download returned {dl.status_code}")
        except Exception as e:
            log_fn(f"  sandbox download failed: {e}")
        return None, None, None

    @staticmethod
    def _diagnostic_error(run_rec: dict | None, state: _TreeRunState,
                          timed_out: bool, timeout_s: int) -> str:
        if timed_out:
            return (
                f"timed out after {timeout_s}s waiting for the tree run; "
                f"session cancel requested; no submission.csv was produced"
            )
        if run_rec is not None:
            status = run_rec.get("status")
            detail = (
                run_rec.get("error_message")
                or run_rec.get("result_summary")
                or state.error_msg
                or "no diagnostic recorded"
            )
            return f"run {status} without a submission.csv: {detail}"
        if state.error_msg:
            return f"run errored without a submission.csv: {state.error_msg}"
        return "run finished but no submission.csv was found in artifacts or sandbox"


def _resolve_artifact_url(agent_api_url: str, sandbox_url: str,
                          session_id: str, artifact: dict) -> str:
    """Resolve an artifact record to a fetchable URL.

    Handles the three URL keys the endpoints emit (signed_url / url /
    download_url) and the gateway-prefix quirk: '/api/v2/...' must be
    rewritten to '<agent_api_url>/v2/...' for direct agent-api access.
    """
    raw = (
        artifact.get("signed_url")
        or artifact.get("url")
        or artifact.get("download_url")
        or ""
    )
    if raw:
        if raw.startswith("/api/"):
            return f"{agent_api_url}{raw[4:]}"
        if raw.startswith("/"):
            return f"{agent_api_url}{raw}"
        return raw
    return f"{sandbox_url}/sessions/{session_id}/files/{artifact.get('filename')}"


def _write_submission(content: bytes, work_dir: Path, sample_csv: Path | None,
                      schema, log_fn) -> Path:
    """Write submission.csv, reshaping to the sample schema when needed.

    If the downloaded CSV's columns already match sample_submission's, write
    it verbatim. Otherwise route through format_submission (id-merge /
    positional alignment + prediction-column mapping). Reshape failures fall
    back to the raw bytes — the grader gives a precise rejection there.
    """
    submission_path = work_dir / "submission.csv"
    try:
        import io

        import pandas as pd

        cols = list(pd.read_csv(io.BytesIO(content), nrows=1).columns)
        if cols != schema.sample_submission_cols and sample_csv is not None:
            from .adapters.submission_format import format_submission

            log_fn(
                f"  reshaping columns {cols} → {schema.sample_submission_cols}"
            )
            raw_path = work_dir / "raw_predictions.csv"
            raw_path.write_bytes(content)
            format_submission(raw_path, sample_csv, submission_path, schema)
            return submission_path
    except Exception as e:
        log_fn(f"  reshape skipped ({e}) — writing raw bytes")
    submission_path.write_bytes(content)
    return submission_path


_BENCH_GIT_SHA_CACHE: str | None = None


def _bench_git_sha() -> str:
    """Best-effort git SHA of the bench harness code (cached)."""
    global _BENCH_GIT_SHA_CACHE
    if _BENCH_GIT_SHA_CACHE is None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=str(Path(__file__).resolve().parent),
            )
            sha = proc.stdout.strip()
            _BENCH_GIT_SHA_CACHE = sha if proc.returncode == 0 and sha else "unknown"
        except Exception:
            _BENCH_GIT_SHA_CACHE = "unknown"
    return _BENCH_GIT_SHA_CACHE


def _build_tree_message(schema) -> str:
    """Build the /v2/chat message that keeps the orchestrator on the tree path.

    Two hard constraints (verified against agent-api/orchestrator.py
    ~6460-6507):
      - MUST NOT name a model family ("lightgbm", "xgboost", "catboost",
        "random forest", "ridge", "lasso", "mlp", "neural network",
        "tabnet", "pca", "svd", "autoencoder") — any of those flips the run
        to DIRECT execution with tree_nodes=0.
      - MUST contain an explore signal ("best model", "best possible",
        "try everything", "explore", "find the best", "optimize") — these
        force tree search ON even if the RS plan looks narrow.
    """
    target_str = ", ".join(schema.target_cols)
    schema_str = ", ".join(schema.sample_submission_cols)

    parts: list[str] = []
    if schema.description:
        parts.append(schema.description.strip()[:8000])

    metric_line = (
        f"This task is graded on **{schema.metric}** — optimize and select "
        f"the winning approach against that metric specifically."
        if schema.metric else
        "Identify the evaluation metric from the task description above and "
        "optimize against it specifically."
    )

    modality = getattr(schema, "modality", "tabular")
    if modality in ("vision", "audio"):
        # Vision/audio tasks are archive-staged: there is no train.csv/test.csv.
        # Media files live in subdirectories of the workspace and any label
        # file sits at the workspace root. Keep the explore signal + the
        # no-model-family-keyword rule (the direct-execution trap applies to
        # every modality), but describe the data the way it's actually staged.
        kind = modality  # "vision" | "audio"
        parts.append(
            f"The task data has been staged and extracted into your workspace "
            f"({kind} files in subdirectories; any label file at the workspace "
            f"root). {metric_line}\n\n"
            f"Train a model for this {kind} task and run inference on the test "
            f"files. Find the best possible model — explore multiple approaches "
            f"and pick the winner.\n\n"
            f"OUTPUT CONTRACT — read carefully:\n"
            f"1. Save your predictions as a CSV named EXACTLY `submission.csv` "
            f"with these columns in this exact order: {schema_str}\n"
            f"2. The id column is `{schema.id_col}` — take its values verbatim "
            f"from the sample submission / test items (do NOT reorder or "
            f"regenerate them).\n"
            f"3. The target column(s) `{target_str}` must contain REAL "
            f"predictions from the winning approach — never placeholder values, "
            f"never a single constant value for every row.\n"
            f"4. Persist submission.csv as an artifact. The run is finished "
            f"only after submission.csv is saved.\n\n"
            f"Do not ask for clarification — proceed directly."
        )
        return "\n\n".join(parts)

    # tabular + nlp: text-in-CSV / numeric-in-CSV — same train.csv/test.csv
    # contract (NLP just has a dominant free-text column).
    parts.append(
        f"{metric_line}\n\n"
        f"The training data is in train.csv (target column(s): {target_str}); "
        f"the held-out rows to predict are in test.csv.\n\n"
        f"Find the best possible model for this task — explore multiple "
        f"approaches and pick the winner.\n\n"
        f"OUTPUT CONTRACT — read carefully:\n"
        f"1. After training, run inference with the winning approach on "
        f"test.csv to produce a prediction for every row.\n"
        f"2. Save those predictions as a CSV named EXACTLY `submission.csv` "
        f"with these columns in this exact order: {schema_str}\n"
        f"3. The id column is `{schema.id_col}` — copy its values verbatim "
        f"from test.csv (do NOT reorder or regenerate them).\n"
        f"4. The target column(s) `{target_str}` must contain REAL "
        f"predictions from the winning approach — never placeholder values, "
        f"never a single constant value for every row.\n"
        f"5. Persist submission.csv as an artifact. The run is finished only "
        f"after submission.csv is saved.\n\n"
        f"Do not ask for clarification — proceed directly."
    )
    return "\n\n".join(parts)


# ---------- submission validation --------------------------------------------
# Shared by every Impulse-driven adapter (ImpulseAgent, ImpulseTreeAgent, …):
# the same defensive checks must run on whatever bytes are about to be graded.

def check_not_sample_submission(
    content: bytes,
    sample_bytes: bytes,
    target_cols: list[str],
    filename: str | None,
    log_fn: Callable[[str], None],
) -> None:
    """Defensive byte-identity check: if the file we're about to grade
    is byte-identical to the sample_submission.csv we uploaded at
    the top of this run, it's the schema reference — NOT a real
    agent-produced submission. Fail loud (RuntimeError) rather than grading
    what would be a meaningless ~0.5 random score.

    No-op when `sample_bytes` is empty (sample file missing/unreadable).
    """
    if not (sample_bytes and content == sample_bytes):
        return
    # Surface diagnostic detail so the operator can see exactly
    # what went wrong without having to manually fetch the file.
    log_fn("BYTE-IDENTITY MATCH — submission == sample_submission.csv")
    head = content[:600].decode("utf-8", errors="replace")
    log_fn(f"  first ~600 bytes of would-be submission:\n{head}")
    # Diagnose target-column shape: if the prediction column has
    # exactly one unique value, the agent's model collapsed to a
    # constant predictor (which would explain why it coincides
    # with the all-False sample). If it has multiple values, the
    # match is more likely a literal file copy rather than a
    # degenerate model.
    try:
        import csv
        import io
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace"))))
        n_rows = len(rows)
        target_col = target_cols[0] if target_cols else None
        if target_col and rows:
            uniq = {(r.get(target_col) or "").strip() for r in rows}
            log_fn(
                f"  diagnostic: {n_rows} rows, "
                f"target column '{target_col}' has {len(uniq)} unique "
                f"value(s): {sorted(uniq)[:5]}"
            )
            if len(uniq) == 1:
                log_fn(
                    "  ⇒ likely cause: model collapsed to a constant "
                    "predictor, OR the agent wrote sample_submission.csv "
                    "verbatim without populating predictions."
                )
            else:
                log_fn(
                    "  ⇒ likely cause: the agent literally copied "
                    "sample_submission.csv as its submission and did "
                    "not overwrite with model predictions."
                )
    except Exception as diag_err:
        log_fn(f"  (diagnostic readout failed: {diag_err})")
    raise RuntimeError(
        f"would-be submission ({filename}, {len(content):,} bytes) "
        f"is byte-identical to the sample_submission.csv we uploaded "
        f"as a schema reference. See log above for the file's contents "
        f"and likely root cause. Refusing to grade a meaningless score."
    )


def check_target_columns_populated(
    content: bytes,
    target_cols: list[str],
    log_fn: Callable[[str], None],
) -> None:
    """Two-level submission sanity check before grading:

      level 1 — HARD FAIL: the target column is entirely empty / NaN
        / missing-like. The grader rejects these with score=None,
        which (a) burns a $ slot on a meaningless grade and (b) used
        to crash the bench-service task wrapper. Refuse to grade
        (RuntimeError).

      level 2 — WARN: the target column has exactly one non-empty
        unique value. Possibly a constant predictor (degenerate), but
        possibly a legitimate majority-class prediction on a heavily
        imbalanced task. Warn (log_fn) and let the grader produce the score.

    Diagnostic-side failures (parse errors etc.) never block grading.
    """
    try:
        import csv as _csv
        import io as _io
        rdr = _csv.DictReader(
            _io.StringIO(content.decode("utf-8", errors="replace"))
        )
        rows = list(rdr)
        if rows and target_cols:
            # Strings that mean "no prediction" — treat any column
            # consisting entirely of these as a hard failure.
            _MISSING_LIKE = {"", "nan", "na", "none", "null", "<na>"}
            for tc in target_cols:
                raw_vals = [(r.get(tc) or "").strip() for r in rows]
                unique = set(raw_vals)
                unique_lower = {v.lower() for v in unique}
                if unique_lower <= _MISSING_LIKE:
                    head = content[:600].decode(
                        "utf-8", errors="replace"
                    )
                    log_fn(
                        f"  ✗ submission's '{tc}' column is entirely "
                        f"empty/missing-like across {len(rows)} rows "
                        f"(unique={sorted(unique)[:5]}). Refusing to "
                        f"grade — the agent wrote a CSV with the right "
                        f"header but never populated predictions.\n"
                        f"  first ~600 bytes:\n{head}"
                    )
                    raise RuntimeError(
                        f"submission's '{tc}' column is entirely "
                        f"empty/missing-like ({len(rows)} rows). "
                        f"Refusing to grade a meaningless score."
                    )
                if len(unique) == 1:
                    log_fn(
                        f"  ⚠ submission's '{tc}' column has only one "
                        f"unique value ({sorted(unique)[0]!r}) across "
                        f"{len(rows)} rows — model may have collapsed "
                        f"to a constant predictor. Grading anyway."
                    )
    except RuntimeError:
        # Re-raise our hard-fail; the caller's outer except surfaces it
        # as a clean adapter failure.
        raise
    except Exception:
        pass  # diagnostic-side failures never block grading


# ---------- staging ----------------------------------------------------------

def _choose_staging(modality: str) -> str:
    """Return the staging strategy for a given task modality.

    "archive"  — vision/audio tasks: the whole task tree (image/audio files +
                 label CSVs) is packed as a tar.gz and delivered to the sandbox
                 via GCS pull; the agent must extract task.tar.gz before training.
    "csv"      — tabular/nlp tasks: train.csv and test.csv are uploaded directly
                 to the sandbox workspace via PUT. NLP tasks are text-in-CSV, so
                 they use the same direct-upload path as tabular.
    """
    return "archive" if modality in ("vision", "audio") else "csv"


# ---------- helpers ----------------------------------------------------------

def _upload_to_sandbox(
    sess, sandbox_url: str, session_id: str, file_path: Path,
) -> None:
    """PUT raw file bytes to sandbox at /sessions/{session_id}/files/{filename}.

    SDK-mode agent-api does NOT have a dataset upload endpoint (that route only
    exists in routes/chat.py, not routes/chat_sdk.py). Files must be staged in
    the sandbox workspace directly. The contract per the sandbox's OpenAPI:
      PUT /sessions/{session_id}/files/{filename}
      body: raw bytes (no multipart, no base64)
    """
    with file_path.open("rb") as f:
        data = f.read()
    resp = sess.put(
        f"{sandbox_url}/sessions/{session_id}/files/{file_path.name}",
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=300,
    )
    if not resp.ok:
        raise RuntimeError(
            f"upload {file_path.name} to sandbox → {resp.status_code}: {resp.text[:400]}"
        )


# Event types we surface as log lines (others suppressed to keep the log readable).
_NOISY_TYPES = {"heartbeat"}
_TERMINAL_OK = {"done", "completed"}
_TERMINAL_FAIL = {"error", "failed"}


def _consume_sse(resp, log_fn, remaining_fn) -> tuple[bool, bool, str | None]:
    """Iterate the SSE stream, logging interesting events. Returns
    (done_seen, failed_seen, error_message).

    Termination semantics:
      - Strong success signal: an explicit `{type: "done"}` / `{type: "completed"}`
        event, OR a literal `[DONE]` line.
      - Weak success signal: the stream closes cleanly after we received at least
        one substantive event AND never saw an error/failed event. Agent-api's
        `chat_sdk.py` exits its async-for loop without always emitting a DONE
        event when the orchestrator runs to completion — we trust the clean
        TCP close as a success signal in that case (verified against
        `agent-api/routes/chat_sdk.py`: the `finally` block logs `session_end`
        with `error=None` for those completions).
      - Failure: an explicit `{type: "error"}` / `{type: "failed"}` event.
      - Timeout: caller's `remaining_fn()` hits ≤ 1 second.
      - Empty stream (zero substantive events received then close): NOT success.
    """
    done = False
    failed = False
    err_msg: str | None = None
    substantive_events = 0  # everything except heartbeats
    timed_out = False
    for raw in resp.iter_lines():
        if remaining_fn() <= 1:
            log_fn("  timeout reached while streaming — closing connection")
            timed_out = True
            break
        if not raw:
            continue
        if not raw.startswith(b"data: "):
            continue
        payload = raw[6:].decode("utf-8", errors="replace").strip()
        if payload == "[DONE]":
            done = True
            log_fn("  [DONE]")
            break
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        et = (ev.get("type") or "").lower()
        if et in _NOISY_TYPES:
            continue
        substantive_events += 1
        content = (ev.get("content") or "").strip()
        if et in _TERMINAL_OK:
            done = True
            log_fn(f"  [DONE] {content[:140]}")
            break
        if et in _TERMINAL_FAIL:
            failed = True
            err_msg = content or json.dumps(ev.get("data") or {})[:300]
            log_fn(f"  [ERR] {err_msg}")
            break
        # Everything else: short event-type prefix + truncated content
        short = (content[:140] + "…") if len(content) > 140 else content
        log_fn(f"  [{et}] {short}" if short else f"  [{et}]")
    # If we exited the loop without an explicit terminal event AND saw real
    # work happen AND didn't time out, treat clean stream close as success.
    if (not done and not failed and not timed_out
            and substantive_events > 0):
        done = True
        log_fn(
            f"  [done-inferred] stream closed cleanly after "
            f"{substantive_events} substantive event(s), no error reported"
        )
    return done, failed, err_msg


def _build_messages(schema, refinement_turns: int) -> list[str]:
    """Build the sequence of chat messages.

    Turn 1: include the full task description so the agent has the same
    context a production user would give it when uploading their data.

    Refinement turns: a power-user-style "now improve" follow-up. We bound
    these because each costs a full agent loop (LLM + sandbox cycles).
    """
    target_str = ", ".join(schema.target_cols)
    schema_str = ", ".join(schema.sample_submission_cols)

    # Task brief — like a real user describing their dataset + goal.
    brief_parts: list[str] = []
    if schema.description:
        brief_parts.append(schema.description.strip()[:8000])
    # Metric line — explicit, because Gemini regularly defaults to AUC for
    # binary classification even when the competition grades on accuracy
    # (and vice versa). The CV score then looks great but the leaderboard
    # rank doesn't.
    metric_line = (
        f"The competition is graded on **{schema.metric}**. Optimize, "
        f"cross-validate, and select models against this metric specifically — "
        f"do not default to AUC, accuracy, or RMSE if a different metric is named."
        if schema.metric else
        "Identify the competition's evaluation metric from the task description "
        "above and optimize against it specifically."
    )

    brief_parts.append(
        f"{metric_line}\n\n"
        f"Train the best model you can on train.csv to predict {target_str}. "
        f"Try multiple diverse model families (e.g. LightGBM + XGBoost + "
        f"CatBoost + a neural net) and build a **blended ensemble** — "
        f"averaging diverse models almost always beats picking the single "
        f"best on this kind of competition data. Use cross-validation, "
        f"not a single 80/20 split, when evaluating models.\n\n"
        f"OUTPUT CONTRACT — read this carefully:\n"
        f"1. After training, run inference with your best ensemble on "
        f"test.csv to produce probabilities/labels for every row.\n"
        f"2. Save those predictions as a CSV named EXACTLY `submission.csv` "
        f"(not `final_submission.csv`, not `predictions.csv`, etc.) with "
        f"these columns in this exact order: {schema_str}\n"
        f"3. The id column is `{schema.id_col}` — use the values from "
        f"test.csv verbatim (do NOT reorder).\n"
        f"4. The target column(s) `{target_str}` must contain YOUR MODEL's "
        f"predictions — NOT placeholder values, NOT all-zero, NOT all-False. "
        f"If your submission's target column has only one unique value, "
        f"something is broken: re-check that you ran model.predict on the "
        f"test feature matrix and put those predictions in the file.\n"
        f"5. Do NOT copy any template/sample file as your submission. There "
        f"is no template file in the workspace — construct the submission "
        f"yourself from `test.csv`'s id column and your model's predictions."
    )
    first = "\n\n".join(brief_parts)

    # Refinement turns. Each carries a hard "submission write policy" block at
    # the end — without it the agent has been replacing a passing
    # submission.csv with broken half-finished work, or writing it with
    # fabricated dummy data when training inputs are missing.
    #
    # Turn kinds:
    #   - "improve" turns: may overwrite submission.csv IFF strictly better on
    #     CV with the contract-correct schema
    #   - "diagnostic" turns (stress-test, CV-gap analysis, error analysis):
    #     MUST NOT touch submission.csv at all
    _IMPROVE_RULE = (
        "\n\nHARD SUBMISSION WRITE POLICY:\n"
        "- You may overwrite `submission.csv` ONLY if your new approach\n"
        "  strictly beats the current best CV score AND your new file has\n"
        f"  the contract schema EXACTLY: {schema_str}.\n"
        "- If anything in your script errors out, fails to load `train.csv`\n"
        "  or `test.csv`, or runs on fabricated/dummy/synthetic data: STOP.\n"
        "  Do NOT write `submission.csv`. Leave the prior one intact and\n"
        "  report the error. A working bronze beats a broken gold.\n"
        "- The id column must come VERBATIM from `test.csv` — never\n"
        "  reconstruct ids (`range(len(df))`, `0..N-1`, etc.). If you\n"
        "  cannot read `test.csv`, halt and report — do not improvise.\n"
        "- If you write `submission.csv` this turn, the LAST line of\n"
        "  your final summary must be exactly:\n"
        "      SUBMISSION OVERWRITTEN — cv=<your_cv_score>\n"
        "  Otherwise the LAST line must be exactly:\n"
        "      SUBMISSION UNCHANGED — <one-line reason>"
    )
    _DIAGNOSTIC_RULE = (
        "\n\nHARD SUBMISSION WRITE POLICY (DIAGNOSTIC TURN):\n"
        "- You MUST NOT write, overwrite, or delete `submission.csv` this\n"
        "  turn. This is a diagnostic / analysis turn, not a modeling turn.\n"
        "- If any of your scripts touches `submission.csv` (even by\n"
        "  accident — e.g. a re-saved file from a previous template), that\n"
        "  is a violation of this contract and counts as a failed turn.\n"
        "- Print findings, save diagnostic JSONs (`stress_test_report.json`,\n"
        "  `cv_gap_analysis.json`, etc.) but the existing `submission.csv`\n"
        "  is OFF LIMITS.\n"
        "- The LAST line of your final summary must be exactly:\n"
        "      SUBMISSION UNCHANGED — diagnostic turn"
    )

    refinements = [
        # Turn 2 — improve
        "Now improve on what you've built. Push the score higher. Try more "
        "aggressive feature engineering — interactions between top features, "
        "target encoding with CV protection, groupby aggregations. Try at "
        "least one model family you haven't used yet. Strengthen the "
        "ensemble (rank-averaged or weighted by CV score)."
        + _IMPROVE_RULE,

        # Turn 3 — improve
        "One more iteration. Look at the most important features and engineer "
        "new ones from them. Run a more thorough hyperparameter search on the "
        "top 2-3 models. Diversify the ensemble."
        + _IMPROVE_RULE,

        # Turn 4 — improve (stacking)
        "Next iteration: try a stacking layer. Generate out-of-fold predictions "
        "from your top base models and train a lightweight meta-learner "
        "(logistic regression or a small GBM) on those OOF predictions. "
        "If the task has plentiful unlabeled test data, also consider "
        "pseudo-labeling the highest-confidence test rows and retraining. "
        "Drop features that consistently rank at the bottom of importance — "
        "noise hurts in ensembles."
        + _IMPROVE_RULE,

        # Turn 5 — DIAGNOSTIC (this is where past runs corrupted the submission)
        "Stress-test what you have. Compare your CV score to your training "
        "score: if the gap is wide, you're overfitting — diagnose and "
        "report. If both are low, you're underfitting — diagnose and "
        "report. If the competition metric expects calibrated probabilities "
        "(log-loss, Brier, etc.), evaluate whether isotonic/Platt scaling "
        "would help and save the calibrator alongside the existing models — "
        "do NOT push calibrated probabilities into submission.csv during "
        "this turn."
        + _DIAGNOSTIC_RULE,

        # Turn 6 — improve (final push)
        "Final push. What approach haven't you tried that might fit this "
        "task? Implement it, validate it on CV, and consider including it. "
        "If you do not strictly beat the current CV score, change nothing."
        + _IMPROVE_RULE,
    ]

    return [first, *refinements[: max(0, refinement_turns)]]


# Listing endpoints are called at the END of an otherwise-successful run.
# A timeout here turns a real medal into a "failed" record. The sandbox can
# legitimately take 30-90s to enumerate a workspace after a heavy training
# session (catboost dumps, joblib snapshots, OOF .npy files all on disk).
# So: generous per-attempt timeout + a small retry loop with backoff.
_LIST_TIMEOUT_S = 180
_LIST_RETRIES = 3
_LIST_BACKOFF_S = 5


def _list_timeout(deadline: float | None, per_call_cap: float | None = None) -> float:
    """Per-call list timeout, clamped to a salvage deadline + per-call cap.

    ``per_call_cap`` (when supplied) bounds any single list call so one hang
    can't consume the whole salvage budget and starve the other fetch paths.
    """
    base = _LIST_TIMEOUT_S
    if per_call_cap is not None:
        base = min(base, per_call_cap)
    if deadline is None:
        return _clamp_call_timeout(base)
    return _clamp_call_timeout(min(base, deadline - time.monotonic()))


def _list_session_artifacts(sess, agent_api_url: str, session_id: str, log_fn,
                            deadline: float | None = None,
                            per_call_cap: float | None = None) -> list[dict]:
    """GET /v2/sessions/{id}/artifacts. Tolerant of both shapes the endpoint
    has returned in practice: bare list, or {"artifacts": [...], "total": N}.
    Returns [] on any non-200 / unexpected shape so the caller can fall back
    to the sandbox listing.

    `deadline` (monotonic) caps per-call timeouts and aborts the retry loop
    once exhausted, so the salvage phase honours its total wall budget.
    """
    last_err: Exception | None = None
    for attempt in range(1, _LIST_RETRIES + 1):
        if deadline is not None and time.monotonic() >= deadline:
            log_fn("  artifacts list: salvage budget exhausted")
            break
        try:
            r = sess.get(
                f"{agent_api_url}/v2/sessions/{session_id}/artifacts",
                timeout=_list_timeout(deadline, per_call_cap),
            )
        except Exception as e:
            last_err = e
            log_fn(f"  artifacts list attempt {attempt}/{_LIST_RETRIES} errored: {e}")
            if attempt < _LIST_RETRIES and (
                deadline is None or time.monotonic() < deadline
            ):
                time.sleep(_LIST_BACKOFF_S * attempt)
            continue
        if not r.ok:
            log_fn(f"  artifacts list returned {r.status_code}")
            return []
        body = r.json()
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            for k in ("artifacts", "items"):
                inner = body.get(k)
                if isinstance(inner, list):
                    return inner
        log_fn(f"  artifacts list unexpected shape: {type(body).__name__}")
        return []
    log_fn(f"  artifacts list gave up after {_LIST_RETRIES} attempts ({last_err})")
    return []


def _list_sandbox_files(sess, sandbox_url: str, session_id: str,
                        deadline: float | None = None,
                        per_call_cap: float | None = None) -> list[dict]:
    """Fall back: list files in the sandbox session workspace. Returns
    artifact-shaped records (only `filename` populated) so downstream code
    can use the same matching path.

    Retries on timeout/connection error — this endpoint is the LAST hop of
    a run and a stale 60s timeout used to flip otherwise-successful runs
    into "no submission produced" failures.

    `deadline` (monotonic) caps per-call timeouts and aborts the retry loop
    once exhausted (salvage budget).
    """
    last_err: Exception | None = None
    for attempt in range(1, _LIST_RETRIES + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            r = sess.get(
                f"{sandbox_url}/sessions/{session_id}/files",
                timeout=_list_timeout(deadline, per_call_cap),
            )
            r.raise_for_status()
            body = r.json()
            raw = body if isinstance(body, list) else (
                body.get("files") if isinstance(body, dict) else []
            )
            out: list[dict] = []
            for f in raw or []:
                if isinstance(f, str):
                    out.append({"filename": f})
                elif isinstance(f, dict):
                    fn = f.get("filename") or f.get("name")
                    if fn:
                        out.append({"filename": fn, "size_bytes": f.get("size_bytes")})
            return out
        except Exception as e:
            last_err = e
            if attempt < _LIST_RETRIES and (
                deadline is None or time.monotonic() < deadline
            ):
                time.sleep(_LIST_BACKOFF_S * attempt)
            continue
    # Caller (ImpulseAgent.run) wraps this in a try/except that records the
    # failure as "no submission produced". Raise so the existing path runs.
    raise RuntimeError(
        f"sandbox file listing failed after {_LIST_RETRIES} attempts "
        f"(timeout={_LIST_TIMEOUT_S}s each): {last_err}"
    )


#: Files WE upload at the start of every run — never to be confused with
#: something the agent produced as a real prediction file.
_BENCH_UPLOADED_FILES = {"train.csv", "test.csv", "sample_submission.csv"}


def _find_submission_artifact(artifacts: list[dict]) -> dict | None:
    """Locate a submission file the AGENT produced (never one we uploaded).

    Priority:
      1. Exact `submission.csv`
      2. Any CSV with "submission" in the name and not in the uploaded
         set (e.g. `final_submission.csv`, `submission_ensemble.csv`).
         Prefer the last in listing order — usually the most recently
         saved variant.

    Critically excludes `sample_submission.csv` — that's the schema
    reference we uploaded at the top of the run, not a prediction file.
    Grading it would produce a meaningless ~0.5 random score.
    """
    # Exact match first
    for a in artifacts:
        if (a.get("filename") or "").lower() == "submission.csv":
            return a
    # Substring fallback, excluding our own uploads + any sample_*.csv
    candidates: list[dict] = []
    for a in artifacts:
        fn = (a.get("filename") or "").lower()
        if fn in _BENCH_UPLOADED_FILES:
            continue
        if fn.startswith("sample_"):
            continue
        if "submission" in fn and fn.endswith(".csv"):
            candidates.append(a)
    return candidates[-1] if candidates else None


def _extract_field(obj: Any, field_name: str) -> Any:
    """Pull a field out of the gateway's chat response, which may nest the
    interesting bits inside a tool-call result of unknown shape.

    Probes (in order): top-level → tool_call_result.* → toolResult.* →
    tool_calls[].result.*  → result.*.
    """
    if not isinstance(obj, dict):
        return None
    if field_name in obj:
        return obj[field_name]
    for key in ("tool_call_result", "toolResult", "result", "data"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            found = _extract_field(nested, field_name)
            if found is not None:
                return found
    for tc in obj.get("tool_calls", []) or []:
        if isinstance(tc, dict):
            found = _extract_field(tc, field_name)
            if found is not None:
                return found
    return None
