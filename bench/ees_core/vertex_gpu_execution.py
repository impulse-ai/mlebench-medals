"""GPU fine-tune execution seam: offload operator training to a Vertex CustomJob.

This is the single place the EES image operators call to run training on a real
GPU instead of the in-process CPU host. It:

- stages the task data to a **unique** ``run_<ts>/`` GCS path (defect 3
  workaround — the write SA lacks ``storage.objects.delete``, so no run ever
  overwrites another),
- wraps the caller's training code in the **launcher pattern** proven in the
  2026-07-09 campaign (set ``LD_LIBRARY_PATH=/usr/local/nvidia/lib64`` then
  ``os.execv`` a fresh interpreter, so the dynamic linker finds the CUDA driver
  libs before torch dlopens ``libcuda`` — guarantees CUDA even if image
  resolution regresses),
- submits a CUDA + GPU (T4, spot) CustomJob through the *fixed*
  ``agent-api/tools/vertex_compute.submit_vertex_job``,
- governs runtime with an in-job wall budget **and** an external watchdog cancel
  (Vertex does not enforce ``timeout_hours`` on the job itself — documented),
- polls job state via the SDK (never ``check_vertex_job``'s turn-budget guard),
- pulls artifacts from the unique GCS path and returns them.

Boundary note: ``bench`` and ``agent-api`` are separate top-level trees.
``submit_vertex_job`` lives under ``agent-api`` and pulls in ``runtime_profiles``
/ ``tools`` / ``services`` at import. We therefore import it **lazily**, inside a
helper that first puts ``agent-api`` on ``sys.path`` and raises a clear
``VertexUnavailableError`` if the boundary can't be crossed — the import never
happens at ``bench`` import time, and unit tests inject a mock ``submit_fn`` so
no ``agent-api`` import (or network) occurs.

All external calls (GCS, Vertex SDK, submit) are injectable so tests run fully
mocked with no real GPU.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Driver library path baked into the GPU sandbox image; the launcher prepends it
# and re-execs so torch's CUDA .so resolves it (glibc captures LD_LIBRARY_PATH at
# process start, so a plain os.environ mutation after launch is too late).
NVIDIA_LIB_DIR = "/usr/local/nvidia/lib64"

# A high per-job ceiling: the GPU lane is governed by the wall budget + watchdog,
# not by submit_vertex_job's CPU-scale cost guard (a T4-hour is ~$0.35 on spot).
_GPU_LANE_PER_JOB_MAX_USD = float(os.environ.get("EES_GPU_PER_JOB_MAX_USD", "50"))


class VertexUnavailableError(RuntimeError):
    """Raised when the agent-api Vertex tooling cannot be imported from bench."""


@dataclass(frozen=True)
class GpuJobResult:
    """What the seam returns to the calling operator."""

    success: bool
    job_id: Optional[str]
    state: str
    run_ts: str
    data_uri: str
    output_uri: str
    output_dir: Path
    artifact_paths: list[str] = field(default_factory=list)
    submission_path: Optional[Path] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# bench ↔ agent-api import boundary
# ---------------------------------------------------------------------------

def _agent_api_dir() -> Path:
    # Honor an explicit override first (deployments where bench and agent-api live
    # in separate trees, e.g. bench under impulse-capA/ and agent-api under
    # impulse/). Otherwise assume the in-repo layout: repo root is parents[2].
    override = os.environ.get("EES_AGENT_API_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "agent-api"


def load_submit_vertex_job() -> Callable[..., Any]:
    """Lazily import ``submit_vertex_job`` from agent-api.

    Kept out of module import so ``bench`` never hard-depends on ``agent-api``.
    Raises ``VertexUnavailableError`` with an actionable message on failure.
    """
    agent_api = _agent_api_dir()
    if not agent_api.is_dir():
        raise VertexUnavailableError(
            f"agent-api not found at {agent_api}; cannot submit Vertex GPU jobs."
        )
    if str(agent_api) not in sys.path:
        sys.path.insert(0, str(agent_api))
    try:
        from tools.vertex_compute import submit_vertex_job  # type: ignore
    except Exception as exc:  # noqa: BLE001 — surface a clear boundary error
        raise VertexUnavailableError(
            f"could not import submit_vertex_job from {agent_api}: {exc}"
        ) from exc
    return submit_vertex_job


def _load_aiplatform() -> Any:
    try:
        from google.cloud import aiplatform  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise VertexUnavailableError(
            f"google-cloud-aiplatform not importable for SDK polling: {exc}"
        ) from exc
    return aiplatform


def _load_storage_client() -> Any:
    try:
        from google.cloud import storage  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise VertexUnavailableError(
            f"google-cloud-storage not importable for data staging: {exc}"
        ) from exc
    return storage.Client()


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------

def default_staging_bucket() -> str:
    project = os.environ.get("GCP_PROJECT_ID", "engg-ai-experimental")
    return os.environ.get("VERTEX_STAGING_BUCKET", f"gs://{project}-gpu-artifacts")


def make_run_ts() -> str:
    """A per-run token unique even within the same wall-clock second.

    Defect 3: every job must write to a fresh path so a retry never overwrites a
    prior object (the SA lacks ``storage.objects.delete``).
    """
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def build_run_paths(bucket: str, task_id: str, run_ts: str) -> dict[str, str]:
    """Unique GCS paths for one run: input tar, output tar, telemetry prefix."""
    base = bucket.rstrip("/")
    if not base.startswith("gs://"):
        base = f"gs://{base}"
    safe_task = (task_id or "gpu_task").strip("/") or "gpu_task"
    prefix = f"{base}/{safe_task}/{run_ts}"
    return {
        "prefix": prefix,
        "data_uri": f"{prefix}/data.tar.gz",
        "output_uri": f"{prefix}/output/results.tar.gz",
        "telemetry_prefix": prefix,  # progress_ep*.json + error.txt convention
    }


# Tail appended to the caller's training code inside the re-exec'd entry. The
# launcher's os.execv bypasses submit_vertex_job's post-wrapper upload, so the
# training entry uploads its own NEW artifacts (everything not present in the
# pre-exec snapshot, i.e. not the extracted input data).
_UPLOAD_TAIL = '''

# --- gpu-launcher artifact upload (bypasses the submit post-wrapper) ---
import os as _u_os, io as _u_io, json as _u_json, tarfile as _u_tar
_u_out = _u_os.environ.get("OUTPUT_GCS_PATH")
if _u_out:
    try:
        _u_before = set()
        if _u_os.path.exists("_gpu_before.json"):
            with open("_gpu_before.json") as _u_fh:
                _u_before = set(_u_json.load(_u_fh))
        _u_scaffold = {"./_gpu_before.json", "./_gpu_train_entry.py",
                       "_gpu_before.json", "_gpu_train_entry.py"}
        _u_buf = _u_io.BytesIO()
        with _u_tar.open(fileobj=_u_buf, mode="w:gz") as _u_t:
            for _u_r, _u_d, _u_fs in _u_os.walk("."):
                for _u_f in _u_fs:
                    _u_p = _u_os.path.join(_u_r, _u_f)
                    if _u_p in _u_before or _u_p in _u_scaffold:
                        continue
                    _u_t.add(_u_p, arcname=_u_os.path.relpath(_u_p, "."))
        _u_buf.seek(0)
        assert _u_out.startswith("gs://"), _u_out
        _u_b, _, _u_k = _u_out[5:].partition("/")
        from google.cloud import storage as _u_st
        _u_st.Client().bucket(_u_b).blob(_u_k).upload_from_file(_u_buf)
        print("[gpu-launcher] uploaded outputs to", _u_out)
    except Exception as _u_e:  # write GCS telemetry (defect 4 convention)
        print("[gpu-launcher] upload failed:", _u_e)
        raise
'''


def build_launcher(training_code: str) -> str:
    """Wrap ``training_code`` in the CUDA-guaranteeing launcher.

    The outer code (run under submit_vertex_job's pre-wrapper, which has already
    downloaded + extracted the input data) snapshots the current files, writes
    the training entry, prepends the driver libs to ``LD_LIBRARY_PATH``, and
    ``os.execv``s a fresh interpreter into the entry — so torch loads CUDA
    against the driver libs. The entry runs the caller's training code then
    uploads its NEW artifacts.
    """
    entry = training_code + _UPLOAD_TAIL
    launcher = (
        "import os, sys, json\n"
        f"_NVIDIA_LIB = {NVIDIA_LIB_DIR!r}\n"
        "if os.environ.get('_GPU_LAUNCHER_RELAUNCHED') != '1':\n"
        "    _before = []\n"
        "    for _r, _d, _fs in os.walk('.'):\n"
        "        for _f in _fs:\n"
        "            _before.append(os.path.join(_r, _f))\n"
        "    with open('_gpu_before.json', 'w') as _fh:\n"
        "        json.dump(_before, _fh)\n"
        "    with open('_gpu_train_entry.py', 'w') as _fh:\n"
        "        _fh.write(_TRAIN_ENTRY)\n"
        "    _ld = os.environ.get('LD_LIBRARY_PATH', '')\n"
        "    os.environ['LD_LIBRARY_PATH'] = _NVIDIA_LIB + ((':' + _ld) if _ld else '')\n"
        "    os.environ['_GPU_LAUNCHER_RELAUNCHED'] = '1'\n"
        "    os.execv(sys.executable, [sys.executable, '_gpu_train_entry.py'])\n"
    )
    # Embed the entry as a literal so the outer process writes it verbatim.
    return f"_TRAIN_ENTRY = {entry!r}\n" + launcher


def parse_job_id(submit_result: str) -> Optional[str]:
    """Pull the ``job_id`` resource name out of submit_vertex_job's text result."""
    if not submit_result or "[SUCCESS]" not in submit_result:
        return None
    for line in submit_result.splitlines():
        line = line.strip()
        if line.startswith("job_id:"):
            return line.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Staging / polling / pulling (external calls injected for tests)
# ---------------------------------------------------------------------------

def _parse_gs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), uri
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def _code_tree_filter(tarinfo: Any) -> Any:
    """Drop __pycache__, compiled artifacts, tests and VCS from a shipped code tree."""
    parts = tarinfo.name.split("/")
    if any(p in ("__pycache__", "tests", ".git", ".pytest_cache") for p in parts):
        return None
    if tarinfo.name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def stage_data_to_gcs(
    data_dir: Path,
    dest_uri: str,
    *,
    storage_client: Any,
    code_trees: Optional[list] = None,
) -> int:
    """Tar ``data_dir`` and upload to ``dest_uri``; return the uploaded size.

    ``code_trees`` — optional list of ``(src_path, arcname)`` pairs added into the
    same tar so the in-container job can import the operator source (the GPU
    sandbox image carries no ``bench`` package). Compiled/test files are skipped.

    The tar is streamed to a temp file (not an in-memory BytesIO) so a large image
    task -- histo train+test is ~200k .tif files -- does not build a multi-GB tar
    in RAM.
    """
    import tempfile

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    try:
        # compresslevel=1: .tif images are already compressed, so level-9 gzip
        # burns minutes of CPU on a ~6GB image task for ~no size gain. r:gz on the
        # container side auto-detects gzip regardless of level.
        with tarfile.open(fileobj=tmp, mode="w:gz", compresslevel=1) as tar:
            tar.add(data_dir, arcname=".")
            for src, arcname in (code_trees or []):
                src = Path(src)
                if src.exists():
                    tar.add(src, arcname=arcname, filter=_code_tree_filter)
        tmp.flush()
        size = tmp.tell()
        if size <= 0:
            raise ValueError(f"staged data tar is empty for {data_dir}")
        tmp.seek(0)
        bucket, key = _parse_gs(dest_uri)
        storage_client.bucket(bucket).blob(key).upload_from_file(tmp)
        return size
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def pull_artifacts_from_gcs(src_uri: str, output_dir: Path, *, storage_client: Any) -> list[str]:
    """Download the results tar from ``src_uri`` and extract into ``output_dir``.

    Returns the extracted file paths; an empty list if the blob is absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket, key = _parse_gs(src_uri)
    blob = storage_client.bucket(bucket).blob(key)
    if not blob.exists():
        return []
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    extracted: list[str] = []
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            tar.extract(member, output_dir)
            extracted.append(str(output_dir / member.name))
    return extracted


_TERMINAL = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def poll_until_terminal(
    job_id: str,
    *,
    aiplatform: Any,
    timeout_s: float,
    poll_interval_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> str:
    """Poll a CustomJob via the SDK until terminal, or cancel past ``timeout_s``.

    This is the external watchdog: Vertex does not enforce our wall budget on the
    job, so we cancel a job that overruns ``timeout_s`` (in-job wall budget +
    grace). Returns the final observed state.
    """
    started = now()
    job = aiplatform.CustomJob.get(job_id)
    while True:
        state = job.state.name if job.state else "UNKNOWN"
        if state in _TERMINAL:
            return state
        if now() - started >= timeout_s:
            try:
                job.cancel()
            except Exception:  # noqa: BLE001 — best-effort watchdog cancel
                pass
            return "JOB_STATE_CANCELLED"
        sleep(poll_interval_s)
        job = aiplatform.CustomJob.get(job_id)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_async(coro: Any) -> Any:
    """Run an async submit call from a sync operator context.

    Operators run in worker processes with no live event loop; asyncio.run is
    the correct entry. Falls back to a fresh loop if one is unexpectedly set.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def run_gpu_training_job(
    training_code: str,
    data_dir: Path,
    *,
    task_id: str,
    output_dir: Path,
    machine_type: str = "n1-standard-8",
    gpu_type: str = "NVIDIA_TESLA_T4",
    gpu_count: int = 1,
    timeout_min: int = 60,
    watchdog_grace_s: float = float(os.environ.get("EES_GPU_WATCHDOG_GRACE_S", "300")),
    poll_interval_s: float = 30.0,
    staging_bucket: Optional[str] = None,
    run_ts: Optional[str] = None,
    session_id: Optional[str] = None,
    submit_fn: Optional[Callable[..., Any]] = None,
    aiplatform: Any = None,
    storage_client: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> GpuJobResult:
    """Run ``training_code`` on a Vertex GPU CustomJob and return its artifacts.

    All external dependencies (``submit_fn``, ``aiplatform``, ``storage_client``)
    are injectable so this is fully unit-testable with no network / GPU.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket = staging_bucket or default_staging_bucket()
    run_ts = run_ts or make_run_ts()
    paths = build_run_paths(bucket, task_id, run_ts)
    session_id = session_id or f"gpu-{run_ts}"

    storage_client = storage_client or _load_storage_client()
    # Ship the bench package into the container so the in-job code can import the
    # operator (the gpu-sandbox image has no bench). bench/ is parents[1] of this
    # file; it lands at ./bench in the container cwd (on sys.path after execv).
    bench_pkg = Path(__file__).resolve().parents[1]
    try:
        stage_data_to_gcs(
            Path(data_dir), paths["data_uri"], storage_client=storage_client,
            code_trees=[(bench_pkg, "bench")],
        )
    except Exception as exc:  # noqa: BLE001
        return GpuJobResult(
            success=False, job_id=None, state="STAGING_FAILED", run_ts=run_ts,
            data_uri=paths["data_uri"], output_uri=paths["output_uri"],
            output_dir=output_dir, error=f"data staging failed: {exc}",
        )

    submit_fn = submit_fn or load_submit_vertex_job()
    job_code = build_launcher(training_code)
    submit_result = _run_async(
        submit_fn(
            code=job_code,
            auto_profile=False,          # profile-less GPU submit (defect 1 path)
            runtime_profile=None,
            machine_type=machine_type,
            gpu_type=gpu_type,           # T4 default (defect 2)
            gpu_count=gpu_count,
            timeout_hours=timeout_min / 60.0,
            preemptible=True,            # spot (defect 2)
            session_id=session_id,
            data_gcs_path=paths["data_uri"],
            output_gcs_path=paths["output_uri"],  # unique path (defect 3)
            per_job_max_usd=_GPU_LANE_PER_JOB_MAX_USD,
        )
    )
    job_id = parse_job_id(submit_result)
    if job_id is None:
        return GpuJobResult(
            success=False, job_id=None, state="SUBMIT_FAILED", run_ts=run_ts,
            data_uri=paths["data_uri"], output_uri=paths["output_uri"],
            output_dir=output_dir, error=f"submit failed: {submit_result}",
        )

    aiplatform = aiplatform or _load_aiplatform()
    # In-job wall budget is enforced by the training code; the watchdog cancels
    # if the job overruns that budget plus a provisioning/teardown grace.
    timeout_s = timeout_min * 60.0 + watchdog_grace_s
    state = poll_until_terminal(
        job_id, aiplatform=aiplatform, timeout_s=timeout_s,
        poll_interval_s=poll_interval_s, sleep=sleep,
    )

    artifact_paths: list[str] = []
    submission_path: Optional[Path] = None
    if state == "JOB_STATE_SUCCEEDED":
        artifact_paths = pull_artifacts_from_gcs(
            paths["output_uri"], output_dir, storage_client=storage_client
        )
        candidate = output_dir / "submission.csv"
        if candidate.exists():
            submission_path = candidate

    return GpuJobResult(
        success=state == "JOB_STATE_SUCCEEDED",
        job_id=job_id,
        state=state,
        run_ts=run_ts,
        data_uri=paths["data_uri"],
        output_uri=paths["output_uri"],
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        submission_path=submission_path,
        error=None if state == "JOB_STATE_SUCCEEDED" else f"job terminal state: {state}",
    )
