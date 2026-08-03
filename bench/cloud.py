"""Vertex AI backed remote runner for EES MLE-bench tasks."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bench.adapters.task_detection import _find_sample_submission as find_sample_submission_in_dir
from bench.ees_core.controller import validate_submission_csv
from bench.grader import GradeResult, grade
from bench.runner import ensure_task_data


PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "engg-ai-experimental")
REGION = os.environ.get("GCP_REGION", "us-central1")
INPUT_BUCKET = os.environ.get("EES_CLOUD_INPUT_BUCKET", "engg-ai-experimental-gpu-inputs")
ARTIFACT_BUCKET = os.environ.get("EES_CLOUD_ARTIFACT_BUCKET", "engg-ai-experimental-gpu-artifacts")
IMAGE_REPO = os.environ.get("GPU_SANDBOX_IMAGE_REPO", f"{REGION}-docker.pkg.dev/{PROJECT_ID}/gpu-sandbox")
DATA_MODES = {"local-upload", "cloud-prepare"}
KAGGLE_CREDENTIAL_MODES = {"auto", "env", "local-json", "none"}

PROFILE_SPECS = {
    "cheap-cpu": {
        "image": f"{IMAGE_REPO}/gpu-sandbox-cheap-cpu:latest",
        "machine_type": "n2-standard-4",
        "gpu_type": None,
        "gpu_count": 0,
    },
    "cheap-gpu": {
        "image": f"{IMAGE_REPO}/gpu-sandbox-cheap-gpu:latest",
        "machine_type": "n1-standard-4",
        "gpu_type": "NVIDIA_TESLA_T4",
        "gpu_count": 1,
    },
    "balanced-gpu": {
        "image": f"{IMAGE_REPO}/gpu-sandbox-balanced-gpu:latest",
        "machine_type": "g2-standard-8",
        "gpu_type": "NVIDIA_L4",
        "gpu_count": 1,
    },
}


@dataclass(frozen=True)
class RemoteRunHandle:
    task_id: str
    job_resource_name: str
    display_name: str
    runtime_profile: str
    input_prefix: str
    artifact_prefix: str
    repo_archive_uri: str
    data_prefix_uri: str
    data_mode: str


@dataclass(frozen=True)
class RemoteCollectedResult:
    task_id: str
    status: str
    artifact_prefix: str
    output_dir: Path
    cloud_result_path: Path | None
    submission_path: Path | None
    ledger_path: Path
    validation: dict[str, Any] | None
    grade: dict[str, Any] | None
    error: str | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _select_runtime_profile(task_id: str, explicit_profile: str | None) -> str:
    if explicit_profile:
        if explicit_profile not in PROFILE_SPECS:
            raise ValueError(f"unknown runtime profile: {explicit_profile}")
        return explicit_profile
    if task_id in {
        "aerial-cactus-identification",
        "aptos2019-blindness-detection",
        "dog-breed-identification",
        "dogs-vs-cats-redux-kernels-edition",
        "histopathologic-cancer-detection",
        "leaf-classification",
        "siim-isic-melanoma-classification",
    }:
        return "balanced-gpu"
    if task_id in {"jigsaw-toxic-comment-classification-challenge"}:
        return "cheap-cpu"
    return "cheap-cpu"


def _task_public_dir(cache_dir: Path, task_id: str) -> Path:
    public = Path(cache_dir) / task_id / "prepared" / "public"
    if not public.exists() or not any(public.iterdir()) or find_sample_submission_in_dir(public) is None:
        public = ensure_task_data(task_id, Path(cache_dir))
    if not public.exists() or not any(public.iterdir()):
        raise FileNotFoundError(f"prepared public data is empty or missing: {public}")
    if find_sample_submission_in_dir(public) is None:
        raise FileNotFoundError(f"sample submission not found under {public}")
    return public


def _make_repo_archive(repo_root: Path, output_path: Path) -> None:
    include_roots: list[tuple[Path, str | None]] = [
        (repo_root / "bench", None),
        (repo_root / "pyproject.toml", None),
    ]
    try:
        import mlebench  # type: ignore

        include_roots.append((Path(mlebench.__file__).resolve().parent, "mlebench"))
    except Exception:
        pass

    with tarfile.open(output_path, "w:gz") as tf:
        for root, arcname_root in include_roots:
            if not root.exists():
                continue
            if root.is_file():
                tf.add(root, arcname=arcname_root or root.name)
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                rel = Path(arcname_root) / path.relative_to(root) if arcname_root else path.relative_to(repo_root)
                if "__pycache__" in rel.parts or rel.suffix == ".pyc":
                    continue
                tf.add(path, arcname=rel.as_posix())


def _upload_file(bucket_name: str, source: Path, blob_name: str) -> str:
    uri = f"gs://{bucket_name}/{blob_name}"
    subprocess.run(["gcloud", "storage", "cp", str(source), uri], check=True)
    return uri


def _upload_directory(bucket_name: str, source_dir: Path, prefix: str) -> str:
    destination = f"gs://{bucket_name}/{prefix.rstrip('/')}"
    subprocess.run(["gcloud", "storage", "cp", "--recursive", str(source_dir), destination], check=True)
    return destination


def _submit_vertex_custom_job(display_name: str, worker_pool: dict[str, Any]) -> str:
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    endpoint = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/customJobs"
    body = json.dumps(
        {
            "displayName": display_name,
            "jobSpec": {"workerPoolSpecs": [worker_pool]},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vertex CustomJob create failed: HTTP {exc.code}: {detail}") from exc
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Vertex CustomJob create response did not include name: {payload}")
    return name


def _read_local_kaggle_json() -> tuple[str, str] | None:
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    path = config_dir / "kaggle.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    username = payload.get("username")
    key = payload.get("key")
    if isinstance(username, str) and username and isinstance(key, str) and key:
        return username, key
    return None


def _kaggle_env_for_remote(credentials_mode: str) -> list[dict[str, str]]:
    if credentials_mode not in KAGGLE_CREDENTIAL_MODES:
        raise ValueError(f"unknown Kaggle credentials mode: {credentials_mode}")
    if credentials_mode == "none":
        return []

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return [
            {"name": "KAGGLE_USERNAME", "value": username},
            {"name": "KAGGLE_KEY", "value": key},
        ]
    if credentials_mode == "env":
        raise RuntimeError("KAGGLE_USERNAME and KAGGLE_KEY must be set for --remote-kaggle-credentials=env")

    local_creds = _read_local_kaggle_json()
    if local_creds is not None:
        username, key = local_creds
        return [
            {"name": "KAGGLE_USERNAME", "value": username},
            {"name": "KAGGLE_KEY", "value": key},
        ]
    if credentials_mode == "local-json":
        raise RuntimeError("~/.kaggle/kaggle.json was not found or is missing username/key")
    raise RuntimeError(
        "Kaggle credentials were not found in KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json. "
        "Use --remote-kaggle-credentials=none only if the runtime image already provides credentials."
    )


def _build_remote_job_code(
    *,
    task_id: str,
    agent_arg: str,
    timeout_s: int,
    repo_archive_uri: str,
    data_prefix_uri: str | None,
    data_mode: str = "local-upload",
) -> str:
    if data_mode not in DATA_MODES:
        raise ValueError(f"unknown remote data mode: {data_mode}")
    # Keep this script self-contained: the runtime image supplies google-cloud-storage.
    return f"""
import json
import os
import shutil
import site
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from google.cloud import storage

TASK_ID = {task_id!r}
AGENT_ARG = {agent_arg!r}
TIMEOUT_S = {int(timeout_s)!r}
REPO_ARCHIVE_URI = {repo_archive_uri!r}
DATA_PREFIX_URI = {data_prefix_uri!r}
DATA_MODE = {data_mode!r}

WORKSPACE = Path('/workspace')
REPO_DIR = WORKSPACE / 'repo'
CACHE_DIR = WORKSPACE / 'cache'
PUBLIC_DIR = CACHE_DIR / TASK_ID / 'prepared' / 'public'
RUNS_DIR = WORKSPACE / 'outputs' / 'runs'
RESULT_PATH = WORKSPACE / 'outputs' / 'cloud_result.json'


def parse_gs(uri):
    assert uri.startswith('gs://'), uri
    bucket, _, blob = uri[5:].partition('/')
    return bucket, blob


def download_blob(uri, destination):
    bucket_name, blob_name = parse_gs(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(blob_name).download_to_filename(destination)


def download_prefix(uri, destination):
    bucket_name, prefix = parse_gs(uri.rstrip('/'))
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination.mkdir(parents=True, exist_ok=True)
    for blob in client.list_blobs(bucket, prefix=prefix + '/'):
        if blob.name.endswith('/'):
            continue
        rel = blob.name[len(prefix.rstrip('/') + '/'):]
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)


def configure_kaggle_credentials():
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    if not username or not key:
        return
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / 'kaggle.json'
    kaggle_json.write_text(json.dumps({{'username': username, 'key': key}}))
    kaggle_json.chmod(0o600)


def ensure_mlebench_available():
    ensure_user_bin_on_path()
    try:
        import mlebench.data  # noqa: F401
        import mlebench.grade  # noqa: F401
        import mlebench.registry  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    deps = [
        'appdirs',
        'diskcache',
        'kaggle==1.6.17',
        'py7zr',
        'python-dotenv',
        'pyyaml',
        'tenacity',
        'tqdm',
        'fastparquet',
    ]
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--default-timeout', '120', '--retries', '5', *deps],
        check=True,
    )
    ensure_user_bin_on_path()
    try:
        import mlebench.data  # noqa: F401
        import mlebench.grade  # noqa: F401
        import mlebench.registry  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            '--default-timeout',
            '120',
            '--retries',
            '5',
            'https://github.com/openai/mle-bench/archive/507f92e1138bb6e40dac5c6ee7a6758e6424bf97.zip',
        ],
        check=True,
    )
    ensure_user_bin_on_path()


def ensure_user_bin_on_path():
    candidate_bins = [
        Path.home() / '.local' / 'bin',
        Path(site.USER_BASE) / 'bin',
    ]
    path = os.environ.get('PATH', '')
    parts = path.split(os.pathsep) if path else []
    prepend = [str(path) for path in candidate_bins if str(path) not in parts]
    if prepend:
        os.environ['PATH'] = os.pathsep.join(prepend + parts)


started = time.time()
try:
    archive = WORKSPACE / 'repo.tar.gz'
    download_blob(REPO_ARCHIVE_URI, archive)
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tf:
        tf.extractall(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    if DATA_MODE == 'local-upload':
        if not DATA_PREFIX_URI:
            raise ValueError('DATA_PREFIX_URI is required for local-upload mode')
        download_prefix(DATA_PREFIX_URI, PUBLIC_DIR)
        data_dir = PUBLIC_DIR
    elif DATA_MODE == 'cloud-prepare':
        configure_kaggle_credentials()
        ensure_mlebench_available()
        os.environ['BENCH_GRADER_DATA_DIR'] = str(CACHE_DIR)
        from bench.runner import ensure_task_data
        data_dir = ensure_task_data(TASK_ID, CACHE_DIR)
    else:
        raise ValueError(f'unknown DATA_MODE: {{DATA_MODE}}')

    from bench.adapters.task_detection import _find_sample_submission
    from bench.ees_core.controller import run_ees_task

    sample_submission = _find_sample_submission(data_dir)
    if sample_submission is None:
        raise FileNotFoundError(f'sample submission not found under {{data_dir}}')
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_submission, RESULT_PATH.parent / 'sample_submission.csv')

    result = run_ees_task(
        task_id=TASK_ID,
        data_dir=data_dir,
        work_dir=RUNS_DIR / TASK_ID / 'ees_core',
        grade_submission=(DATA_MODE == 'cloud-prepare'),
    )
    payload = {{
        'success': bool(result.success),
        'task_id': TASK_ID,
        'agent': AGENT_ARG,
        'data_mode': DATA_MODE,
        'public_dir': str(data_dir),
        'candidate_id': result.candidate_id,
        'recipe_id': result.recipe_id,
        'score': result.score,
        'medal': result.medal,
        'no_score_reason': result.no_score_reason,
        'submission_path': str(result.submission_path) if result.submission_path else None,
        'artifact_paths': result.artifact_paths,
        'official_grade': result.metrics.get('official_grade') if isinstance(result.metrics, dict) else None,
        'wall_time_s': time.time() - started,
    }}
except Exception as exc:
    payload = {{
        'success': False,
        'task_id': TASK_ID,
        'agent': AGENT_ARG,
        'error': f'{{type(exc).__name__}}: {{exc}}',
        'wall_time_s': time.time() - started,
    }}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')
print(json.dumps(payload, sort_keys=True))
if not payload.get('success'):
    raise SystemExit(1)
"""


def submit_remote_run(
    *,
    task_id: str,
    agent_arg: str = "ees:standalone",
    cache_dir: Path,
    timeout_s: int,
    runtime_profile: str | None = None,
    wait: bool = False,
    data_mode: str = "local-upload",
    kaggle_credentials_mode: str = "auto",
) -> RemoteRunHandle:
    """Package and submit one EES benchmark task to Vertex AI."""
    if agent_arg != "ees:standalone":
        raise ValueError("remote runner currently supports only agent_arg='ees:standalone'")
    if data_mode not in DATA_MODES:
        raise ValueError(f"unknown remote data mode: {data_mode}")

    repo_root = Path(__file__).resolve().parents[1]
    profile = _select_runtime_profile(task_id, runtime_profile)
    spec = PROFILE_SPECS[profile]

    run_id = f"{task_id}-{_utc_stamp()}-{os.getpid()}"
    input_prefix = f"mlebench/eES/{run_id}"
    artifact_prefix = f"mlebench/eES/{run_id}"

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "repo.tar.gz"
        _make_repo_archive(repo_root, archive)
        repo_uri = _upload_file(INPUT_BUCKET, archive, f"{input_prefix}/repo.tar.gz")

    kaggle_env: list[dict[str, str]] = []
    if data_mode == "local-upload":
        public_dir = _task_public_dir(Path(cache_dir), task_id)
        data_uri: str | None = _upload_directory(INPUT_BUCKET, public_dir, f"{input_prefix}/data")
    else:
        kaggle_env = _kaggle_env_for_remote(kaggle_credentials_mode)
        data_uri = None

    code = _build_remote_job_code(
        task_id=task_id,
        agent_arg=agent_arg,
        timeout_s=timeout_s,
        repo_archive_uri=repo_uri,
        data_prefix_uri=data_uri,
        data_mode=data_mode,
    )

    env = [
        {"name": "ARTIFACT_BUCKET", "value": ARTIFACT_BUCKET},
        {"name": "ARTIFACT_PREFIX", "value": artifact_prefix},
        *kaggle_env,
    ]
    worker_pool: dict[str, Any] = {
        "machineSpec": {"machineType": spec["machine_type"]},
        "replicaCount": 1,
        "containerSpec": {
            "imageUri": spec["image"],
            "args": ["python", "-c", code],
            "env": env,
        },
    }
    if spec["gpu_type"]:
        worker_pool["machineSpec"]["acceleratorType"] = spec["gpu_type"]
        worker_pool["machineSpec"]["acceleratorCount"] = spec["gpu_count"]

    display_name = f"ees-{task_id}-{_utc_stamp()}"
    job_resource_name = _submit_vertex_custom_job(display_name, worker_pool)
    if wait and job_resource_name:
        _wait_for_vertex_job(job_resource_name)

    return RemoteRunHandle(
        task_id=task_id,
        job_resource_name=job_resource_name,
        display_name=display_name,
        runtime_profile=profile,
        input_prefix=f"gs://{INPUT_BUCKET}/{input_prefix}",
        artifact_prefix=f"gs://{ARTIFACT_BUCKET}/{artifact_prefix}",
        repo_archive_uri=repo_uri,
        data_prefix_uri=data_uri or data_mode,
        data_mode=data_mode,
    )


def _wait_for_vertex_job(job_resource_name: str, poll_s: int = 60) -> str:
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    while True:
        state = _describe_vertex_job_state(job_resource_name)
        print(f"{job_resource_name} {state}", flush=True)
        if state in terminal_states:
            return state
        time.sleep(poll_s)


def _describe_vertex_job_state(job_resource_name: str) -> str:
    proc = subprocess.run(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "describe",
            job_resource_name,
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            "--format=value(state)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gcloud ai custom-jobs describe failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith("Using endpoint")]
    return lines[-1] if lines else "JOB_STATE_UNSPECIFIED"


def poll_remote_run(*args, **kwargs):
    raise NotImplementedError("poll by job_resource_name via Vertex AI console or google.cloud.aiplatform.CustomJob")


def pull_remote_result(*, artifact_prefix: str, output_dir: Path) -> Path:
    """Download a remote run artifact prefix into ``output_dir`` using gcloud auth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = artifact_prefix.rstrip("/") + "/outputs"
    subprocess.run(["gcloud", "storage", "cp", "--recursive", source, str(output_dir)], check=True)
    return output_dir / "outputs"


def collect_remote_result(
    *,
    task_id: str,
    artifact_prefix: str,
    output_dir: Path,
    cache_dir: Path,
    grade_fn: Callable[[str, Path], GradeResult] = grade,
) -> RemoteCollectedResult:
    """Pull remote artifacts, validate the submission, grade it, and write a terminal ledger."""
    output_dir = Path(output_dir)
    ledger_path = output_dir / "remote_result.json"
    try:
        outputs_dir = pull_remote_result(artifact_prefix=artifact_prefix, output_dir=output_dir)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "terminal": True,
            "task_id": task_id,
            "status": "collection_failed",
            "artifact_prefix": artifact_prefix,
            "output_dir": None,
            "cloud_result_path": None,
            "cloud_result": None,
            "submission": {"path": None, "validation": None},
            "grade": None,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return RemoteCollectedResult(
            task_id=task_id,
            status="collection_failed",
            artifact_prefix=artifact_prefix,
            output_dir=output_dir,
            cloud_result_path=None,
            submission_path=None,
            ledger_path=ledger_path,
            validation=None,
            grade=None,
            error=error,
        )
    cloud_result_path = outputs_dir / "cloud_result.json"
    cloud_result = _read_json_file(cloud_result_path)
    submission_path = _find_remote_submission(outputs_dir, task_id, cloud_result)
    validation: dict[str, Any] | None = None
    grade_payload: dict[str, Any] | None = None
    status = "graded"
    error = None

    try:
        if submission_path is None:
            raise FileNotFoundError(f"submission.csv not found under {outputs_dir}")
        sample_submission = _find_collected_sample_submission(outputs_dir)
        if sample_submission is None:
            sample_submission = _find_sample_submission(Path(cache_dir), task_id)
        validation = validate_submission_csv(submission_path, sample_submission)
        grade_payload = _cloud_grade_to_dict(cloud_result) if cloud_result else None
        if grade_payload is None:
            grade_payload = _grade_to_dict(grade_fn(task_id, submission_path))
        if not grade_payload["success"]:
            status = "grade_failed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    payload = {
        "terminal": True,
        "task_id": task_id,
        "status": status,
        "artifact_prefix": artifact_prefix,
        "output_dir": str(outputs_dir),
        "cloud_result_path": str(cloud_result_path) if cloud_result_path.exists() else None,
        "cloud_result": cloud_result,
        "submission": {
            "path": str(submission_path) if submission_path else None,
            "validation": validation,
        },
        "grade": grade_payload,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return RemoteCollectedResult(
        task_id=task_id,
        status=status,
        artifact_prefix=artifact_prefix,
        output_dir=outputs_dir,
        cloud_result_path=cloud_result_path if cloud_result_path.exists() else None,
        submission_path=submission_path,
        ledger_path=ledger_path,
        validation=validation,
        grade=grade_payload,
        error=error,
    )


def handle_to_json(handle: RemoteRunHandle) -> str:
    return json.dumps(asdict(handle), indent=2, sort_keys=True) + "\n"


def _parse_job_resource_name(stdout: str, stderr: str) -> str:
    combined = "\n".join([stdout or "", stderr or ""])
    try:
        payload = json.loads(stdout)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("name", "resourceName"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    match = re.search(r"projects/[^\\s'\\\"]+/locations/[^\\s'\\\"]+/customJobs/\\d+", combined)
    return match.group(0) if match else ""


def _find_remote_submission(
    outputs_dir: Path,
    task_id: str,
    cloud_result: dict[str, Any] | None = None,
) -> Path | None:
    if cloud_result:
        submission_path = cloud_result.get("submission_path")
        if isinstance(submission_path, str) and submission_path:
            marker = f"/outputs/"
            rel = submission_path.split(marker, 1)[1] if marker in submission_path else ""
            if rel:
                candidate = outputs_dir / rel
                if candidate.exists():
                    return candidate
    preferred_root = outputs_dir / "runs" / task_id / "ees_core"
    candidates: list[Path] = []
    if preferred_root.exists():
        candidates = sorted(path for path in preferred_root.glob("*") if path.name == "submission.csv")
        if not candidates:
            candidates = sorted(preferred_root.glob("*/submission.csv"))
        if not candidates:
            candidates = sorted(preferred_root.rglob("submission.csv"))
    if not candidates:
        candidates = sorted(outputs_dir.rglob("submission.csv"))
    return candidates[0] if candidates else None


def _find_sample_submission(cache_dir: Path, task_id: str) -> Path:
    public_dir = _task_public_dir(cache_dir, task_id)
    sample_submission = find_sample_submission_in_dir(public_dir)
    if sample_submission is not None:
        return sample_submission
    raise FileNotFoundError(f"sample submission not found under {public_dir}")


def _find_collected_sample_submission(outputs_dir: Path) -> Path | None:
    sample_submission = find_sample_submission_in_dir(outputs_dir)
    if sample_submission is not None:
        return sample_submission
    for path in sorted(outputs_dir.rglob("*.csv")):
        low = path.name.lower().replace("_", "").replace("-", "")
        if "samplesubmission" in low:
            return path
    return None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _grade_to_dict(result: GradeResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "score": result.score,
        "metric": result.metric,
        "medal": result.medal(),
        "gold_threshold": result.gold_threshold,
        "silver_threshold": result.silver_threshold,
        "bronze_threshold": result.bronze_threshold,
        "raw": result.raw,
        "error": result.error,
    }


def _cloud_grade_to_dict(cloud_result: dict[str, Any]) -> dict[str, Any] | None:
    raw = cloud_result.get("official_grade")
    if not isinstance(raw, dict):
        return None
    medal = "none"
    if raw.get("gold_medal"):
        medal = "gold"
    elif raw.get("silver_medal"):
        medal = "silver"
    elif raw.get("bronze_medal"):
        medal = "bronze"
    success = bool(raw.get("valid_submission", True)) and raw.get("score") is not None
    return {
        "success": success,
        "score": raw.get("score"),
        "metric": raw.get("metric") or raw.get("score_metric"),
        "medal": medal,
        "gold_threshold": raw.get("gold_threshold"),
        "silver_threshold": raw.get("silver_threshold"),
        "bronze_threshold": raw.get("bronze_threshold"),
        "raw": raw,
        "error": None if success else f"cloud grader did not produce a valid score: {raw}",
    }
