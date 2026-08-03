"""Per-task execution: provision task data, run agent, grade, persist."""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .agents import Agent, AgentResult
from .adapters.task_detection import _find_sample_submission
from .grader import GradeResult, grade


def _run_kaggle_download(task_id: str, competition_dir: Path, timeout_s: int) -> Path:
    competition_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["kaggle", "competitions", "download", "-c", task_id, "-p", str(competition_dir)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`kaggle competitions download -c {task_id}` failed (exit {proc.returncode}). "
            f"If this is a 403/rules error, the KAGGLE_USERNAME account must accept the "
            f"competition rules on kaggle.com first.\n"
            f"stdout:\n{proc.stdout[-800:]}\nstderr:\n{proc.stderr[-800:]}"
        )
    zip_files = sorted(competition_dir.glob("*.zip"))
    if len(zip_files) != 1:
        raise RuntimeError(
            f"Expected exactly one zip for {task_id} in {competition_dir}, found {len(zip_files)}."
        )
    return zip_files[0]


def _prepare_from_zip(task_id: str, cache_dir: Path, timeout_s: int) -> Path:
    from mlebench.data import is_dataset_prepared
    from mlebench.registry import registry
    from mlebench.utils import extract, is_empty

    competition = registry.set_data_dir(cache_dir).get_competition(task_id)
    competition_dir = competition.raw_dir.parent
    competition.raw_dir.mkdir(exist_ok=True, parents=True)
    competition.public_dir.mkdir(exist_ok=True, parents=True)
    competition.private_dir.mkdir(exist_ok=True, parents=True)

    zip_files = sorted(competition_dir.glob("*.zip"))
    if len(zip_files) == 1 and not zipfile.is_zipfile(zip_files[0]):
        print(f"  cached zip for {task_id} is invalid; redownloading…")
        zip_files[0].unlink()
        zip_files = []

    if not zip_files:
        zip_path = _run_kaggle_download(task_id, competition_dir, timeout_s)
    elif len(zip_files) == 1:
        zip_path = zip_files[0]
    else:
        raise RuntimeError(
            f"Expected exactly one zip for {task_id} in {competition_dir}, found {len(zip_files)}."
        )

    if is_empty(competition.raw_dir):
        print(f"  extracting cached/downloaded data for {task_id}…")
        extract(zip_path, competition.raw_dir, recursive=False)

    if not is_dataset_prepared(competition):
        print(f"  preparing public/private split for {task_id}…")
        competition.prepare_fn(
            raw=competition.raw_dir,
            public=competition.public_dir,
            private=competition.private_dir,
        )

    if not is_dataset_prepared(competition):
        raise RuntimeError(f"local prepare completed but {task_id} is still not prepared")

    (competition.public_dir / "description.md").write_text(competition.description)
    return competition.public_dir


def _is_usable_public_data(data_dir: Path) -> bool:
    return data_dir.exists() and any(data_dir.iterdir()) and _find_sample_submission(data_dir) is not None


@dataclass
class TaskRun:
    run_id: str
    task_id: str
    agent_name: str
    agent_result: AgentResult
    grade_result: GradeResult | None
    work_dir: Path
    created_at: float

    def to_dict(self) -> dict:
        ar, gr = self.agent_result, self.grade_result
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "agent_success": ar.success,
            "wall_time_s": ar.wall_time_s,
            "submission_path": str(ar.submission_path) if ar.submission_path else None,
            "agent_error": ar.error,
            "agent_meta": getattr(ar, "meta", None) or {},
            "score": gr.score if gr else None,
            "metric": gr.metric if gr else None,
            "medal": gr.medal() if gr else "none",
            "gold_threshold": gr.gold_threshold if gr else None,
            "silver_threshold": gr.silver_threshold if gr else None,
            "bronze_threshold": gr.bronze_threshold if gr else None,
            "grade_error": gr.error if gr else None,
            "work_dir": str(self.work_dir),
            "created_at": self.created_at,
        }


def ensure_task_data(task_id: str, cache_dir: Path) -> Path:
    """Ensure the task's data is prepared via `mlebench prepare`.

    Returns the path that should be passed to the agent as --data-dir.
    """
    data_dir = cache_dir / task_id / "prepared" / "public"
    if _is_usable_public_data(data_dir):
        return data_dir
    print(f"  preparing task data for {task_id} (one-time)…")
    # stdin=DEVNULL is critical: on a Kaggle 403 (creds invalid, or the account
    # hasn't accepted the competition rules) mlebench calls input("accept
    # rules? (y/n)"). With no TTY that blocks forever — a silent, un-killable
    # hang, since output is captured and there's no timeout. DEVNULL makes the
    # prompt hit EOF and fail fast; the timeout bounds a hung download/auth.
    timeout_s = int(os.getenv("BENCH_PREPARE_TIMEOUT_S", "900"))
    if os.getenv("BENCH_USE_LOCAL_PREPARE", "1") != "0":
        try:
            prepared_dir = _prepare_from_zip(task_id, cache_dir, timeout_s)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"`kaggle competitions download -c {task_id}` timed out after {timeout_s}s."
            ) from e
        if not _is_usable_public_data(prepared_dir):
            raise RuntimeError(
                f"local prepare completed but {task_id} public data is not usable: "
                f"missing sample submission or empty directory at {prepared_dir}"
            )
        return prepared_dir

    try:
        proc = subprocess.run(
            ["mlebench", "prepare", "-c", task_id, "--skip-verification"],
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"`mlebench prepare -c {task_id} --skip-verification` timed out after {timeout_s}s — "
            f"usually a Kaggle auth/rules problem. Confirm the KAGGLE_USERNAME "
            f"account has joined and accepted this competition's rules on "
            f"kaggle.com.\npartial stdout:\n{(e.stdout or '')[:600]}"
        ) from e
    if proc.returncode != 0:
        # mlebench prints the 403 / 'accept rules' message to stdout, so surface
        # both streams — stderr alone is often empty for this failure.
        raise RuntimeError(
            f"`mlebench prepare -c {task_id} --skip-verification` failed (exit {proc.returncode}). "
            f"If this is a 403/rules error, the KAGGLE_USERNAME account must "
            f"accept the competition rules on kaggle.com first.\n"
            f"stdout:\n{proc.stdout[-800:]}\nstderr:\n{proc.stderr[-800:]}"
        )
    if not data_dir.exists():
        raise RuntimeError(
            f"mlebench prepare succeeded but {data_dir} does not exist. "
            f"Check `mlebench` cache layout."
        )
    if not _is_usable_public_data(data_dir):
        raise RuntimeError(
            f"mlebench prepare completed but {task_id} public data is not usable: "
            f"missing sample submission or empty directory at {data_dir}"
        )
    return data_dir


def run_task(
    task_id: str,
    agent: Agent,
    runs_dir: Path,
    cache_dir: Path,
    timeout_s: int,
) -> TaskRun:
    """Run one (task, agent) end-to-end. No exceptions escape — all are
    captured in the returned TaskRun."""
    run_id = f"{task_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    work_dir = runs_dir / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare task data
    try:
        data_dir = ensure_task_data(task_id, cache_dir)
    except Exception as e:
        return _finalize(
            run_id, task_id, agent, work_dir,
            AgentResult(False, None, 0.0, "", "", f"task prep failed: {e}"),
            None,
        )

    # 2. Run agent
    agent_result = agent.run(data_dir=data_dir, work_dir=work_dir, timeout_s=timeout_s)

    # 3. Grade if agent produced a submission
    grade_result: GradeResult | None = None
    if agent_result.success and agent_result.submission_path:
        grade_result = grade(task_id, agent_result.submission_path)

    return _finalize(run_id, task_id, agent, work_dir, agent_result, grade_result)


def _finalize(
    run_id: str,
    task_id: str,
    agent: Agent,
    work_dir: Path,
    agent_result: AgentResult,
    grade_result: GradeResult | None,
) -> TaskRun:
    run = TaskRun(
        run_id=run_id, task_id=task_id, agent_name=agent.name,
        agent_result=agent_result, grade_result=grade_result,
        work_dir=work_dir, created_at=time.time(),
    )
    (work_dir / "result.json").write_text(json.dumps(run.to_dict(), indent=2))
    return run
