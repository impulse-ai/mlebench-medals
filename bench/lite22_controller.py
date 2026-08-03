"""Sequential MCP/Impulse controller for MLE-bench Lite-22 runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters.task_detection import _find_sample_submission
from .agents import Agent
from .grader import GradeResult, grade
from .run import DEFAULT_CACHE_DIR, build_agent
from .runner import ensure_task_data
from .splits.lite22 import get_lite22_tasks


MCP_UNSUPPORTED_MODALITY_TASKS = {
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "leaf-classification",
    "mlsp-2013-birds",
    "plant-pathology-2020-fgvc7",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "the-icml-2013-whale-challenge-right-whale-redux",
}


MCP_OVERSIZE_DATASET_TASKS = {
    "jigsaw-toxic-comment-classification-challenge",
    "new-york-city-taxi-fare-prediction",
    "random-acts-of-pizza",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
}


@dataclass(frozen=True)
class Lite22TaskPaths:
    task_dir: Path
    trace_jsonl: Path
    request_json: Path
    session_json: Path
    artifacts_json: Path
    submission_csv: Path
    grade_json: Path
    result_json: Path
    model_dir: Path
    logs_dir: Path


def prepare_task_dir(root: Path, task_id: str) -> Lite22TaskPaths:
    task_dir = root / task_id
    model_dir = task_dir / "model"
    logs_dir = task_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return Lite22TaskPaths(
        task_dir=task_dir,
        trace_jsonl=task_dir / "trace.jsonl",
        request_json=task_dir / "request.json",
        session_json=task_dir / "session.json",
        artifacts_json=task_dir / "artifacts.json",
        submission_csv=task_dir / "submission.csv",
        grade_json=task_dir / "grade.json",
        result_json=task_dir / "result.json",
        model_dir=model_dir,
        logs_dir=logs_dir,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_trace(path: Path, event: str, **data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _utc_now(), "event": event, **data}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_task_result(paths: Lite22TaskPaths, payload: dict[str, Any]) -> None:
    paths.result_json.parent.mkdir(parents=True, exist_ok=True)
    out = {**payload, "terminal": True}
    paths.result_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def _artifact_entry(paths: Lite22TaskPaths, path: Path, kind: str) -> dict[str, Any]:
    try:
        rel = path.relative_to(paths.task_dir)
    except ValueError:
        rel = path
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "relative_path": str(rel),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def write_artifacts_manifest(
    paths: Lite22TaskPaths,
    *,
    task_id: str,
    agent_name: str,
    status: str,
    grade_result: dict[str, Any] | None = None,
) -> None:
    model_payload = {
        "task_id": task_id,
        "agent": agent_name,
        "status": status,
        "model_used": agent_name,
        "grade": grade_result,
        "updated_at": _utc_now(),
        "note": (
            "Concrete model details live in agent-generated artifacts/transcripts "
            "when the agent emits them; this manifest records the benchmark adapter "
            "and terminal status for every challenge."
        ),
    }
    model_manifest = paths.model_dir / "model_manifest.json"
    model_manifest.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")

    candidates = [
        (paths.trace_jsonl, "trace"),
        (paths.request_json, "request"),
        (paths.session_json, "session"),
        (paths.result_json, "result"),
        (paths.grade_json, "grade"),
        (paths.submission_csv, "submission"),
        (paths.logs_dir / "agent_stdout.log", "log"),
        (paths.logs_dir / "agent_stderr.log", "log"),
        (paths.task_dir / "mcp_transcript.json", "mcp_transcript"),
        (model_manifest, "model_manifest"),
    ]
    artifacts = [
        _artifact_entry(paths, path, kind)
        for path, kind in candidates
        if path.exists()
    ]
    paths.artifacts_json.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "agent": agent_name,
                "status": status,
                "updated_at": _utc_now(),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def validate_submission_csv(submission: Path, sample_submission: Path) -> dict[str, Any]:
    with sample_submission.open(newline="", encoding="utf-8") as f:
        sample_rows = list(csv.reader(f))
    with submission.open(newline="", encoding="utf-8") as f:
        submission_rows = list(csv.reader(f))

    if not sample_rows:
        raise ValueError(f"sample submission is empty: {sample_submission}")
    if not submission_rows:
        raise ValueError(f"submission is empty: {submission}")

    sample_header = sample_rows[0]
    submission_header = submission_rows[0]
    if submission_header != sample_header:
        raise ValueError(
            f"submission columns {submission_header!r} do not match "
            f"sample columns {sample_header!r}"
        )

    sample_body = sample_rows[1:]
    submission_body = submission_rows[1:]
    if len(submission_body) != len(sample_body):
        raise ValueError(
            f"submission row count {len(submission_body)} does not match "
            f"sample row count {len(sample_body)}"
        )

    sample_ids = [row[0] for row in sample_body]
    submission_ids = [row[0] for row in submission_body]
    if submission_ids != sample_ids:
        sample_row_keys = [row[1:] for row in sample_body]
        submission_row_keys = [row[1:] for row in submission_body]
        if sample_row_keys != submission_row_keys:
            raise ValueError("submission id values do not match sample submission order")

    return {"rows": len(submission_body), "columns": submission_header}


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


def grade_submission(task_id: str, submission: Path) -> dict[str, Any]:
    return _grade_to_dict(grade(task_id, submission))


def _write_task_bundle(paths: Lite22TaskPaths, task_id: str,
                       payload: dict[str, Any], *, agent_name: str,
                       cache_dir: Path, output_root: Path) -> None:
    """Reproducibility bundle: official submission + steps + README. Never fails the run."""
    try:
        from bench.reporting.bundle import BundleInputs, capture_environment, write_run_bundle

        submission = paths.submission_csv
        grade_payload = None
        if paths.grade_json.exists():
            grade_payload = json.loads(paths.grade_json.read_text())
        steps: list[dict] = []
        trace = paths.task_dir / "trace.jsonl"
        if trace.exists():
            steps = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        code_paths = sorted((paths.task_dir / "ees_core").rglob("strategy_plan.json"))
        command = (f"python -m bench.lite22_controller --agent {agent_name} "
                   f"--cache-dir {cache_dir} "
                   f"--output-root {output_root} --task {task_id}")
        write_run_bundle(paths.task_dir / "bundle", BundleInputs(
            task_id=task_id,
            submission_path=submission,
            grade=grade_payload,
            code_paths=list(code_paths),
            steps=steps,
            command=command,
            environment=capture_environment(),
            notes=f"agent={agent_name}. Full decision trace and per-candidate artifacts "
                  f"live alongside this bundle in the task output directory.",
        ))
    except Exception as exc:  # reporting-only; never fail the run on bundling
        append_trace(paths.task_dir / "trace.jsonl", "bundle_error",
                     error=f"{type(exc).__name__}: {exc}")


def _copy_submission(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _terminal_result_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("terminal") is True


def run_lite22(
    root: Path,
    agent: Agent,
    cache_dir: Path,
    timeout_s: int,
    force: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    tasks = get_lite22_tasks()
    if task_ids is not None:
        known = set(tasks)
        unknown = [task_id for task_id in task_ids if task_id not in known]
        if unknown:
            raise ValueError(f"unknown Lite-22 task id(s): {', '.join(unknown)}")
        tasks = list(task_ids)
    if limit is not None:
        tasks = tasks[:limit]

    results: list[dict[str, Any]] = []
    for index, task_id in enumerate(tasks, start=1):
        paths = prepare_task_dir(root, task_id)
        if not force and _terminal_result_exists(paths.result_json):
            payload = json.loads(paths.result_json.read_text())
            append_trace(paths.trace_jsonl, "skipped_existing_result", task_id=task_id)
            results.append(payload)
            continue

        request_payload = {
            "task_id": task_id,
            "agent": agent.name,
            "timeout_s": timeout_s,
            "cache_dir": str(cache_dir),
            "index": index,
            "total": len(tasks),
        }
        paths.request_json.write_text(json.dumps(request_payload, indent=2) + "\n")
        paths.session_json.write_text(json.dumps({"task_id": task_id, "agent": agent.name}, indent=2) + "\n")
        paths.artifacts_json.write_text(json.dumps({"artifacts": []}, indent=2) + "\n")

        append_trace(paths.trace_jsonl, "task_started", **request_payload)
        try:
            if agent.name == "mcp:http" and task_id in MCP_UNSUPPORTED_MODALITY_TASKS:
                error = (
                    "Current MCP adapter uploads scalar files for chat/tool execution but "
                    "does not yet support image/audio directory or nested archive datasets "
                    "required by this MLE-bench challenge."
                )
                (paths.logs_dir / "agent_stdout.log").write_text("")
                (paths.logs_dir / "agent_stderr.log").write_text(error + "\n")
                append_trace(paths.trace_jsonl, "mcp_modality_unsupported", error=error)
                payload = {
                    **request_payload,
                    "status": "agent_failed",
                    "agent": {
                        "success": False,
                        "wall_time_s": 0.0,
                        "error": error,
                    },
                }
                write_task_result(paths, payload)
                write_artifacts_manifest(
                    paths,
                    task_id=task_id,
                    agent_name=agent.name,
                    status="agent_failed",
                )
                append_trace(paths.trace_jsonl, "task_finished", status="agent_failed")
                results.append(payload | {"terminal": True})
                continue

            if agent.name == "mcp:http" and task_id in MCP_OVERSIZE_DATASET_TASKS:
                error = (
                    "Current MCP endpoint rejects this challenge's prepared text/tabular "
                    "payload because it exceeds the request/upload size limit."
                )
                (paths.logs_dir / "agent_stdout.log").write_text("")
                (paths.logs_dir / "agent_stderr.log").write_text(error + "\n")
                append_trace(paths.trace_jsonl, "mcp_payload_oversize", error=error)
                payload = {
                    **request_payload,
                    "status": "agent_failed",
                    "agent": {
                        "success": False,
                        "wall_time_s": 0.0,
                        "error": error,
                    },
                }
                write_task_result(paths, payload)
                write_artifacts_manifest(
                    paths,
                    task_id=task_id,
                    agent_name=agent.name,
                    status="agent_failed",
                )
                append_trace(paths.trace_jsonl, "task_finished", status="agent_failed")
                results.append(payload | {"terminal": True})
                continue

            data_dir = ensure_task_data(task_id, cache_dir)
            append_trace(paths.trace_jsonl, "data_prepared", data_dir=str(data_dir))

            sample_submission = _find_sample_submission(data_dir)
            if sample_submission is None:
                raise FileNotFoundError(f"No sample submission found in {data_dir}")

            agent_result = agent.run(data_dir=data_dir, work_dir=paths.task_dir, timeout_s=timeout_s)
            (paths.logs_dir / "agent_stdout.log").write_text(agent_result.stdout or "")
            (paths.logs_dir / "agent_stderr.log").write_text(agent_result.stderr or "")
            append_trace(
                paths.trace_jsonl,
                "agent_finished",
                success=agent_result.success,
                wall_time_s=agent_result.wall_time_s,
                error=agent_result.error,
                submission_path=str(agent_result.submission_path) if agent_result.submission_path else None,
            )

            if not agent_result.success or agent_result.submission_path is None:
                payload = {
                    **request_payload,
                    "status": "agent_failed",
                    "agent": {
                        "success": agent_result.success,
                        "wall_time_s": agent_result.wall_time_s,
                        "error": agent_result.error,
                    },
                }
                write_task_result(paths, payload)
                write_artifacts_manifest(
                    paths,
                    task_id=task_id,
                    agent_name=agent.name,
                    status="agent_failed",
                )
                append_trace(paths.trace_jsonl, "task_finished", status="agent_failed")
                results.append(payload | {"terminal": True})
                continue

            _copy_submission(agent_result.submission_path, paths.submission_csv)
            validation = validate_submission_csv(paths.submission_csv, sample_submission)
            append_trace(paths.trace_jsonl, "submission_validated", **validation)

            grade_result = grade_submission(task_id, paths.submission_csv)
            paths.grade_json.write_text(json.dumps(grade_result, indent=2, sort_keys=True) + "\n")
            append_trace(paths.trace_jsonl, "submission_graded", **grade_result)

            status = "graded" if grade_result["success"] else "grade_failed"
            payload = {
                **request_payload,
                "status": status,
                "submission": {
                    "path": str(paths.submission_csv),
                    "validation": validation,
                },
                "agent": {
                    "success": agent_result.success,
                    "wall_time_s": agent_result.wall_time_s,
                    "error": agent_result.error,
                },
                "grade": grade_result,
            }
            write_task_result(paths, payload)
            _write_task_bundle(paths, task_id, payload, agent_name=agent.name,
                             cache_dir=cache_dir, output_root=root)
            write_artifacts_manifest(
                paths,
                task_id=task_id,
                agent_name=agent.name,
                status=status,
                grade_result=grade_result,
            )
            append_trace(paths.trace_jsonl, "task_finished", status=status)
            results.append(payload | {"terminal": True})
        except Exception as exc:
            payload = {
                **request_payload,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_task_result(paths, payload)
            write_artifacts_manifest(
                paths,
                task_id=task_id,
                agent_name=agent.name,
                status="failed",
            )
            append_trace(paths.trace_jsonl, "task_failed", error=payload["error"])
            results.append(payload | {"terminal": True})

    summary_results: list[dict[str, Any]] = []
    for summary_task_id in get_lite22_tasks():
        result_path = root / summary_task_id / "result.json"
        if result_path.exists():
            try:
                summary_results.append(json.loads(result_path.read_text()))
            except Exception:
                continue
    (root / "summary.json").write_text(json.dumps(summary_results, indent=2, sort_keys=True) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MLE-bench Lite-22 through the benchmark-first EES agent.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent", default="ees:standalone")
    parser.add_argument("--impulse-base-url", default="http://localhost:3001")
    parser.add_argument("--mcp-url", default=os.environ.get("IMPULSE_MCP_URL"))
    parser.add_argument("--mcp-api-key", default=os.environ.get("IMPULSE_MCP_API_KEY") or os.environ.get("IMPULSE_API_KEY"))
    parser.add_argument("--mcp-bearer-token", default=os.environ.get("IMPULSE_MCP_BEARER_TOKEN") or os.environ.get("IMPULSE_BEARER_TOKEN"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--timeout", type=int, default=60 * 60 * 4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task", action="append", dest="tasks", help="Run only this Lite-22 task id; repeatable")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    agent = build_agent(
        args.agent,
        base_url=args.impulse_base_url,
        mcp_url=args.mcp_url,
        mcp_api_key=args.mcp_api_key,
        mcp_bearer_token=args.mcp_bearer_token,
    )
    results = run_lite22(
        root=args.output_root,
        agent=agent,
        cache_dir=args.cache_dir,
        timeout_s=args.timeout,
        force=args.force,
        limit=args.limit,
        task_ids=args.tasks,
    )
    failures = [r for r in results if r.get("status") not in {"graded"}]
    print(f"Lite-22 controller wrote {len(results)} result(s) to {args.output_root}")
    print(f"Failures/non-graded: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
