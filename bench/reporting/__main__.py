"""Backfill reproducibility bundles for existing run output directories.

Usage:
  python -m bench.reporting --from-run <output_root>/<task_id> [--command "<cmd>"]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.reporting.bundle import BundleInputs, capture_environment, write_run_bundle


def backfill_run_dir(run_dir: Path, *, command: str) -> Path:
    run_dir = Path(run_dir)
    submission = run_dir / "submission.csv"
    if not submission.exists():
        raise FileNotFoundError(f"no submission.csv in {run_dir}")
    grade = None
    if (run_dir / "grade.json").exists():
        grade = json.loads((run_dir / "grade.json").read_text())
    steps: list[dict] = []
    trace = run_dir / "trace.jsonl"
    if trace.exists():
        steps = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    code_paths = sorted((run_dir / "ees_core").rglob("strategy_plan.json")) if (run_dir / "ees_core").exists() else []
    best_candidate = run_dir / "best_candidate.py"
    if best_candidate.exists():
        code_paths.append(best_candidate)
    return write_run_bundle(run_dir / "bundle", BundleInputs(
        task_id=run_dir.name,
        submission_path=submission,
        grade=grade,
        code_paths=code_paths,
        steps=steps,
        command=command,
        environment=capture_environment(),
        notes="Backfilled from an existing run directory.",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True, help="existing <output_root>/<task_id> dir")
    parser.add_argument("--command", default="(original command not recorded)")
    args = parser.parse_args()
    bundle = backfill_run_dir(Path(args.from_run), command=args.command)
    print(f"bundle written: {bundle}")


if __name__ == "__main__":
    main()
