"""Console + JSON reporting for bench runs."""
from __future__ import annotations

import json
from pathlib import Path

from .runner import TaskRun


MEDAL_EMOJI = {"gold": "G", "silver": "S", "bronze": "B", "none": " "}


def print_task_result(run: TaskRun) -> None:
    """One summary line per task with medal status."""
    ar, gr = run.agent_result, run.grade_result
    if not ar.success:
        print(f"  [FAIL] {run.task_id:50s}  agent error: {ar.error}")
        return
    if not gr or not gr.success:
        err = gr.error if gr else "no grade"
        print(f"  [GRADE-FAIL] {run.task_id:50s}  {err}")
        return
    medal = gr.medal()
    score_str = f"{gr.score:.5f}" if gr.score is not None else "n/a"
    print(
        f"  [{MEDAL_EMOJI[medal]}] {run.task_id:50s}  "
        f"score={score_str:>10s}  medal={medal:6s}  ({ar.wall_time_s:.0f}s)"
    )
    if medal == "none" and gr.bronze_threshold is not None and gr.score is not None:
        # Bench metrics aren't always "higher is better" — gap may be negative; still informative.
        gap = gr.bronze_threshold - gr.score
        print(f"      bronze threshold: {gr.bronze_threshold:.5f}  (gap: {gap:+.5f})")


def print_split_summary(runs: list[TaskRun]) -> None:
    n = len(runs)
    if n == 0:
        return
    n_success = sum(1 for r in runs if r.agent_result.success)
    n_graded = sum(1 for r in runs if r.grade_result and r.grade_result.success)
    medals = {"gold": 0, "silver": 0, "bronze": 0, "none": 0}
    for r in runs:
        if r.grade_result and r.grade_result.success:
            medals[r.grade_result.medal()] += 1
    any_medal = medals["gold"] + medals["silver"] + medals["bronze"]
    print()
    print("=" * 72)
    print(f"Split summary  ({n} tasks)")
    print(f"  Agent success: {n_success}/{n}  ({100 * n_success / n:.1f}%)")
    print(f"  Graded:        {n_graded}/{n}")
    print(f"  Gold:    {medals['gold']}")
    print(f"  Silver:  {medals['silver']}")
    print(f"  Bronze:  {medals['bronze']}")
    print(f"  None:    {medals['none']}")
    print(f"  Any-medal rate: {100 * any_medal / n:.1f}%")
    print("=" * 72)


def write_split_json(runs: list[TaskRun], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.to_dict() for r in runs], indent=2))
