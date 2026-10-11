from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


SCHEMA = "impulse-mlebench-hard15:v1"
MEDALS = ("gold", "silver", "bronze")
MEDAL_EMOJI = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}
HIGH_SPLIT = {
    "3d-object-detection-for-autonomous-vehicles",
    "bms-molecular-translation",
    "google-research-identify-contrails-reduce-global-warming",
    "hms-harmful-brain-activity-classification",
    "iwildcam-2019-fgvc6",
    "nfl-player-contact-detection",
    "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "siim-covid19-detection",
    "smartphone-decimeter-2022",
    "stanford-covid-vaccine",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
}
EXPECTED_MISSES = {
    "bms-molecular-translation",
    "rsna-2022-cervical-spine-fracture-detection",
    "siim-covid19-detection",
    "vesuvius-challenge-ink-detection",
}
EXPECTED_SUMMARY = {
    "tasks": 15,
    "medal_tasks": 11,
    "any_medal_percentage": 73.33,
    "medals": {"gold": 7, "silver": 1, "bronze": 3},
    "engine_run_medal_tasks": 9,
    "engine_run_percentage": 60.0,
}
ENGINE_METHODS = {"Engine run", "Engine run (exploit discovery)"}
PIPELINE_METHOD = "Impulse ML pipeline"
ENGINE_LLM = "GPT-6-Luna"
WALL_TIME_LIMIT_HOURS = 24.0
# High-split figures published on the MLE-bench main leaderboard
# (https://github.com/openai/mle-bench#leaderboard), retrieved 2026-10-11.
PUBLISHED_HIGH = [
    {
        "agent": "AIBuildAI",
        "llm": "Claude-Opus-4.6",
        "high_any_medal": "46.67 ± 0.00",
        "source_code_available": False,
    },
    {
        "agent": "CAIR MARS+",
        "llm": "Gemini-3-Pro-Preview",
        "high_any_medal": "44.44 ± 2.22",
        "source_code_available": False,
    },
    {
        "agent": "Famou-Agent 2.0",
        "llm": "Gemini-3-Pro-Preview",
        "high_any_medal": "42.22 ± 2.22",
        "source_code_available": False,
    },
    {
        "agent": "MLEvolve",
        "llm": "Gemini-3-Pro-Preview",
        "high_any_medal": "42.22 ± 2.22",
        "source_code_available": True,
    },
    {
        "agent": "ML-Master 2.0",
        "llm": "Deepseek-V3.2-Speciale",
        "high_any_medal": "42.22 ± 2.22",
        "source_code_available": False,
    },
    {
        "agent": "PiEvolve (Fractal AI Research)",
        "llm": "Gemini-3-Pro-Preview",
        "high_any_medal": "40.0 ± 0.00",
        "source_code_available": False,
    },
    {
        "agent": "AIDE (OpenAI's reference agent)",
        "llm": "o1-preview",
        "high_any_medal": "11.67 ± 1.27",
        "source_code_available": True,
    },
]
REQUIRED_PAGE_CLAIMS = (
    "11 medals on MLE-bench HARD",
    "11 of the 15",
    "7 gold / 1 silver / 3 bronze",
    "60.00% on HARD",
    "46.67% for the top entry on the MLE-bench main leaderboard",
    "| **Impulse AI, all verified medals (this repository)** | "
    "GPT-6-Luna (engine runs) | **73.33 (11/15)** | ✗ |",
    "| **Impulse EES, engine runs (this repository)** | GPT-6-Luna | "
    "**60.00 (9/15)** | ✗ |",
    "2026-04-24 update states that the leaderboard is not taking new submissions",
    "https://github.com/openai/mle-bench#leaderboard",
)
REQUIRED_README_CLAIMS = (
    "[11 medals on MLE-bench HARD](HARD.md)",
    "11 of the 15",
    "7 gold / 1 silver / 3 bronze",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PAGE_ROW_RE = re.compile(r"^\| \d+ \|")
EVIDENCE_ROW_RE = re.compile(r"^\| \[")
CONTEXT_HEADING = "## Published HARD context"


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_number(text: object) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def medal_for(score: float, thresholds: dict[str, float], direction: str) -> str | None:
    for medal in MEDALS:
        line = thresholds[medal]
        if (score <= line) if direction == "minimize" else (score >= line):
            return medal
    return None


def _wall_time(task: dict[str, object]) -> str:
    hours = task.get("wall_time_hours")
    return f"{hours:.2f}" if _is_number(hours) else "n/a"


def page_row(index: int, task: dict[str, object]) -> str:
    task_id = task["task_id"]
    medal = task["medal"]
    return (
        f"| {index} | {task_id} | {MEDAL_EMOJI[medal]} {medal} | "
        f"{task['display_score']} | {task['display_thresholds']} | "
        f"{task['method']} | [solution](solutions/{task_id}/) |"
    )


def context_row(entry: dict[str, object]) -> str:
    mark = "✓" if entry["source_code_available"] else "✗"
    return (
        f"| {entry['agent']} | {entry['llm']} | {entry['high_any_medal']} | "
        f"{mark} |"
    )


def evidence_row(task: dict[str, object]) -> str:
    task_id = task["task_id"]
    return (
        f"| [{task_id}](../solutions/{task_id}/) | {task['medal']}, "
        f"{task['display_score']} | {task['method']} | "
        f"{task['llm'] or 'n/a'} | {_wall_time(task)} | {task['graded_at']} | "
        f"`{task['submission']['sha256']}` | "
        f"`{task['evidence_hashes']['grade_sha256']}` |"
    )


def result_block(task: dict[str, object]) -> str:
    hours = task.get("wall_time_hours")
    wall = f"{hours:.2f} hours" if _is_number(hours) else "not applicable"
    better = "lower" if task["metric_direction"] == "minimize" else "higher"
    return (
        "## Result\n\n"
        f"**Result:** {task['medal']}, `{task['display_score']}`.\n"
        "**Medal lines (gold / silver / bronze):** "
        f"{task['display_thresholds']}; {better} is better.\n"
        f"**Method:** {task['method']}.\n"
        f"**LLM:** {task['llm'] or 'not applicable'}.\n"
        f"**Run wall time:** {wall}.\n"
        f"**Graded:** {task['graded_at']}.\n"
        f"**Submission SHA-256:** `{task['submission']['sha256']}`.\n"
        "**Grade report SHA-256:** "
        f"`{task['evidence_hashes']['grade_sha256']}`."
    )


def _section(text: str, heading: str) -> str | None:
    start = text.find(heading)
    if start < 0:
        return None
    end = text.find("\n## ", start + len(heading))
    return text[start : len(text) if end < 0 else end].strip()


def _read(path: Path, label: str, errors: list[str]) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None


def _verify_medal_task(
    root: Path, task: dict[str, object], errors: list[str]
) -> bool:
    """Validate one medal record and its files. True when it can be rendered."""
    task_id = task["task_id"]
    before = len(errors)
    medal = task.get("medal")
    score = task.get("score")
    direction = task.get("metric_direction")
    thresholds = task.get("thresholds")
    method = task.get("method")

    if medal not in MEDALS:
        errors.append(f"{task_id} has an invalid medal")
    if not _is_number(score):
        errors.append(f"{task_id} has an invalid score")
    if direction not in ("maximize", "minimize"):
        errors.append(f"{task_id} has an invalid metric direction")
    thresholds_ok = isinstance(thresholds, dict) and all(
        _is_number(thresholds.get(name)) for name in MEDALS
    )
    if not thresholds_ok:
        errors.append(f"{task_id} has invalid medal lines")
    if len(errors) == before and medal_for(score, thresholds, direction) != medal:
        errors.append(f"{task_id} medal does not follow from its score and lines")

    if _display_number(task.get("display_score")) != score:
        errors.append(f"{task_id} display score does not match its score")
    display_lines = task.get("display_thresholds")
    parts = display_lines.split(" / ") if isinstance(display_lines, str) else []
    if not thresholds_ok or [_display_number(part) for part in parts] != [
        thresholds[name] for name in MEDALS
    ]:
        errors.append(f"{task_id} displayed medal lines do not match the ledger")

    if method in ENGINE_METHODS:
        if task.get("llm") != ENGINE_LLM or task.get("llm_record") not in (
            "run manifest",
            "run LLM transcript",
        ):
            errors.append(f"{task_id} engine run must record {ENGINE_LLM}")
        hours = task.get("wall_time_hours")
        if not _is_number(hours) or not 0 < hours < WALL_TIME_LIMIT_HOURS:
            errors.append(f"{task_id} engine run must finish inside 24 hours")
    elif method == PIPELINE_METHOD:
        if any(
            task.get(field) is not None
            for field in ("llm", "llm_record", "wall_time_hours")
        ):
            errors.append(f"{task_id} pipeline result must not record run fields")
    else:
        errors.append(f"{task_id} has an invalid method")

    run_id = task.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"hard15-(engine|pipeline)-\d{2}", run_id
    ):
        errors.append(f"{task_id} has an invalid run id")
    graded_at = task.get("graded_at")
    if not isinstance(graded_at, str) or not DATE_RE.fullmatch(graded_at):
        errors.append(f"{task_id} has an invalid grading date")

    submission = task.get("submission")
    hashes = task.get("evidence_hashes")
    expected_path = f"solutions/{task_id}/submission.csv"
    if (
        not isinstance(submission, dict)
        or submission.get("path") != expected_path
        or not _is_sha256(submission.get("sha256"))
        or type(submission.get("bytes")) is not int
    ):
        errors.append(f"{task_id} has an invalid submission record")
    else:
        path = root / expected_path
        if not path.is_file():
            errors.append(f"{task_id} graded submission is missing")
        elif (
            path.stat().st_size != submission["bytes"]
            or _sha256_file(path) != submission["sha256"]
        ):
            errors.append(
                f"{task_id} graded submission does not match its SHA-256"
            )

    if not isinstance(hashes, dict) or not _is_sha256(hashes.get("grade_sha256")):
        errors.append(f"{task_id} has an invalid grade report hash")
    else:
        grade_path = root / "solutions" / task_id / "grade.json"
        if not grade_path.is_file():
            errors.append(f"{task_id} grade report is missing")
        elif _sha256_file(grade_path) != hashes["grade_sha256"]:
            errors.append(f"{task_id} grade report does not match its SHA-256")
        else:
            try:
                grade = json.loads(grade_path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                grade = None
            consistent = (
                isinstance(grade, dict)
                and thresholds_ok
                and grade.get("competition_id") == task_id
                and grade.get("valid_submission") is True
                and grade.get("score") == score
                and grade.get("is_lower_better") == (direction == "minimize")
                and all(
                    grade.get(f"{name}_threshold") == thresholds[name]
                    for name in MEDALS
                )
                and medal in MEDALS
                and grade.get(f"{medal}_medal") is True
                and grade.get("any_medal") is True
            )
            if not consistent:
                errors.append(f"{task_id} grade report does not match the ledger")
    return len(errors) == before


def _verify_pages(
    root: Path,
    medal_tasks: list[dict[str, object]],
    context: list[dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    page = _read(root / "HARD.md", "HARD.md", errors)
    if page is not None:
        rows = [line for line in page.splitlines() if PAGE_ROW_RE.match(line)]
        if rows != [page_row(i, task) for i, task in enumerate(medal_tasks, 1)]:
            errors.append("HARD medal table does not match the results ledger")
        for claim in REQUIRED_PAGE_CLAIMS:
            if claim not in page:
                errors.append(f"HARD.md is missing the public claim: {claim}")
        section = _section(page, CONTEXT_HEADING) or ""
        published = [
            line
            for line in section.splitlines()
            if line.startswith("| ") and "±" in line
        ]
        if published != [context_row(entry) for entry in context]:
            errors.append(
                "HARD published-context table does not match the results ledger"
            )
        minimize_rows = [
            str(i)
            for i, task in enumerate(medal_tasks, 1)
            if task["metric_direction"] == "minimize"
        ]
        if minimize_rows:
            listed = ", ".join(minimize_rows[:-1]) + f" and {minimize_rows[-1]}"
            if f"Lower is better for rows {listed};" not in page:
                errors.append("HARD.md lower-is-better rows do not match the ledger")

    readme = _read(root / "README.md", "README.md", errors)
    if readme is not None:
        for claim in REQUIRED_README_CLAIMS:
            if claim not in readme:
                errors.append(f"README.md is missing the HARD headline: {claim}")

    evidence = _read(
        root / "reproduce/EVIDENCE_HARD.md", "reproduce/EVIDENCE_HARD.md", errors
    )
    if evidence is not None:
        rows = [line for line in evidence.splitlines() if EVIDENCE_ROW_RE.match(line)]
        if rows != [evidence_row(task) for task in medal_tasks]:
            errors.append("HARD evidence table does not match the results ledger")

    checksums = _read(
        root / "reproduce/hard15_submission_checksums.txt",
        "reproduce/hard15_submission_checksums.txt",
        errors,
    )
    if checksums is not None:
        expected = "".join(
            f"{task['submission']['sha256']}  {task['submission']['path']}\n"
            for task in medal_tasks
        )
        if checksums != expected:
            errors.append("HARD checksum file does not match the results ledger")

    for task in medal_tasks:
        task_id = task["task_id"]
        relative = f"solutions/{task_id}/README.md"
        text = _read(root / relative, relative, errors)
        if text is None:
            continue
        if _section(text, "## Result") != result_block(task):
            errors.append(f"{task_id} result block does not match the results ledger")
    return errors


def verify_hard(root: Path) -> list[str]:
    errors: list[str] = []
    path = root / "results/hard15.json"
    try:
        raw_text = path.read_text()
        data = json.loads(raw_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"HARD results ledger is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["HARD results ledger root must be an object"]

    if data.get("schema") != SCHEMA:
        errors.append("HARD results ledger schema is invalid")
    if data.get("summary") != EXPECTED_SUMMARY:
        errors.append(
            "HARD summary must equal 11 of 15 (73.33%), 7 gold / 1 silver / "
            "3 bronze, and 9 engine runs (60.00%)"
        )

    context_block = data.get("published_context")
    context = context_block.get("entries") if isinstance(context_block, dict) else None
    if context != PUBLISHED_HIGH:
        errors.append("published High-split figures do not match the cited source")
        context = PUBLISHED_HIGH
    if not isinstance(context_block, dict) or (
        context_block.get("source")
        != "https://github.com/openai/mle-bench#leaderboard"
        or context_block.get("leaderboard_update")
        != "04-24-2026: not taking new submissions to the leaderboard"
    ):
        errors.append("published-context source record is invalid")

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        errors.append("HARD tasks must be an array")
        return errors
    tasks = [task for task in raw_tasks if isinstance(task, dict)]
    task_ids = [
        task.get("task_id") for task in tasks if isinstance(task.get("task_id"), str)
    ]
    if (
        len(raw_tasks) != 15
        or len(task_ids) != 15
        or set(task_ids) != HIGH_SPLIT
    ):
        errors.append("HARD results must contain the 15 High-split tasks")
        return errors

    medal_tasks: list[dict[str, object]] = []
    renderable = True
    for task in tasks:
        task_id = task["task_id"]
        if task.get("medal") is None:
            if task != {
                "task_id": task_id,
                "medal": None,
                "score": None,
                "method": "no medal",
                "evidence_hashes": {},
            }:
                errors.append(f"{task_id} must be an all-null miss record")
            continue
        medal_tasks.append(task)
        renderable = _verify_medal_task(root, task, errors) and renderable

    misses = {task["task_id"] for task in tasks if task.get("medal") is None}
    if misses != EXPECTED_MISSES:
        errors.append(
            "HARD miss set must equal SIIM COVID-19, Cervical Spine, Vesuvius, "
            "and BMS"
        )
    medals = Counter(task.get("medal") for task in medal_tasks)
    if medals != Counter(EXPECTED_SUMMARY["medals"]):
        errors.append("HARD medal breakdown must equal 7 gold, 1 silver, 3 bronze")
    engine = [task for task in medal_tasks if task.get("method") in ENGINE_METHODS]
    pipeline = [task for task in medal_tasks if task.get("method") == PIPELINE_METHOD]
    if len(engine) != 9 or len(pipeline) != 2:
        errors.append("HARD results must hold 9 engine runs and 2 pipeline results")
    if (
        round(100 * len(medal_tasks) / 15, 2) != 73.33
        or round(100 * len(engine) / 15, 2) != 60.0
    ):
        errors.append("HARD rates must equal 73.33% (11/15) and 60.00% (9/15)")
    run_ids = [task.get("run_id") for task in medal_tasks]
    if len(set(run_ids)) != len(run_ids):
        errors.append("HARD run ids must be unique")

    if renderable and len(medal_tasks) == 11:
        errors.extend(_verify_pages(root, medal_tasks, context))
    return errors
