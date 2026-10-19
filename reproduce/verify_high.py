from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


SCHEMA = "impulse-mlebench-high15:v1"
EVAL_SCHEMA = "impulse-mlebench-high15-eval:v1"
EVAL_LEDGER = "results/high15-eval-three-seed.json"
EVAL_TAG = "eval-high-20261013"
EVAL_COMMIT = "926abe7d0c17c767b564ee303fb7924b412f2133"
EVAL_SEEDS = [101, 202, 303]
EVAL_FLAGS = ("relaunched", "salvage", "recovery")
# What the three-seed evaluation must show (results/high15-eval-three-seed.json).
EXPECTED_EVAL_SUMMARY = {
    "tasks": 15,
    "tasks_run": 11,
    "per_seed_medal_tasks": [10, 10, 10],
    "any_medal_percentage_mean": 66.67,
    "any_medal_percentage_standard_error": 0.0,
    "display": "66.67 ± 0.00",
    "seed_runs": 33,
    "medal_seed_runs": 30,
    "tasks_with_a_medal_in_every_seed": 9,
    "tasks_with_a_medal_in_any_seed": 11,
    "any_seed_percentage": 73.33,
    "best_medals": {"gold": 7, "silver": 2, "bronze": 2},
}
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
    "medals": {"gold": 7, "silver": 2, "bronze": 2},
    "engine_run_medal_tasks": 11,
    "three_seed_evaluation": {
        "ledger": "results/high15-eval-three-seed.json",
        "seeds": [101, 202, 303],
        "medal_tasks_per_seed": [10, 10, 10],
        "any_medal_percentage": 66.67,
        "standard_error": 0.0,
        "display": "66.67 ± 0.00",
    },
}
ENGINE_METHODS = {"Engine run", "Engine run (exploit discovery)"}
EVAL_METHOD_RE = re.compile(r"^Engine run \(three-seed evaluation, seed (\d+)\)$")
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
IMPULSE_CONTEXT_ROWS = (
    "| **Impulse EES, three-seed evaluation (this repository)** | GPT-6-Luna | "
    "**66.67 ± 0.00 (10/15 per seed)** | ✗ |",
    "| **Impulse EES, tasks medaled across the three seeds (this repository)** | "
    "GPT-6-Luna | **73.33 (11/15)** | ✗ |",
)
REQUIRED_PAGE_CLAIMS = (
    "# Impulse AI on MLE-bench High: 66.67% ± 0.00 across three independent seeds",
    "**66.67% ± 0.00 across three independent seeds, against 46.67% for the top "
    "entry on the MLE-bench main leaderboard.**",
    "11 of the 15",
    "7 gold / 2 silver / 2 bronze",
    "30 of the 33 task-seed runs",
    *IMPULSE_CONTEXT_ROWS,
    "2026-04-24 update states that the leaderboard is not taking new submissions",
    "https://github.com/openai/mle-bench#leaderboard",
)
REQUIRED_README_CLAIMS = (
    "[MLE-bench High: 66.67% ± 0.00 across three independent seeds](HIGH.md)",
    "**66.67% ± 0.00 across three independent seeds**",
    "11 of the 15",
    "7 gold / 2 silver / 2 bronze",
    "18 gold / 7 silver / 5 bronze",
    "**66.67% ± 0.00** across three independent seeds · **73.33%** tasks medaled "
    "across the seeds",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PAGE_ROW_RE = re.compile(r"^\| \d+ \|")
EVIDENCE_ROW_RE = re.compile(r"^\| \[")
CONTEXT_HEADING = "## Published High context"
SEED_HEADING = "## Three-seed evaluation"
SEED_ROW_RE = re.compile(r"^\| (\*\*Medals per seed\*\*|[a-z0-9])")


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
    root: Path,
    task: dict[str, object],
    errors: list[str],
    eval_seeds: dict[str, dict[int, dict[str, object]]] | None = None,
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

    eval_match = EVAL_METHOD_RE.fullmatch(method) if isinstance(method, str) else None
    if method in ENGINE_METHODS or eval_match:
        if task.get("llm") != ENGINE_LLM or task.get("llm_record") not in (
            "run manifest",
            "run LLM transcript",
        ):
            errors.append(f"{task_id} engine run must record {ENGINE_LLM}")
        hours = task.get("wall_time_hours")
        if not _is_number(hours) or not 0 < hours < WALL_TIME_LIMIT_HOURS:
            errors.append(f"{task_id} engine run must finish inside 24 hours")
    else:
        errors.append(f"{task_id} has an invalid method")

    run_id = task.get("run_id")
    pattern = r"high15-eval-\d{2}" if eval_match else r"high15-engine-\d{2}"
    if not isinstance(run_id, str) or not re.fullmatch(pattern, run_id):
        errors.append(f"{task_id} has an invalid run id")
    evaluation = task.get("evaluation")
    if eval_match:
        seed = int(eval_match.group(1))
        record = (eval_seeds or {}).get(task_id, {}).get(seed)
        submission = task.get("submission")
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("ledger") != EVAL_LEDGER
            or evaluation.get("seed") != seed
            or record is None
            or evaluation.get("campaign_id") != record.get("campaign_id")
            or record.get("score") != score
            or record.get("medal") != medal
            or record.get("wall_time_hours") != task.get("wall_time_hours")
            or record.get("graded_at") != task.get("graded_at")
            or not isinstance(submission, dict)
            or record.get("submission_sha256") != submission.get("sha256")
        ):
            errors.append(
                f"{task_id} evaluation row does not match the three-seed ledger"
            )
    elif evaluation is not None:
        errors.append(f"{task_id} records an evaluation seed without saying so")
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


def seed_cell(record: dict[str, object], task_run: bool) -> str:
    if not task_run:
        return "not run (miss)"
    medal = record["medal"]
    if medal is None:
        return f"{record['display_score']} (no medal)"
    return f"{MEDAL_EMOJI[medal]} {record['display_score']}"


def seed_rows(evaluation: dict[str, object]) -> list[str]:
    """The per-seed table rows of HIGH.md, in ledger order, then the totals row."""
    rows = []
    for task in evaluation["tasks"]:
        seeds = task["seeds"]
        cells = " | ".join(seed_cell(record, task["run"]) for record in seeds)
        medaled = sum(record["medal"] is not None for record in seeds)
        rows.append(f"| {task['task_id']} | {cells} | {medaled}/{len(seeds)} |")
    per_seed = " | ".join(
        f"**{item['medal_tasks']}/15**" for item in evaluation["summary"]["per_seed"]
    )
    rows.append(f"| **Medals per seed** | {per_seed} | |")
    return rows


def _verify_pages(
    root: Path,
    medal_tasks: list[dict[str, object]],
    context: list[dict[str, object]],
    evaluation: dict[str, object] | None = None,
) -> list[str]:
    errors: list[str] = []
    page = _read(root / "HIGH.md", "HIGH.md", errors)
    if page is not None and evaluation is not None:
        section = _section(page, SEED_HEADING) or ""
        rows = [line for line in section.splitlines() if SEED_ROW_RE.match(line)]
        if rows != seed_rows(evaluation):
            errors.append("High per-seed table does not match the three-seed ledger")
    if page is not None:
        rows = [line for line in page.splitlines() if PAGE_ROW_RE.match(line)]
        if rows != [page_row(i, task) for i, task in enumerate(medal_tasks, 1)]:
            errors.append("High medal table does not match the results ledger")
        for claim in REQUIRED_PAGE_CLAIMS:
            if claim not in page:
                errors.append(f"HIGH.md is missing the public claim: {claim}")
        section = _section(page, CONTEXT_HEADING) or ""
        published = [
            line
            for line in section.splitlines()
            if line.startswith("| ") and "±" in line and "Impulse" not in line
        ]
        if published != [context_row(entry) for entry in context]:
            errors.append(
                "High published-context table does not match the results ledger"
            )
        minimize_rows = [
            str(i)
            for i, task in enumerate(medal_tasks, 1)
            if task["metric_direction"] == "minimize"
        ]
        if minimize_rows:
            listed = ", ".join(minimize_rows[:-1]) + f" and {minimize_rows[-1]}"
            if f"Lower is better for rows {listed};" not in page:
                errors.append("HIGH.md lower-is-better rows do not match the ledger")

    # HARD.md is the page's former path; it stays as a pointer for old links.
    moved = _read(root / "HARD.md", "HARD.md", errors)
    if moved is not None and "[HIGH.md](HIGH.md)" not in moved:
        errors.append("HARD.md must point to HIGH.md")

    readme = _read(root / "README.md", "README.md", errors)
    if readme is not None:
        for claim in REQUIRED_README_CLAIMS:
            if claim not in readme:
                errors.append(f"README.md is missing the High headline: {claim}")

    evidence = _read(
        root / "reproduce/EVIDENCE_HIGH.md", "reproduce/EVIDENCE_HIGH.md", errors
    )
    if evidence is not None:
        rows = [line for line in evidence.splitlines() if EVIDENCE_ROW_RE.match(line)]
        if rows != [evidence_row(task) for task in medal_tasks]:
            errors.append("High evidence table does not match the results ledger")

    checksums = _read(
        root / "reproduce/high15_submission_checksums.txt",
        "reproduce/high15_submission_checksums.txt",
        errors,
    )
    if checksums is not None:
        expected = "".join(
            f"{task['submission']['sha256']}  {task['submission']['path']}\n"
            for task in medal_tasks
        )
        if checksums != expected:
            errors.append("High checksum file does not match the results ledger")

    for task in medal_tasks:
        task_id = task["task_id"]
        relative = f"solutions/{task_id}/README.md"
        text = _read(root / relative, relative, errors)
        if text is None:
            continue
        if _section(text, "## Result") != result_block(task):
            errors.append(f"{task_id} result block does not match the results ledger")
    return errors


def _seed_flags_ok(record: dict[str, object], seed: int) -> bool:
    flags = record.get("flags")
    campaign = record.get("campaign_id")
    if not isinstance(flags, list) or not set(flags) <= set(EVAL_FLAGS):
        return False
    if len(set(flags)) != len(flags) or not isinstance(campaign, str):
        return False
    relaunched = "relaunched" in flags
    suffix = f"-s{seed}-relaunch1" if relaunched else f"-s{seed}"
    cause = record.get("relaunch_cause")
    return (
        campaign.endswith(suffix)
        and EVAL_TAG in campaign
        and (isinstance(cause, str) and cause.strip() != "") == relaunched
    )


def verify_evaluation(
    root: Path, high_tasks: dict[str, dict[str, object]]
) -> tuple[list[str], dict[str, object] | None]:
    """Check the three-seed ledger. Returns errors and the ledger when usable."""
    errors: list[str] = []
    try:
        data = json.loads((root / EVAL_LEDGER).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"three-seed ledger is not valid JSON: {exc}"], None
    if not isinstance(data, dict):
        return ["three-seed ledger root must be an object"], None
    if (
        data.get("schema") != EVAL_SCHEMA
        or data.get("code_tag") != EVAL_TAG
        or data.get("code_commit") != EVAL_COMMIT
        or data.get("seeds") != EVAL_SEEDS
        or data.get("llm") != ENGINE_LLM
        or data.get("time_limit_hours") != WALL_TIME_LIMIT_HOURS
        or not isinstance(data.get("rule"), str)
        or "No relaunch after a score exists" not in data["rule"]
        or not isinstance(data.get("flag_definitions"), dict)
        or set(data["flag_definitions"]) != set(EVAL_FLAGS)
    ):
        errors.append("three-seed ledger header (schema, tag, commit, seeds, rule) is invalid")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(t, dict) for t in tasks):
        return errors + ["three-seed tasks must be an array of objects"], None
    ids = [task.get("task_id") for task in tasks]
    if len(ids) != 15 or set(ids) != HIGH_SPLIT:
        return errors + ["three-seed ledger must contain the 15 High-split tasks"], None

    usable = True
    for task in tasks:
        task_id = task["task_id"]
        seeds = task.get("seeds")
        if not isinstance(seeds, list) or [
            r.get("seed") if isinstance(r, dict) else None for r in seeds
        ] != EVAL_SEEDS:
            errors.append(f"{task_id} must list seeds 101, 202 and 303 once each")
            usable = False
            continue
        if task_id in EXPECTED_MISSES:
            if task.get("run") is not False or any(
                r != {"seed": r["seed"], "campaign_id": None, "score": None,
                      "medal": None, "grade_count": 0}
                for r in seeds
            ):
                errors.append(f"{task_id} must be recorded as not run, a miss in every seed")
            continue
        if task.get("run") is not True:
            errors.append(f"{task_id} must be recorded as run in the evaluation")
            usable = False
            continue
        high = high_tasks.get(task_id, {})
        thresholds = task.get("thresholds")
        direction = task.get("metric_direction")
        if (
            thresholds != high.get("thresholds")
            or direction != high.get("metric_direction")
            or task.get("display_thresholds") != high.get("display_thresholds")
        ):
            errors.append(f"{task_id} medal lines differ from the High ledger")
            usable = False
            continue
        for record in seeds:
            seed = record["seed"]
            score = record.get("score")
            medal = record.get("medal")
            hours = record.get("wall_time_hours")
            if (
                not _is_number(score)
                or medal not in (*MEDALS, None)
                or medal_for(score, thresholds, direction) != medal
                or _display_number(record.get("display_score")) != score
            ):
                errors.append(
                    f"{task_id} seed {seed} medal does not follow from its score and lines"
                )
                usable = False
            if record.get("grade_count") != 1:
                errors.append(f"{task_id} seed {seed} must be graded exactly once")
            if not _is_number(hours) or not 0 < hours < WALL_TIME_LIMIT_HOURS:
                errors.append(f"{task_id} seed {seed} must finish inside 24 hours")
            if not _is_sha256(record.get("submission_sha256")):
                errors.append(f"{task_id} seed {seed} has an invalid submission hash")
            graded = record.get("graded_at")
            if not isinstance(graded, str) or not DATE_RE.fullmatch(graded):
                errors.append(f"{task_id} seed {seed} has an invalid grading date")
            if not _seed_flags_ok(record, seed):
                errors.append(f"{task_id} seed {seed} has invalid flags or campaign id")

    campaigns = [r.get("campaign_id") for t in tasks if t.get("run") for r in t["seeds"]]
    if len(set(campaigns)) != len(campaigns):
        errors.append("three-seed campaign ids must be unique")

    summary = data.get("summary")
    if not usable or not isinstance(summary, dict):
        return errors + (["three-seed summary is missing"] if usable else []), None

    order = {name: rank for rank, name in enumerate(MEDALS)}
    per_seed = [
        sum(t["seeds"][i]["medal"] is not None for t in tasks) for i in range(3)
    ]
    rates = [100 * count / 15 for count in per_seed]
    mean = sum(rates) / 3
    sem = math.sqrt(sum((rate - mean) ** 2 for rate in rates) / 2) / math.sqrt(3)
    best = {
        t["task_id"]: min((r["medal"] for r in t["seeds"] if r["medal"]), key=order.get)
        for t in tasks
        if any(r["medal"] for r in t["seeds"])
    }
    every = [t for t in tasks if all(r["medal"] for r in t["seeds"])]
    computed = {
        "tasks": 15,
        "tasks_run": sum(t["run"] for t in tasks),
        "per_seed_medal_tasks": per_seed,
        "any_medal_percentage_mean": round(mean, 2),
        "any_medal_percentage_standard_error": round(sem, 2),
        "display": f"{mean:.2f} ± {sem:.2f}",
        "seed_runs": len(campaigns),
        "medal_seed_runs": sum(per_seed),
        "tasks_with_a_medal_in_every_seed": len(every),
        "tasks_with_a_medal_in_any_seed": len(best),
        "any_seed_percentage": round(100 * len(best) / 15, 2),
        "best_medals": {name: list(best.values()).count(name) for name in MEDALS},
    }
    if computed != EXPECTED_EVAL_SUMMARY:
        errors.append(
            "three-seed results must equal 10/15 in every seed (66.67 ± 0.00), "
            "30 of 33 task-seed runs, and 11/15 across the seeds (7 / 2 / 2)"
        )
    listed = summary.get("per_seed")
    listed = listed if isinstance(listed, list) else []
    stated = {
        **{key: summary.get(key) for key in computed if key != "per_seed_medal_tasks"},
        "per_seed_medal_tasks": [
            item.get("medal_tasks") if isinstance(item, dict) else None
            for item in listed
        ],
    }
    seeds_listed = [item.get("seed") if isinstance(item, dict) else None for item in listed]
    if stated != computed or seeds_listed != EVAL_SEEDS:
        errors.append("three-seed summary does not match its task-seed records")
    if summary.get("tasks_not_run") != [
        t["task_id"] for t in tasks if not t["run"]
    ]:
        errors.append("three-seed summary must list the tasks that were not run")
    for task_id, medal in best.items():
        if high_tasks.get(task_id, {}).get("medal") != medal:
            errors.append(
                f"{task_id} High medal differs from its best three-seed medal"
            )
    return errors, data


def verify_high(root: Path) -> list[str]:
    errors: list[str] = []
    path = root / "results/high15.json"
    try:
        raw_text = path.read_text()
        data = json.loads(raw_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"High results ledger is not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return ["High results ledger root must be an object"]

    if data.get("schema") != SCHEMA:
        errors.append("High results ledger schema is invalid")
    if data.get("summary") != EXPECTED_SUMMARY:
        errors.append(
            "High summary must equal 11 of 15 (73.33%), 7 gold / 2 silver / "
            "2 bronze, all engine runs, and 66.67 ± 0.00 across three seeds"
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
        errors.append("High tasks must be an array")
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
        errors.append("High results must contain the 15 High-split tasks")
        return errors

    eval_errors, evaluation = verify_evaluation(
        root, {task["task_id"]: task for task in tasks}
    )
    errors.extend(eval_errors)
    eval_seeds = (
        {
            task["task_id"]: {record["seed"]: record for record in task["seeds"]}
            for task in evaluation["tasks"]
        }
        if evaluation is not None
        else None
    )

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
        renderable = _verify_medal_task(root, task, errors, eval_seeds) and renderable

    misses = {task["task_id"] for task in tasks if task.get("medal") is None}
    if misses != EXPECTED_MISSES:
        errors.append(
            "High miss set must equal SIIM COVID-19, Cervical Spine, Vesuvius, "
            "and BMS"
        )
    medals = Counter(task.get("medal") for task in medal_tasks)
    if medals != Counter(EXPECTED_SUMMARY["medals"]):
        errors.append("High medal breakdown must equal 7 gold, 2 silver, 2 bronze")
    engine = [
        task
        for task in medal_tasks
        if task.get("method") in ENGINE_METHODS
        or EVAL_METHOD_RE.fullmatch(str(task.get("method")))
    ]
    if len(engine) != 11:
        errors.append("High results must hold 11 engine runs")
    if round(100 * len(medal_tasks) / 15, 2) != 73.33:
        errors.append("High rate must equal 73.33% (11/15)")
    run_ids = [task.get("run_id") for task in medal_tasks]
    if len(set(run_ids)) != len(run_ids):
        errors.append("High run ids must be unique")

    if renderable and len(medal_tasks) == 11:
        errors.extend(_verify_pages(root, medal_tasks, context, evaluation))
    return errors
