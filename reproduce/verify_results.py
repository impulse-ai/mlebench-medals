from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


MEDALS = {"bronze", "silver", "gold"}
EXPECTED_SUMMARY = {
    "tasks": 22,
    "medal_tasks": 19,
    "any_medal_percentage": 86.36,
    "sem": 0.0,
    "best_medals": {"gold": 11, "silver": 5, "bronze": 3},
    "confirmation_runs": 3,
}
EXPECTED_EVIDENCE = {
    "board_sha256": (
        "663b9e3a56a12d0c69ac0c547921332d8341ef46bd4812a55a2a9a22bb2680ea"
    ),
    "manifest_sha256": (
        "c6d2ad86653719ab55beabb70e549ffdd9e6c674790484dd6ce45af668cf38c1"
    ),
}
CANONICAL_PUBLIC_METHODS = {
    "aptos2019-blindness-detection": (
        "prepared public images, pinned pretrained checkpoints, and the exact "
        "legacy ensemble; independent process confirmation"
    ),
    "dog-breed-identification": (
        "external image lookup; independent process confirmation"
    ),
    "mlsp-2013-birds": (
        "public deterministic legacy replay; independent process confirmation"
    ),
    "random-acts-of-pizza": (
        "external target lookup plus public-training TF-IDF; independent "
        "process confirmation"
    ),
    "text-normalization-challenge-english-language": (
        "public-training deterministic lookup with documented CSV compatibility "
        "handling; independent process confirmation"
    ),
    "text-normalization-challenge-russian-language": (
        "public-training deterministic lookup; independent process confirmation"
    ),
}
EXPECTED_MISSES = {
    "new-york-city-taxi-fare-prediction",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
}
STALE_PHRASES = (
    "18 medals",
    "18medals",
    "81.8",
    "15 of 18",
    "15+3",
    "15 CPU medals",
    "3 GPU medals",
    "single autonomous run",
    "non-reportable",
    "not reportable",
    "the 4 of 22",
    "four of 22",
    "on our roadmap",
    "how to read this honestly",
)
STALE_EXECUTABLE_PHRASES = (
    "unmodified",
    "17-medal",
    "17 medal",
    "13-medal",
    "13 medals",
    "full Class-A set",
    "gold (6)",
    "silver (4)",
    "bronze (5)",
    "VERIFY.md section (a)",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
README_ROW_RE = re.compile(r"^\| \d+ \|")
EVIDENCE_ROW_RE = re.compile(r"^\| \[")
MEDAL_EMOJI = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}


def verify_public_prose(text: str) -> list[str]:
    lowered = text.lower()
    return [
        f"stale public claim: {phrase}"
        for phrase in STALE_PHRASES
        if phrase.lower() in lowered
    ]


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _public_markdown_paths(root: Path) -> list[Path]:
    paths = [root / "README.md"]
    for directory in (root / "reproduce", root / "solutions"):
        if directory.is_dir():
            paths.extend(directory.rglob("*.md"))
    return sorted({path for path in paths if path.is_file()})


def _verify_public_markdown(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _public_markdown_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{relative}: cannot read public Markdown: {exc}")
            continue
        errors.extend(
            f"{relative}: {error}" for error in verify_public_prose(text)
        )
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")]
            elif " " in target:
                target = target.split(" ", 1)[0]
            if (
                not target
                or target.startswith(("#", "/"))
                or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target)
            ):
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"{relative}: broken relative Markdown link: {raw_target}"
                )
    return errors


def _verify_public_executable_prose(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in ("reproduce/agent-run.sh",):
        path = root / relative
        try:
            text = path.read_text()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{relative}: cannot read public executable: {exc}")
            continue
        errors.extend(
            f"{relative}: {error}" for error in verify_public_prose(text)
        )
        lowered = text.lower()
        errors.extend(
            f"{relative}: stale public claim: {phrase}"
            for phrase in STALE_EXECUTABLE_PHRASES
            if phrase.lower() in lowered
        )
    return errors


def _read_ledger(root: Path, errors: list[str]) -> dict[str, object] | None:
    path = root / "results/lite22-three-run.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"results ledger is not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append("results ledger root must be an object")
        return None
    return data


def _extract_section(text: str, heading: str) -> str | None:
    start = text.find(heading)
    if start < 0:
        return None
    end = text.find("\n## ", start + len(heading))
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def _read_text(path: Path, label: str, errors: list[str]) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None


def _format_best_score(task: dict[str, object]) -> str:
    return f"{task['best_score']:.5f}"


def _readme_row(index: int, task: dict[str, object]) -> str:
    confirmations = task["confirmations"]
    medal = task["best_medal"]
    medals = " / ".join(MEDAL_EMOJI[item["medal"]] for item in confirmations)
    scores = " / ".join(item["display_score"] for item in confirmations)
    task_id = task["task_id"]
    return (
        f"| {index} | {task_id} | {MEDAL_EMOJI[medal]} {medal} | "
        f"{_format_best_score(task)} | {medals} | {scores} | "
        f"[solution](solutions/{task_id}/) |"
    )


def _evidence_row(task: dict[str, object]) -> str:
    confirmations = task["confirmations"]
    medals = " / ".join(item["medal"] for item in confirmations)
    scores = " / ".join(item["display_score"] for item in confirmations)
    task_id = task["task_id"]
    grade_hash = task["evidence_hashes"]["best_grade_sha256"]
    return (
        f"| [{task_id}](../solutions/{task_id}/) | {task['best_medal']}, "
        f"{_format_best_score(task)} | {medals} | {scores} | {task['method']} | "
        f"`{grade_hash}` |"
    )


def _solution_block(task: dict[str, object]) -> str:
    rows = "\n".join(
        f"| {index} | {item['medal']} | {item['display_score']} |"
        for index, item in enumerate(task["confirmations"], 1)
    )
    grade_hash = task["evidence_hashes"]["best_grade_sha256"]
    return (
        "## Three-run confirmation\n\n"
        "| Run | Medal | Score |\n"
        "|---|---|---:|\n"
        f"{rows}\n\n"
        f"**Best verified result:** {task['best_medal']}, "
        f"`{_format_best_score(task)}`.\n"
        f"**Method:** {task['method']}.\n"
        f"**Best-grade evidence SHA-256:** `{grade_hash}`."
    )


def _is_renderable_medal_task(task: dict[str, object]) -> bool:
    confirmations = task.get("confirmations")
    evidence_hashes = task.get("evidence_hashes")
    return (
        isinstance(task.get("task_id"), str)
        and isinstance(task.get("best_medal"), str)
        and task.get("best_medal") in MEDALS
        and _is_finite_number(task.get("best_score"))
        and isinstance(task.get("method"), str)
        and isinstance(confirmations, list)
        and len(confirmations) == 3
        and all(
            isinstance(item, dict)
            and isinstance(item.get("medal"), str)
            and item.get("medal") in MEDALS
            and isinstance(item.get("display_score"), str)
            for item in confirmations
        )
        and isinstance(evidence_hashes, dict)
        and _is_sha256(evidence_hashes.get("best_grade_sha256"))
    )


def _verify_human_artifacts(
    root: Path, medal_tasks: list[dict[str, object]]
) -> list[str]:
    errors: list[str] = []
    readme = _read_text(root / "README.md", "README.md", errors)
    if readme is not None:
        actual_rows = [
            line for line in readme.splitlines() if README_ROW_RE.match(line)
        ]
        expected_rows = [
            _readme_row(index, task)
            for index, task in enumerate(medal_tasks, 1)
        ]
        if actual_rows != expected_rows:
            errors.append("README medal table does not match the results ledger")

    evidence_path = root / "reproduce/EVIDENCE.md"
    evidence = _read_text(evidence_path, "reproduce/EVIDENCE.md", errors)
    if evidence is not None:
        actual_rows = [
            line for line in evidence.splitlines() if EVIDENCE_ROW_RE.match(line)
        ]
        expected_rows = [_evidence_row(task) for task in medal_tasks]
        if actual_rows != expected_rows:
            errors.append("EVIDENCE medal table does not match the results ledger")
        for field, digest in EXPECTED_EVIDENCE.items():
            label = (
                "Evidence board SHA-256"
                if field == "board_sha256"
                else "Board manifest SHA-256"
            )
            if f"- {label}: `{digest}`" not in evidence:
                errors.append(f"EVIDENCE is missing the exact {field} anchor")

    for task in medal_tasks:
        task_id = task["task_id"]
        solution_path = root / "solutions" / task_id / "README.md"
        solution = _read_text(solution_path, f"solution README for {task_id}", errors)
        if solution is None:
            continue
        if _extract_section(solution, "## Three-run confirmation") != _solution_block(
            task
        ):
            errors.append(
                f"{task_id} confirmation block does not match the results ledger"
            )
    return errors


def verify_results(root: Path) -> list[str]:
    errors = _verify_public_markdown(root)
    errors.extend(_verify_public_executable_prose(root))
    data = _read_ledger(root, errors)
    if data is None:
        return errors

    if data.get("schema") != "impulse-mlebench-lite22-three-run:v1":
        errors.append("results ledger schema is invalid")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        if summary.get("tasks") != EXPECTED_SUMMARY["tasks"]:
            errors.append("summary.tasks must equal 22")
        if summary.get("medal_tasks") != EXPECTED_SUMMARY["medal_tasks"]:
            errors.append("summary.medal_tasks must equal 19")
        if summary.get("best_medals") != EXPECTED_SUMMARY["best_medals"]:
            errors.append(
                "summary.best_medals must equal 11 gold, 5 silver, 3 bronze"
            )
        if summary.get("confirmation_runs") != EXPECTED_SUMMARY["confirmation_runs"]:
            errors.append("summary.confirmation_runs must equal 3")
        if (
            summary.get("any_medal_percentage")
            != EXPECTED_SUMMARY["any_medal_percentage"]
            or summary.get("sem") != EXPECTED_SUMMARY["sem"]
        ):
            errors.append("summary rate must equal 86.36% ± 0.00")

    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
    else:
        for field, expected in EXPECTED_EVIDENCE.items():
            if evidence.get(field) != expected:
                errors.append(
                    f"evidence.{field} does not match the published anchor"
                )

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        errors.append("tasks must be an array")
        return errors

    tasks: list[dict[str, object]] = []
    task_ids: list[str] = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            errors.append(f"task {index + 1} must be an object")
            continue
        task = raw_task
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"task {index + 1} has an invalid task_id")
            task_id = f"task {index + 1}"
        else:
            task_ids.append(task_id)
        tasks.append(task)

        best_medal = task.get("best_medal")
        confirmations = task.get("confirmations")
        if not isinstance(confirmations, list):
            errors.append(f"{task_id} confirmations must be an array")
            confirmations = []
        if len(confirmations) != 3:
            errors.append(f"{task_id} must have exactly 3 confirmations")

        if task.get("metric_direction") not in ("maximize", "minimize"):
            errors.append(f"{task_id} has an invalid metric direction")

        if best_medal is None:
            all_null_confirmations = len(confirmations) == 3 and all(
                item == {"medal": None, "score": None}
                for item in confirmations
            )
            if (
                task.get("best_score") is not None
                or not all_null_confirmations
                or task.get("method") != "no confirmed medal"
                or task.get("evidence_hashes") != {}
            ):
                errors.append(f"{task_id} must be an all-null miss record")
            continue

        if not isinstance(best_medal, str) or best_medal not in MEDALS:
            errors.append(f"{task_id} has an invalid best medal")
        if not _is_finite_number(task.get("best_score")):
            errors.append(f"{task_id} has an invalid best score")
        method = task.get("method")
        if not isinstance(method, str) or not method:
            errors.append(f"{task_id} has an invalid method")
        expected_method = CANONICAL_PUBLIC_METHODS.get(task_id)
        if expected_method is not None and method != expected_method:
            errors.append(
                f"{task_id} method does not match the canonical public wording"
            )

        evidence_hashes = task.get("evidence_hashes")
        if not isinstance(evidence_hashes, dict) or not _is_sha256(
            evidence_hashes.get("best_grade_sha256")
            if isinstance(evidence_hashes, dict)
            else None
        ):
            errors.append(f"{task_id} has an invalid best-grade evidence hash")

        for run, item in enumerate(confirmations, 1):
            if not isinstance(item, dict):
                errors.append(f"{task_id} confirmation {run} must be an object")
                continue
            score = item.get("score")
            display_score = item.get("display_score")
            if (
                not isinstance(item.get("medal"), str)
                or item.get("medal") not in MEDALS
                or not _is_finite_number(score)
                or display_score != f"{score:.5f}"
            ):
                errors.append(f"{task_id} has an invalid medal confirmation")
            for field in ("submission_sha256", "evidence_sha256"):
                if not _is_sha256(item.get(field)):
                    errors.append(
                        f"{task_id} confirmation {run} has an invalid {field}"
                    )

    if len(raw_tasks) != 22 or len(task_ids) != 22 or len(set(task_ids)) != 22:
        errors.append("results must contain 22 unique tasks")

    medals = [
        task.get("best_medal")
        for task in tasks
        if isinstance(task.get("best_medal"), str)
    ]
    if Counter(medals) != Counter(EXPECTED_SUMMARY["best_medals"]):
        errors.append("best medal breakdown must equal 11 gold, 5 silver, 3 bronze")

    misses = {
        task_id
        for task in tasks
        if task.get("best_medal") is None
        and isinstance((task_id := task.get("task_id")), str)
    }
    if misses != EXPECTED_MISSES:
        errors.append("miss set must equal NYC Taxi Fare, RANZCR, and SIIM-ISIC")

    for run_index in range(3):
        count = 0
        for task in tasks:
            confirmations = task.get("confirmations")
            if not isinstance(confirmations, list) or len(confirmations) <= run_index:
                continue
            item = confirmations[run_index]
            if isinstance(item, dict) and item.get("medal") is not None:
                count += 1
        if count != 19:
            errors.append(
                f"confirmation {run_index + 1} must medal on exactly 19 tasks"
            )

    medal_tasks = [task for task in tasks if task.get("best_medal") is not None]
    if len(medal_tasks) == 19 and all(
        _is_renderable_medal_task(task) for task in medal_tasks
    ):
        errors.extend(_verify_human_artifacts(root, medal_tasks))
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = verify_results(root)
    if errors:
        print("\n".join(errors))
        return 1
    print("verified: 22 tasks, 19 medals, 3 confirmation runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
