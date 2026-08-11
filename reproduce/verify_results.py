from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path


MEDAL_ORDER = {None: 0, "bronze": 1, "silver": 2, "gold": 3}
STALE_PHRASES = (
    "18 medals",
    "81.8%",
    "single autonomous run",
    "The 4 of 22",
    "not reportable",
    "on our roadmap",
    "how to read this honestly",
)


def verify_public_prose(text: str) -> list[str]:
    return [
        f"stale public claim: {phrase}"
        for phrase in STALE_PHRASES
        if phrase.lower() in text.lower()
    ]


def verify_results(root: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads((root / "results/lite22-three-run.json").read_text())
    tasks = data["tasks"]
    summary = data["summary"]
    if summary["medal_tasks"] != 19:
        errors.append("summary.medal_tasks must equal 19")
    if len(tasks) != 22 or len({task["task_id"] for task in tasks}) != 22:
        errors.append("results must contain 22 unique tasks")
    medals = [task["best_medal"] for task in tasks if task["best_medal"]]
    if Counter(medals) != Counter({"gold": 11, "silver": 5, "bronze": 3}):
        errors.append("best medal breakdown must equal 11 gold, 5 silver, 3 bronze")
    for task in tasks:
        confirmations = task["confirmations"]
        if len(confirmations) != 3:
            errors.append(f"{task['task_id']} must have exactly 3 confirmations")
            continue
        if task["best_medal"]:
            if not all(
                item["medal"] in MEDAL_ORDER
                and item["medal"] is not None
                and isinstance(item["score"], (int, float))
                and math.isfinite(item["score"])
                for item in confirmations
            ):
                errors.append(f"{task['task_id']} has an invalid medal confirmation")
            if not (root / "solutions" / task["task_id"] / "README.md").is_file():
                errors.append(f"missing solution README for {task['task_id']}")
    for index in range(3):
        count = sum(task["confirmations"][index]["medal"] is not None for task in tasks)
        if count != 19:
            errors.append(f"confirmation {index + 1} must medal on exactly 19 tasks")
    expected_rate = round(100 * 19 / 22, 2)
    if summary["any_medal_percentage"] != expected_rate or summary["sem"] != 0.0:
        errors.append("summary rate must equal 86.36% ± 0.00")
    for path in (root / "README.md", root / "reproduce/EVIDENCE.md"):
        errors.extend(verify_public_prose(path.read_text()))
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
