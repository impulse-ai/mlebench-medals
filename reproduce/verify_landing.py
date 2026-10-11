from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_charts  # noqa: E402

WALKTHROUGH = (
    "**[Interactive walkthrough: ten problems, the data, the model the engine "
    "built, and the result](https://mlebench.impulselabs.ai)**"
)
HIGH_IMAGES = (
    "](assets/high-any-medal-light.svg#gh-light-mode-only)",
    "](assets/high-any-medal-dark.svg#gh-dark-mode-only)",
)
LITE_IMAGES = (
    "](assets/lite-any-medal-light.svg#gh-light-mode-only)",
    "](assets/lite-any-medal-dark.svg#gh-dark-mode-only)",
)
LITE_HEADING = "## Published Lite context"
LITE_SOURCE = "https://github.com/openai/mle-bench#leaderboard"


def _lite_row(entry: dict[str, object]) -> str:
    mark = "✓" if entry["source_code_available"] else "✗"
    return f"| {entry['agent']} | {entry['llm']} | {entry['lite_any_medal']} | {mark} |"


def _load(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def verify_landing(root: Path) -> list[str]:
    errors: list[str] = []
    try:
        readme = (root / "README.md").read_text()
        high_page = (root / "HIGH.md").read_text()
    except (OSError, UnicodeError) as exc:
        return [f"cannot read the landing pages: {exc}"]

    high = _load(root / "results/high15.json")
    lite = _load(root / "results/lite22-three-run.json")
    context = _load(root / "results/published-lite-context.json")
    if high is None or lite is None or context is None:
        return ["landing checks need the three results files as valid JSON"]

    try:
        entries = context["entries"]
        lite_rows = [_lite_row(entry) for entry in entries]
        lite_top = max(float(e["lite_any_medal"].split("±")[0]) for e in entries)
        high_top = max(
            float(e["high_any_medal"].split("±")[0])
            for e in high["published_context"]["entries"]
        )
        high_summary = high["summary"]
        lite_summary = lite["summary"]
        high_medals = high_summary["medals"]
        lite_medals = lite_summary["best_medals"]
        total = high_summary["medal_tasks"] + lite_summary["medal_tasks"]
        competitions = high_summary["tasks"] + lite_summary["tasks"]
        tally = " / ".join(
            f"{high_medals[m] + lite_medals[m]} {m}" for m in ("gold", "silver", "bronze")
        )
        engine = f"{high_summary['engine_run_percentage']:.2f}%"
        lite_rate = f"{lite_summary['any_medal_percentage']:.2f}%"
        stale = make_charts.stale(root)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"landing checks cannot read the results files: {exc!r}"]

    if context.get("source") != LITE_SOURCE:
        errors.append("published Lite context source record is invalid")
    start = readme.find(LITE_HEADING)
    end = readme.find("\n## ", start + 1)
    section = readme[start : len(readme) if end < 0 else end] if start >= 0 else ""
    published = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and "±" in line and "Impulse" not in line
    ]
    if published != lite_rows:
        errors.append("published Lite context file does not match the README table")

    errors.extend(f"chart is out of date with the ledgers: {name}" for name in stale)

    at_a_glance = (
        f"| {total} medals | High: {engine} | Lite: {lite_rate} |\n"
        "|:---:|:---:|:---:|\n"
        f"| on {competitions} competitions | engine runs, against {high_top:.2f}% for "
        "the top entry on the main leaderboard | three confirmed runs, against "
        f"{lite_top:.2f}% for the top entry on the main leaderboard |"
    )
    readme_claims = (
        WALKTHROUGH,
        f"**{total} medals on {competitions} competitions**",
        f"**{tally}**",
        at_a_glance,
        "- **Medium (38 competitions)** is the next split to be published here.",
        *HIGH_IMAGES,
        *LITE_IMAGES,
    )
    for claim in readme_claims:
        if claim not in readme:
            errors.append(f"README.md landing section is missing: {claim}")
    for claim in (WALKTHROUGH, *HIGH_IMAGES):
        if claim not in high_page:
            errors.append(f"HIGH.md is missing: {claim}")
    return errors
