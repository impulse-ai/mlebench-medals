"""Build the comparison charts in assets/ from the checked-in ledgers.

Run from the repository root:

    python reproduce/make_charts.py            # write assets/*.svg
    python reproduce/make_charts.py --check    # fail if assets/ is out of date

Each chart is written twice, for light and dark page backgrounds. Every value
comes from results/high15.json, results/lite22-three-run.json and
results/published-lite-context.json; nothing is typed into this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

THEMES = {
    "light": {"ink": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "impulse": "#2a78d6", "other": "#7d848c"},
    "dark": {"ink": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d",
             "impulse": "#3987e5", "other": "#6e7781"},
}
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, "
        "Arial, sans-serif")
WIDTH = 800
LABEL_X = 250          # right edge of the row labels
BAR_X = 262            # x of 0 %
BAR_W = 380            # width of 100 %
ROW = 32
BAR_H = 18
TOP = 50
RUN_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _rate(text: str) -> float:
    return float(text.split("±")[0].strip())


def high_chart(root: Path) -> dict[str, object]:
    data = json.loads((root / "results/high15.json").read_text())
    summary = data["summary"]
    runs = summary["engine_run_medal_tasks"]
    rows = [{
        "label": "Impulse (engine runs)",
        "value": summary["engine_run_percentage"],
        "text": f"{summary['engine_run_percentage']:.2f}% ({runs}/{summary['tasks']})",
        "impulse": True,
    }]
    for entry in data["published_context"]["entries"]:
        rows.append({"label": entry["agent"], "value": _rate(entry["high_any_medal"]),
                     "text": entry["high_any_medal"], "impulse": False})
    return {
        "name": "high-any-medal",
        "title": "MLE-bench High split, any-medal rate (%)",
        "caption": ("Impulse: single runs, each scored once. Others: MLE-bench "
                    "main leaderboard, means over seeds."),
        "rows": rows,
    }


def lite_chart(root: Path) -> dict[str, object]:
    summary = json.loads((root / "results/lite22-three-run.json").read_text())["summary"]
    context = json.loads((root / "results/published-lite-context.json").read_text())
    rows = [{
        "label": "Impulse",
        "value": summary["any_medal_percentage"],
        "text": (f"{summary['any_medal_percentage']:.2f}% "
                 f"({summary['medal_tasks']}/{summary['tasks']})"),
        "impulse": True,
    }]
    for entry in context["entries"]:
        rows.append({"label": entry["agent"], "value": _rate(entry["lite_any_medal"]),
                     "text": entry["lite_any_medal"], "impulse": False})
    return {
        "name": "lite-any-medal",
        "title": "MLE-bench Lite split, any-medal rate (%)",
        "caption": (f"Impulse: {RUN_WORDS[summary['confirmation_runs']]} confirmed runs. Others: "
                    "MLE-bench main leaderboard, means over seeds."),
        "rows": rows,
    }


def _bar(x: float, y: float, width: float, height: float, radius: float = 4) -> str:
    """Bar with a square start on the baseline and a rounded data end."""
    radius = min(radius, width, height / 2)
    return (f"M{x:.1f},{y:.1f} h{width - radius:.1f} "
            f"a{radius:.1f},{radius:.1f} 0 0 1 {radius:.1f},{radius:.1f} "
            f"v{height - 2 * radius:.1f} "
            f"a{radius:.1f},{radius:.1f} 0 0 1 -{radius:.1f},{radius:.1f} "
            f"h-{width - radius:.1f} z")


def render(chart: dict[str, object], theme: str) -> str:
    colors = THEMES[theme]
    rows = chart["rows"]
    plot_bottom = TOP + ROW * len(rows)
    height = plot_bottom + 48
    summary = "; ".join(f"{row['label']} {row['text']}" for row in rows)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" font-family="{FONT}">',
        f"<title>{escape(chart['title'])}</title>",
        f"<desc>{escape(chart['caption'] + ' ' + summary)}</desc>",
        f'<text x="16" y="30" font-size="17" font-weight="600" '
        f'fill="{colors["ink"]}">{escape(chart["title"])}</text>',
    ]
    out.append(f'<line x1="{BAR_X}" y1="{TOP - 4}" x2="{BAR_X}" y2="{plot_bottom + 4}" '
               f'stroke="{colors["grid"]}" stroke-width="1"/>')
    for index, row in enumerate(rows):
        y = TOP + ROW * index + (ROW - BAR_H) / 2
        mid = y + BAR_H / 2 + 4.5
        weight = "600" if row["impulse"] else "400"
        fill = colors["impulse"] if row["impulse"] else colors["other"]
        width = BAR_W * row["value"] / 100
        out.append(f'<text x="{LABEL_X}" y="{mid:.1f}" font-size="13" '
                   f'font-weight="{weight}" text-anchor="end" '
                   f'fill="{colors["ink"]}">{escape(row["label"])}</text>')
        out.append(f'<path d="{_bar(BAR_X, y, width, BAR_H)}" fill="{fill}"/>')
        out.append(f'<text x="{BAR_X + width + 8:.1f}" y="{mid:.1f}" font-size="13" '
                   f'font-weight="{weight}" fill="{colors["ink"]}">'
                   f'{escape(row["text"])}</text>')
    out.append(f'<text x="16" y="{plot_bottom + 30}" font-size="12" '
               f'fill="{colors["muted"]}">{escape(chart["caption"])}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def build(root: Path) -> dict[str, str]:
    """Relative path -> SVG text for every chart file."""
    files: dict[str, str] = {}
    for chart in (high_chart(root), lite_chart(root)):
        for theme in THEMES:
            files[f"assets/{chart['name']}-{theme}.svg"] = render(chart, theme)
    return files


def stale(root: Path) -> list[str]:
    """Chart files that are missing or differ from what the ledgers produce."""
    out = []
    for relative, text in build(root).items():
        path = root / relative
        if not path.is_file() or path.read_text() != text:
            out.append(relative)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if "--check" in sys.argv[1:]:
        out_of_date = stale(root)
        for relative in out_of_date:
            print(f"out of date: {relative}")
        return 1 if out_of_date else 0
    for relative, text in build(root).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        print(f"wrote {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
