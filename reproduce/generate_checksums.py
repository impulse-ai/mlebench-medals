#!/usr/bin/env python3
"""Generate reproduce/data_checksums.txt from a local MLE-bench prepared cache.

A verifier runs `mlebench prepare -c <task>` (per-task, after accepting each
competition's Kaggle rules) and then this script to confirm they prepared the
*same* public+private data our medals were graded against. The grading anchor is
the private answer file (`prepared/private/test.csv` for most Lite-22 tasks);
if that sha256 matches, the official grader is scoring against identical labels.

Usage:
    python reproduce/generate_checksums.py             # scan default cache
    python reproduce/generate_checksums.py --data-dir /path/to/mle-bench/data
    python reproduce/generate_checksums.py --verify     # compare against the
                                                        # committed manifest
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# The official Lite-22 task list (mirrors splits/lite22.py).
LITE22 = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]

# Relative paths (under <cache>/<task>/) whose checksums pin the graded data.
# The private answer file is the load-bearing one for grading.
TARGET_RELPATHS = [
    "prepared/public/sample_submission.csv",
    "prepared/private/test.csv",
]

MANIFEST = Path(__file__).resolve().parent / "data_checksums.txt"


def _default_cache() -> Path:
    try:
        from mlebench.registry import registry  # type: ignore

        return Path(registry.get_data_dir())
    except Exception:
        # macOS default; Linux uses ~/.cache/mle-bench/data
        mac = Path.home() / "Library" / "Caches" / "mle-bench" / "data"
        return mac if mac.exists() else Path.home() / ".cache" / "mle-bench" / "data"


def _sha256_and_rows(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    rows = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            rows += chunk.count(b"\n")
    return h.hexdigest(), rows


def scan(data_dir: Path) -> list[str]:
    lines: list[str] = []
    for task in LITE22:
        for rel in TARGET_RELPATHS:
            p = data_dir / task / rel
            if p.is_file():
                digest, rows = _sha256_and_rows(p)
                lines.append(f"{digest}  {rows:>9d}  {task}/{rel}")
            else:
                lines.append(f"{'-'*64}  {'NA':>9}  {task}/{rel}  (NOT PREPARED)")
    return lines


def render(data_dir: Path, lines: list[str]) -> str:
    header = [
        "# MLE-bench Lite-22 prepared-data checksum manifest",
        "#",
        "# Columns: sha256  rows(newlines)  task/relative_path",
        "# The private answer file (prepared/private/test.csv) is the grading anchor:",
        "# if its sha256 matches, mlebench is scoring your submission against the",
        "# identical held-out labels our medals were graded against.",
        "#",
        "# Regenerate/verify:  python reproduce/generate_checksums.py [--verify]",
        "# '(NOT PREPARED)' = task absent from this cache; run `mlebench prepare -c <task>`.",
        "#",
    ]
    return "\n".join(header + lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="compare a fresh scan against the committed manifest")
    args = ap.parse_args(argv)

    data_dir = args.data_dir or _default_cache()
    if not data_dir.exists():
        print(f"cache dir not found: {data_dir}", file=sys.stderr)
        return 2

    lines = scan(data_dir)
    text = render(data_dir, lines)

    if args.verify:
        if not MANIFEST.exists():
            print(f"no committed manifest at {MANIFEST}", file=sys.stderr)
            return 2
        want = {ln.split("  ")[0] + ln.split(maxsplit=3)[-1]
                for ln in MANIFEST.read_text().splitlines()
                if ln and not ln.startswith("#") and "NOT PREPARED" not in ln}
        got = {ln.split("  ")[0] + ln.split(maxsplit=3)[-1]
               for ln in lines if "NOT PREPARED" not in ln}
        missing = want - got
        if missing:
            print("MISMATCH — these committed checksums did not reproduce locally:")
            for m in sorted(missing):
                print(f"  {m}")
            return 1
        print(f"OK — all {len(want)} committed data checksums reproduced from {data_dir}")
        return 0

    print(text)
    MANIFEST.write_text(text)
    print(f"# wrote {MANIFEST}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
