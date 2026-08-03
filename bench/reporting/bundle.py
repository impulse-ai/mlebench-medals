"""Write a self-contained reproducibility bundle for a benchmark run.

Every run — classical operators or the Gemini code-gen loop — ends with a
`bundle/` folder holding the official submission, the exact code that
produced it, the ordered steps of the run, the environment, and a README
with reproduction instructions. Bundles are output-only: they are written
AFTER grading and never feed back into selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BundleInputs:
    task_id: str
    submission_path: Path
    grade: dict | None
    code_paths: list[Path]
    steps: list[dict]
    command: str
    environment: dict
    notes: str = ""


def write_run_bundle(bundle_dir: Path, inputs: BundleInputs) -> Path:
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "submission").mkdir(parents=True)
    (bundle_dir / "code").mkdir()

    shutil.copy(inputs.submission_path, bundle_dir / "submission" / "submission.csv")

    # Track written filenames to detect and disambiguate basename collisions
    written_names = set()
    for code_path in inputs.code_paths:
        if Path(code_path).is_file():
            original_name = Path(code_path).name
            target_name = original_name

            # If basename already written, add numeric suffix before extension
            if target_name in written_names:
                stem = original_name.rsplit(".", 1)[0] if "." in original_name else original_name
                ext = original_name[len(stem):] if "." in original_name else ""
                counter = 2
                while f"{stem}_{counter}{ext}" in written_names:
                    counter += 1
                target_name = f"{stem}_{counter}{ext}"

            shutil.copy(code_path, bundle_dir / "code" / target_name)
            written_names.add(target_name)
    if inputs.grade is not None:
        (bundle_dir / "grade.json").write_text(json.dumps(inputs.grade, indent=2, sort_keys=True) + "\n")
    (bundle_dir / "steps.jsonl").write_text(
        "".join(json.dumps(step, default=str) + "\n" for step in inputs.steps)
    )
    (bundle_dir / "environment.json").write_text(json.dumps(inputs.environment, indent=2, sort_keys=True, default=str) + "\n")
    (bundle_dir / "command.txt").write_text(inputs.command.rstrip() + "\n")
    (bundle_dir / "README.md").write_text(_render_readme(inputs))
    _write_manifest(bundle_dir, inputs.task_id)
    return bundle_dir


def _reproduce_line(inputs: BundleInputs) -> str:
    """One copy-pasteable line: captured env overrides prefixed onto the command.

    command.txt stays the faithful raw command; the README line is the
    self-contained version (env vars the run actually had, e.g. the corpus
    dir a provenance run needs, are invisible in argv alone).
    """
    env = inputs.environment.get("env") or {}
    prefix = " ".join(f"{key}={value}" for key, value in sorted(env.items())
                      if key not in inputs.command)
    return f"{prefix} {inputs.command}".strip()


def _render_readme(inputs: BundleInputs) -> str:
    if inputs.grade is not None:
        score = inputs.grade.get("score")
        medal = inputs.grade.get("medal", "none")
        result_line = f"**Official score:** {score}   **Medal:** {medal}"
    else:
        result_line = "**Official score:** ungraded"
    sections = [
        f"# Run bundle: {inputs.task_id}",
        "",
        result_line,
        "",
        "## Contents",
        "",
        "- `submission/submission.csv` — the official graded submission",
        "- `code/` — the exact code that produced it",
        "- `steps.jsonl` — ordered events of the run (probes, candidates, scores, selection)",
        "- `environment.json` — interpreter, package versions, git commit, EES_* env",
        "- `command.txt` — the exact command used",
        "- `manifest.json` — sha256 of every file in this bundle",
        "",
        "## Reproduce",
        "",
        "From the repository root, with the task's prepared data in the mle-bench",
        "cache (see `environment.json` for versions and env overrides):",
        "",
        "```bash",
        _reproduce_line(inputs),
        "```",
    ]
    if inputs.notes:
        sections += ["", "## Notes", "", inputs.notes]
    return "\n".join(sections) + "\n"


def _write_manifest(bundle_dir: Path, task_id: str) -> None:
    files = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(path.relative_to(bundle_dir)),
                      "sha256": digest, "bytes": path.stat().st_size})
    manifest = {"task_id": task_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "files": files}
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


_SNAPSHOT_PACKAGES = ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
                      "scipy", "torch", "torchvision", "mlebench", "google-genai")


def capture_environment(env_prefixes: tuple[str, ...] = ("EES_",)) -> dict:
    from importlib import metadata

    packages: dict[str, str] = {}
    for name in _SNAPSHOT_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        git_commit = None
    env = {key: value for key, value in os.environ.items()
           if any(key.startswith(prefix) for prefix in env_prefixes)}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit,
        "packages": packages,
        "env": env,
    }
