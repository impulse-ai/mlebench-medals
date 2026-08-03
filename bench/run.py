"""MLE-bench harness CLI — INTEL-001 / 002 in PRD_MODEL_INTELLIGENCE.md.

Three modes:

    # Single experiment (with medal output)
    python -m bench.run --task spaceship-titanic \\
        --agent bench/experiments/tps_may22_trio_tabpfn.py

    # Light split (our curated fast-iteration set)
    python -m bench.run --split light \\
        --agent bench/experiments/some_agent.py

    # Full bench (all 75 tasks; multi-hour)
    python -m bench.run --split full \\
        --agent bench/experiments/some_agent.py

    # Remote (GCP)
    python -m bench.run --task X --agent ... --remote

Agent argument accepts either:
  - A path to a script (default; uses ScriptAgent — must accept
    --data-dir and --output).
  - "orchestrator:<url>" to invoke our production agent over HTTP
    (stub; blocked on INTEL-108).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .agents import (
    Agent,
    EESStandaloneAgent,
    ImpulseAgent,
    ImpulseTreeAgent,
    McpHttpImpulseAgent,
    ScriptAgent,
)
from .reporter import print_split_summary, print_task_result, write_split_json
from .runner import run_task
from .splits.lite22 import get_lite22_tasks


BENCH_ROOT = Path(__file__).resolve().parent


def _default_cache_dir() -> Path:
    """Ask mlebench where it stores prepared data; platform-aware."""
    try:
        from mlebench.registry import registry  # type: ignore
        return Path(registry.get_data_dir())
    except Exception:
        return Path.home() / ".cache" / "mle-bench" / "data"


DEFAULT_CACHE_DIR = _default_cache_dir()
DEFAULT_RUNS_DIR = BENCH_ROOT / "runs"
DEFAULT_TIMEOUT_S = 60 * 60 * 4  # 4h per task


def resolve_tasks(split: str | None, task: str | None) -> list[str]:
    if task:
        return [task]
    if split == "light":
        cfg = yaml.safe_load((BENCH_ROOT / "splits" / "light.yaml").read_text())
        return list(cfg["tasks"])
    if split == "lite":
        return get_lite22_tasks()
    if split == "full":
        # Defer to mlebench's authoritative competition list.
        try:
            out = subprocess.check_output(
                ["mlebench", "list-comps"], text=True,
            )
        except FileNotFoundError as e:
            raise SystemExit(
                "mlebench CLI not found. Install via `pip install mlebench` "
                "or clone https://github.com/openai/mle-bench and `pip install -e .`"
            ) from e
        tasks = [line.strip() for line in out.splitlines() if line.strip()]
        if not tasks:
            raise SystemExit(f"`mlebench list-comps` for split={split!r} returned no tasks.")
        return tasks
    raise SystemExit(f"unknown split: {split!r}")


def build_agent(
    agent_arg: str,
    base_url: str | None = None,
    *,
    mcp_url: str | None = None,
    mcp_api_key: str | None = None,
    mcp_bearer_token: str | None = None,
) -> Agent:
    """Resolve the --agent argument to an Agent instance.

    Formats accepted:
      - "impulse:current"    → ImpulseAgent(version="current", base_url=<flag>)
      - "impulse:new"        → ImpulseAgent(version="new", base_url=<flag>)
      - "impulse:tree"       → ImpulseTreeAgent (production tree-search path;
                                requires agent-api with USE_SDK_MODE unset/false)
      - "impulse:<version>"  → ImpulseAgent(version="<version>", base_url=<flag>)
      - "mcp:http"           → McpHttpImpulseAgent(url=<mcp-url>)
      - "ees:standalone"     → local benchmark-first EES core
      - "<path>.py"          → ScriptAgent(script_path=<path>)
    """
    if agent_arg == "ees:standalone":
        return EESStandaloneAgent()
    if agent_arg == "mcp:http":
        return McpHttpImpulseAgent(
            mcp_url=mcp_url
            or os.environ.get("IMPULSE_MCP_URL")
            or "https://api.dev.impulselabs.ai/api/mcp-http",
            api_key=mcp_api_key
            or os.environ.get("IMPULSE_MCP_API_KEY")
            or os.environ.get("IMPULSE_API_KEY"),
            bearer_token=mcp_bearer_token
            or os.environ.get("IMPULSE_MCP_BEARER_TOKEN")
            or os.environ.get("IMPULSE_BEARER_TOKEN"),
        )
    if agent_arg.startswith("impulse:"):
        version = agent_arg.split(":", 1)[1] or "current"
        if version == "tree":
            tree_kwargs = {}
            if base_url:
                tree_kwargs["agent_api_url"] = base_url
            return ImpulseTreeAgent(**tree_kwargs)
        # ImpulseAgent talks to agent-api directly (default :8005); the optional
        # --impulse-base-url flag overrides that endpoint.
        kwargs = {"version": version}
        if base_url:
            kwargs["agent_api_url"] = base_url
        return ImpulseAgent(**kwargs)
    path = Path(agent_arg).expanduser().resolve()
    if not path.exists():
        raise SystemExit(
            f"agent script not found: {path}\n"
            f"Tip: use 'impulse:current' or 'impulse:new' to drive Impulse via the gateway."
        )
    return ScriptAgent(script_path=path)


def main() -> int:
    p = argparse.ArgumentParser(prog="bench.run", description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", help="Single MLE-bench task id (per-experiment mode)")
    mode.add_argument("--split", choices=["light", "lite", "full"],
                     help="light=curated fast set, lite=official 22, full=all 75")
    p.add_argument("--agent", default="ees:standalone",
                   help="Path to a script, 'ees:standalone' (default), 'impulse:current' / 'impulse:new', or 'mcp:http'")
    p.add_argument("--impulse-base-url", default="http://localhost:3001",
                   help="Gateway URL when --agent=impulse:*  (default: local dev)")
    p.add_argument("--mcp-url", default=os.environ.get("IMPULSE_MCP_URL"),
                   help="HTTP MCP endpoint when --agent=mcp:http")
    p.add_argument("--mcp-api-key", default=os.environ.get("IMPULSE_MCP_API_KEY") or os.environ.get("IMPULSE_API_KEY"),
                   help="Impulse API key for --agent=mcp:http (or IMPULSE_MCP_API_KEY/IMPULSE_API_KEY)")
    p.add_argument("--mcp-bearer-token", default=os.environ.get("IMPULSE_MCP_BEARER_TOKEN") or os.environ.get("IMPULSE_BEARER_TOKEN"),
                   help="Bearer token for --agent=mcp:http (or IMPULSE_MCP_BEARER_TOKEN/IMPULSE_BEARER_TOKEN)")
    p.add_argument("--remote", action="store_true",
                   help="Submit run(s) to Vertex AI instead of executing locally")
    p.add_argument("--remote-profile", choices=["cheap-cpu", "cheap-gpu", "balanced-gpu"],
                   help="Vertex runtime profile for --remote (default: inferred per task)")
    p.add_argument("--remote-data-mode", choices=["cloud-prepare", "local-upload"],
                   default=os.environ.get("EES_REMOTE_DATA_MODE", "cloud-prepare"),
                   help="Data path for --remote: cloud-prepare downloads/prepares inside Vertex; local-upload uploads local prepared data")
    p.add_argument("--remote-kaggle-credentials", choices=["auto", "env", "local-json", "none"],
                   default=os.environ.get("EES_REMOTE_KAGGLE_CREDENTIALS", "auto"),
                   help="How cloud-prepare supplies Kaggle credentials to Vertex")
    p.add_argument("--remote-wait", action="store_true",
                   help="Wait for each submitted Vertex job to finish")
    p.add_argument("--remote-collect-dir", type=Path,
                   help="After --remote-wait, download outputs and write official remote_result.json ledgers here")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                   help="Per-task timeout in seconds (default: 14400 = 4h)")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                   help="Directory for per-run working dirs + result.json")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="mlebench prepared-data cache")
    p.add_argument("--output-json", type=Path,
                   help="Write aggregated results to this path (split mode)")
    args = p.parse_args()

    agent = build_agent(
        args.agent,
        base_url=args.impulse_base_url,
        mcp_url=args.mcp_url,
        mcp_api_key=args.mcp_api_key,
        mcp_bearer_token=args.mcp_bearer_token,
    )
    task_ids = resolve_tasks(args.split, args.task)

    if args.remote:
        from .cloud import collect_remote_result, handle_to_json, submit_remote_run

        if args.agent != "ees:standalone":
            raise SystemExit("--remote currently supports --agent ees:standalone only")
        if args.remote_collect_dir and not args.remote_wait:
            raise SystemExit("--remote-collect-dir requires --remote-wait")
        handles = []
        for task_id in task_ids:
            handle = submit_remote_run(
                task_id=task_id,
                agent_arg=args.agent,
                cache_dir=args.cache_dir,
                timeout_s=args.timeout,
                runtime_profile=args.remote_profile,
                wait=args.remote_wait,
                data_mode=args.remote_data_mode,
                kaggle_credentials_mode=args.remote_kaggle_credentials,
            )
            handles.append(handle)
            print(handle_to_json(handle), flush=True)
            if args.remote_collect_dir:
                collected = collect_remote_result(
                    task_id=task_id,
                    artifact_prefix=handle.artifact_prefix,
                    output_dir=args.remote_collect_dir / task_id,
                    cache_dir=args.cache_dir,
                )
                print(json.dumps({
                    "task_id": collected.task_id,
                    "status": collected.status,
                    "ledger_path": str(collected.ledger_path),
                    "submission_path": str(collected.submission_path) if collected.submission_path else None,
                    "grade": collected.grade,
                    "error": collected.error,
                }, indent=2, sort_keys=True))
        return 0

    print(f"Running {len(task_ids)} task(s) with agent {agent.name}")
    print(f"  Cache:   {args.cache_dir}")
    print(f"  Runs:    {args.runs_dir}")
    print(f"  Timeout: {args.timeout}s/task")
    print()

    runs = []
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        run = run_task(
            task_id=task_id, agent=agent,
            runs_dir=args.runs_dir, cache_dir=args.cache_dir,
            timeout_s=args.timeout,
        )
        runs.append(run)
        print_task_result(run)

    if len(runs) > 1:
        print_split_summary(runs)

    if args.output_json:
        write_split_json(runs, args.output_json)
        print(f"Wrote aggregated results to {args.output_json}")

    failures = [
        r for r in runs
        if not r.agent_result.success
        or (r.grade_result and not r.grade_result.success)
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
