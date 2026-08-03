# Tree-Search Baseline Protocol (MLE-bench)

Measures the **production Intelligence tree-search path** (Orchestrator → `SolutionTreeRunner`)
on MLE-bench via the `impulse:tree` agent. This is the reference number every future
search/operator/model change must beat. The baseline configuration is **current prod model
config** (Gemini 2.5 Pro ML-Engineer nodes per `agent-api/model_config.py`), fixed env, N=1 seed.

> **Why a dedicated launcher:** the older bench scripts (`start-mlebench.sh`, `start_agentic.sh`)
> export `USE_SDK_MODE=true`, which makes agent-api silently ignore `mode: "intelligence"` and
> run the flat SDK path — any result produced through them does **not** measure the tree.
> `bench/scripts/start_tree_stack.sh` exists precisely to launch the stack in orchestrator mode.

## Pre-flight checklist

1. **mlebench CLI** works: `mlebench --help`. Install: `pip install -r bench/requirements.txt`
   (mlebench comes from GitHub — see the comment in `bench/requirements.txt`).
2. **Kaggle credentials** at `~/.kaggle/kaggle.json` AND competition rules accepted for every
   task in the split — an unaccepted competition makes the data download 403/hang
   (see `bench/runner.py` for the failure mode).
3. **LLM keys** exported or in `.env` at repo root: `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`
   (the ML Engineer nodes run `gemini-2.5-pro`).
4. **Spend cap** chosen and exported — the launcher refuses to start without it:
   ```bash
   export BUDGET_MAX_LLM_COST_USD=<per-task cap in USD>
   ```
   Rough planning math: total sweep spend ≈ cap × task count (light ≈ 5–8 tasks,
   lite = 22 tasks). The cap is enforced per task because the budget tracker is
   per-session and each bench task is one session. `BUDGET_*` is read at agent-api
   **import time** → one budget config per process; **restart the stack between configs**.
5. **Quiet machine** — benchmark timings are only meaningful on an unloaded host.
   The launcher does one `docker stats` pass and **warns** (non-fatal) about
   containers in `restarting` state (Docker restart policies resurrect dev stacks
   after a Docker restart) and containers above ~20% CPU. Stop them
   (`docker stop <name>`) before a measured run; a starved sandbox makes
   executions time out and is indistinguishable from a bad agent in the results.
6. **Stack up**:
   ```bash
   bash bench/scripts/start_tree_stack.sh
   ```
   (boots the sandbox via docker compose, a **dedicated** postgres container
   `mle-ai-postgres-tree` on **:15432** via `docker run` (volume
   `postgres_tree_data`), and agent-api on :8005 in orchestrator mode).

   **Foreign-stack failure modes (the launcher now hard-fails on these):**
   - `❌ Port :8006 is served by a FOREIGN sandbox` — something other than this
     compose file's `mle-ai-sandbox` container answers :8006 (typically the dev
     stack's `mle-ai-sandbox-dev`, resurrected by its restart policy). The error
     names the squatter. Remediation: `docker stop <name>` and re-run; or point at
     it deliberately with `SANDBOX_URL=http://host:port`; or force adoption with
     `ALLOW_FOREIGN_SANDBOX=1` (warns — results may not be comparable).
   - **Postgres host-shadow gotcha** (the failure that motivated the dedicated
     container): a HOST-level postgres (e.g. a native macOS install) can bind
     `127.0.0.1:5432`/`[::1]:5432` and **shadow Docker's `*:5432` proxy** for any
     `localhost` connection. agent-api then reaches the host postgres —
     `role "mle_ai" does not exist` → **silent in-memory fallback for the entire
     run** — while the docker postgres is healthy and every `docker exec psql`
     probe PASSES (it's container-internal; both listeners coexist, so port-conflict
     checks see nothing either). The launcher defends in two ways: (a) the tree
     stack uses its **own** postgres `mle-ai-postgres-tree` on the uncommon port
     **:15432**, and (b) after readiness it probes auth over the **exact host path
     agent-api uses** (agent-api venv python + asyncpg → `localhost:15432`) and
     hard-fails with remediation if that connection doesn't work. Anything still
     listening on :5432 is ignored (one informational line).
7. **Calibration probe** (automatic, fatal): after the stack is healthy the
   launcher POSTs a representative fit (ExtraTrees, 300 trees, 8700x25 random
   data) to the sandbox and prints `sandbox calibration: X s`. A healthy machine
   finishes in a few seconds; **>30s aborts the launch** — the host is too
   starved to produce meaningful benchmark numbers (catches it in ~30s instead
   of after a wasted 90-minute run). Threshold override: `CALIBRATION_MAX_S`.
8. **THE one check that protects the whole measurement**:
   ```bash
   curl -s localhost:8005/v2/meta | jq '.use_sdk_mode'   # MUST print "false"
   ```
   If it prints `"true"`, you are about to benchmark the wrong orchestrator. Stop.
9. Disk: `~/.cache/mle-bench/data` needs room for the split's competition data.
10. **Execution-timeout ceilings** (automatic): the launcher exports
   `SANDBOX_MAX_EXEC_TIMEOUT_S` (sandbox-side pydantic cap on per-call
   `timeout_seconds`) and `EXEC_TIMEOUT_CEILING_S` (agent-api-side clamp +
   tool-schema/hint text), both defaulting to `$BUDGET_MAX_WALL_TIME_S`, so a
   heavy CPU dataset can train in **one long run** instead of being chopped at
   the prod 1800s cap. The per-call `timeout_seconds` choice still belongs to
   the agent — these only lift the upper bound. Anywhere the vars are unset
   (prod, dev compose) the defaults are byte-for-byte unchanged (1800/1800).
   Gotcha: the sandbox bakes its cap in at **container start** — an
   already-running sandbox keeps its old cap (the launcher warns; fix with
   `docker compose up -d --force-recreate sandbox`, never mid-sweep). Both
   knobs are recorded in `/v2/meta` → `tree_env` for reproducibility.

## Run 1 — light split (CPU-friendly curated subset)

```bash
python -m bench.run --split light --agent impulse:tree \
  --impulse-base-url http://localhost:8005 \
  --timeout 7200 \
  --output-json bench/runs/baseline_tree_light_$(date +%Y%m%d).json
```

## Run 2 — lite split (official MLE-bench Lite, 22 tasks; only after light passes)

```bash
python -m bench.run --split lite --agent impulse:tree \
  --impulse-base-url http://localhost:8005 \
  --timeout 14400 \
  --output-json bench/runs/baseline_tree_lite_$(date +%Y%m%d).json
```

## Reading and recording the result

- Medal rate + per-task scores: printed by the split summary and persisted in the
  `--output-json` file.
- **Cost**: sum `agent_meta.budget.llm_cost_used_usd` across tasks. Fetch happens right
  after each run reaches terminal state (the budget endpoint is in-memory — gone after an
  agent-api restart).
- **Reproducibility envelope**: every task's `agent_meta.stack_meta` records the model
  map, agent-api git SHA, `use_sdk_mode`, and the tree/budget env knobs;
  `agent_meta.bench_git_sha` records the harness revision. `agent_meta.tree` records
  nodes explored, best node/score, and the verification outcome
  (`winner_verified` / `metric_unverified` / `verification_failed` — from the Phase A gates).
- **Commit the baseline JSON** (or at minimum its summary) under `bench/runs/` so every
  future change diffs against it.

## Interpreting failures

- `agent_success=false` with `error` from the run record (`failure_reason` /
  `success_basis`) — the run genuinely failed; the tree shape in `agent_meta.tree`
  tells you how far it got.
- `success=true` with `agent_meta.run_status="failed"` — the run record failed but a
  gradable `submission.csv` was salvaged from the sandbox (a gradable submission is the
  bench's success criterion; the run-status discrepancy is preserved for analysis).
- Per-task timeout fired → the agent cancels the server-side session
  (`POST /v2/sessions/{id}/cancel`) before returning, so a timed-out task does not keep
  burning LLM spend in the background.
