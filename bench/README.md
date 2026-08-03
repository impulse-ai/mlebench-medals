# MLE-bench harness

Per-task and per-split runner for [openai/mle-bench](https://github.com/openai/mle-bench).
Implements INTEL-001 (harness), INTEL-002 (splits), partial INTEL-003 (per-run JSON).

## Install

```bash
pip install -r bench/requirements.txt
# mlebench CLI must be on PATH; verify:
mlebench --help
```

`mlebench` requires the official Kaggle CLI to be authenticated (it downloads
competition data through the Kaggle API on first run for each task).

## Three modes

### Single experiment (per-experiment, with medal)

```bash
python -m bench.run \
  --task spaceship-titanic \
  --agent bench/experiments/tps_may22_trio_tabpfn.py
```

Output:

```
Running 1 task(s) with agent script:tps_may22_trio_tabpfn
  Cache:   /home/you/.cache/mle-bench/data
  Runs:    bench/runs
  Timeout: 14400s/task

[1/1] spaceship-titanic
  preparing task data for spaceship-titanic (one-time)…
  [S] spaceship-titanic                                   score=  0.81543  medal=silver  (847s)
```

Per-run JSON is persisted to `bench/runs/<run_id>/result.json` with score,
medal, gold/silver/bronze thresholds, agent stdout/stderr, timings.

### Light split (curated fast iteration)

```bash
python -m bench.run --split light --agent <path>
```

Our curated subset (`bench/splits/light.yaml`) — ~5-8 tasks across modalities,
each <30 min on CPU. Distinct from MLE-bench's official `lite` split.

### Lite split (MLE-bench official, 22 tasks)

```bash
python -m bench.run --split lite --agent <path>
```

Defers to `mlebench list-comps --split lite`.

### Full split (all 75 tasks)

```bash
python -m bench.run --split full --agent <path> \
  --output-json bench/runs/full_run.json
```

Multi-hour. Gates Phase 1/2/3 transitions per PRD.

### Remote (GCP) — stub

```bash
python -m bench.run --task X --agent ... --remote
```

Currently raises `NotImplementedError`. See `bench/cloud.py` for the
intended interface; depends on IMP-1568 (GCloud GPU sandbox) being done.

## Agent interface

Two adapters (`bench/agents.py`):

| Adapter | Status | Use when |
|---|---|---|
| `ScriptAgent` | Real | The agent is a Python script. Convention: must accept `--data-dir <task-data>` and `--output <submission.csv>`. |
| `ImpulseAgent` | Real | Drive Impulse end-to-end via the gateway. Tests intent-routing, training-mcp, and prediction in one pass. |

### Running the bench against Impulse

```bash
# Local dev: gateway on localhost:3001, no Auth0 (uses x-guest-id header)
python -m bench.run \
  --task spaceship-titanic \
  --agent impulse:current \
  --impulse-base-url http://localhost:3001
```

The flow `ImpulseAgent.run()` exercises:

1. detect schema from `sample_submission.csv` + `description.md`
2. `POST /api/datasets/upload` (train.csv) → poll until profiled
3. `POST /api/chat` with a `"Train …"` prompt → keyword-routes to `dataset_training` tool
4. `GET /api/jobs/{job_id}` poll until succeeded (default 30 min)
5. `POST /api/datasets/upload` (test.csv) — required: predictions on training data are rejected
6. `POST /api/chat` with a `"Generate predictions …"` prompt
7. download predictions
8. reshape to match `sample_submission.csv` schema → `submission.csv`
9. grade via `mlebench grade-sample`

`--agent impulse:new` uses the same client but points at the post-replacement
Designer/Coder/Tuner/Critic agent (INTEL-108) once it ships. Same Agent
interface either way.

### Running with the script harness

To add a new script-based agent, drop a file under `bench/experiments/`
that follows the `--data-dir / --output` convention. Examples:

- `bench/experiments/tps_may22_trio_tabpfn.py`

## Output layout

```
bench/
└── runs/
    └── <task>_<unix-ts>_<rand>/
        ├── submission.csv      # what the agent produced
        ├── result.json         # score + medal + thresholds + timings
        └── (agent's own scratch files)
```

`result.json` schema:

```json
{
  "run_id": "spaceship-titanic_1715900000_abc123",
  "task_id": "spaceship-titanic",
  "agent_name": "script:tps_may22_trio_tabpfn",
  "agent_success": true,
  "wall_time_s": 847.3,
  "submission_path": "bench/runs/.../submission.csv",
  "agent_error": null,
  "score": 0.81543,
  "metric": "accuracy",
  "medal": "silver",
  "gold_threshold": 0.82,
  "silver_threshold": 0.80,
  "bronze_threshold": 0.78,
  "grade_error": null,
  "work_dir": "bench/runs/...",
  "created_at": 1715900847.3
}
```

## What's not in v1

- Cloud Run Jobs orchestration (see `bench/cloud.py` stub)
- Parallel multi-task execution (sequential for v1; trivial to add a worker pool)
- Live streaming dashboard (rich-table mode planned)
- Cost dashboard (INTEL-CR2)
- bench_runs SQL table (INTEL-003 — the JSON files are the precursor)
- Skip-list integration (INTEL-CR3)
