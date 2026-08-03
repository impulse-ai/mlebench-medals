# Prompt-Tuning Action Plan

**Status:** implementations landed (uncommitted on `bench-impulse-agent`), awaiting measurement runs by the operator
**Owner:** @juanpmcianci
**Scope:** improve MLE-bench medal rate by tightening agent-api specialist prompts. Production-faithful changes only — no task-specific hints baked into the bench's outer prompt.
**Branch:** `bench-impulse-agent`

---

## 1. Goal

Get spaceship-titanic from ~0.81 (current) to **≥ 0.84** (top-2 territory) through prompt changes alone, while not regressing any other task. Cumulative lift across all four planned changes: predicted **+0.05 to +0.13** based on the audit in `bench/PROMPT_AUDIT.md` (forthcoming) and the per-task variance baseline established in Step 0.

This is also the validation loop for INTEL-108-adjacent prompt work in `PRD_MODEL_INTELLIGENCE.md` — if these prompt-only changes don't move the needle, the role-split agent is the next escalation.

---

## 2. Method

**Per-change attribution.** Each change ships as one commit, is measured on the fixed task set, and survives a gate or is reverted. We never bundle prompt changes — bundling makes failed-gate analysis impossible.

**Measurement task set** (from `bench/splits/light.yaml`):

| Task | Modality | Metric | Why |
|---|---|---|---|
| `spaceship-titanic` | tabular | accuracy | The target — top-2 ~0.85, we're at ~0.81 |
| `nomad2018-predict-transparent-conductors` | tabular | mean-column-wise-rmsle | Multi-target regression, lower=better |
| `tabular-playground-series-may-2022` | tabular | auc-roc | Large tabular (900K rows), AUC metric |
| `detecting-insults-in-social-commentary` | NLP | auc-roc | NLP smoke test |
| `aerial-cactus-identification` | vision | auc-roc | Vision smoke test |

**Variance discipline.** N=3 runs per task per measurement. If a task's σ > 0.02 across N=3, bump that task to N=5 going forward.

**Per-step gate.** A change is kept if:
- Mean ↑ on ≥ 3 of 5 tasks (≥ 2 for low-leverage changes — see per-step table)
- No single task regresses by more than 0.01 (0.02 for high-variance Step 4)

If gate fails: `git revert <commit>`, log why, move to next step.

---

## 3. Step 0 — Baseline (~1 h, ~$1)

- [ ] `git commit` everything currently on `bench-impulse-agent` (clean starting state)
- [ ] Confirm `bench/splits/light.yaml` lists the 5 measurement tasks
- [ ] Build a tiny measurement script `bench/scripts/measure.py` (~50 lines) that:
    - Reads `bench/runs/*/result.json` (or hits bench-service `/runs` API)
    - Filters to a target run-tag (e.g. `--tag baseline`)
    - Prints per-task mean ± std and aggregate medal count
- [ ] Run **N=3 × 5 tasks = 15 baseline runs**, tagged `baseline`
- [ ] Record results in `bench/results/baseline.json` (committed alongside the run-record artifacts)
- [ ] Confirm σ on each task. If any > 0.02, document and bump N for that task

This is the "before" number for every subsequent step. It is **not** subject to a gate — it's the reference.

---

## 4. Step 1 — Metric handoff threading (~1.5 h, ~$1)

**The gap:** specialists default to "accuracy" / "AUC" even when the competition grades on something else. The outer message names the metric but it's not reliably propagated.

**Files touched:**

- [ ] `agent-api/prompts/orchestrator_sdk.md` — orchestrator MUST include the metric name in every `delegate_*` task description
- [ ] `agent-api/prompts/methodology_advisor.md` — add `primary_metric` field to JSON output, populated from the user message
- [ ] `agent-api/prompts/ml_engineer_sdk.md` — explicit "use this metric for CV scoring and final reporting; do NOT default to accuracy/AUC if a different metric is named"
- [ ] `agent-api/prompts/insight_writer_sdk.md` — reports must name the optimized metric, not a default

**Predicted lift:** +0.01 to +0.03 across multiple tasks (especially tabular-playground-may-2022 — auc-roc — and nomad2018 — rmsle — where Gemini's defaults are most off).

**Gate:** mean ↑ on ≥ 3 of 5 tasks; no task regresses by > 0.01.

**Steps:**

- [ ] Edit the 4 prompt files (see Appendix A for exact text proposals)
- [ ] `git commit -m "agent: thread competition metric through specialists"`
- [ ] Restart `start-mlebench.sh` (agent-api re-reads prompts at startup)
- [ ] Run 15 measurement runs, tagged `step1-metric`
- [ ] Run `bench/scripts/measure.py --before baseline --after step1-metric`
- [ ] Check gate; commit + proceed, or revert + document

---

## 5. Step 2 — Structural EDA in Data Profiler (~2 h, ~$1)

**The gap:** profiler reports shape/dtypes/missing-% only. Misses:
- Regex-able compositional columns (`PassengerId=4408_01` → group_id + member; `Cabin=B/5/P` → Deck/Num/Side)
- Group structure in ID-like columns (multi-member groups → GroupKFold candidate)
- Target × "obvious column" cross-tab anomalies (e.g. Cryosleep × all-zero spending)

**Files touched:**

- [ ] `agent-api/prompts/data_profiler_sdk.md` — expand the Procedure section with a "Compositional & structural analysis" step. See Appendix B for proposed text.

**Predicted lift:** +0.02 to +0.04, concentrated on tabular tasks with structured columns. This is the single biggest predicted win.

**Gate:** mean ↑ on ≥ 3 of 5 tasks; no task regresses by > 0.01.

**Steps:**

- [ ] Edit `data_profiler_sdk.md`
- [ ] `git commit -m "agent: structural EDA pass in data profiler"`
- [ ] Restart launcher
- [ ] Run 15 measurement runs, tagged `step2-eda`
- [ ] Measure against `step1-metric`
- [ ] Check gate

**Risk note:** if this step underdelivers (< +0.01 lift), the audit's prediction was wrong and we should pause to investigate before continuing to Step 3 — the structural-EDA hypothesis is the load-bearing assumption of this whole plan.

---

## 6. Step 3 — Validation strategy auto-pick (~2 h, ~$1)

**The gap:** ML Engineer defaults to random KFold regardless of data structure. Should be: StratifiedKFold for class imbalance > 1.5:1, GroupKFold when group structure detected (from Step 2's output), TimeSeriesSplit when datetime column present.

**Files touched:**

- [ ] `agent-api/prompts/data_profiler_sdk.md` — report `class_balance_ratio`, `detected_group_column`, `time_column` fields in the data profile output (small extension of Step 2 work)
- [ ] `agent-api/prompts/ml_engineer_sdk.md` — conditional CV strategy block keyed on those profile fields

**Predicted lift:** +0.005 to +0.015. Lower than 1&2 but reduces CV-LB gap (more reliable scores) and avoids overfitting on bad splits.

**Gate:** mean ↑ on ≥ 2 of 5 tasks; no task regresses by > 0.01. (Lower bar than Steps 1&2 because the lift prediction is smaller.)

**Steps:**

- [ ] Edit `data_profiler_sdk.md` (profile output extension)
- [ ] Edit `ml_engineer_sdk.md` (conditional CV block)
- [ ] `git commit -m "agent: auto-select CV strategy from data profile"`
- [ ] Restart launcher
- [ ] Run 15 measurement runs, tagged `step3-cv`
- [ ] Measure against `step2-eda`
- [ ] Check gate

---

## 7. Step 4 — Web search MCP tool (~5 h, ~$1-2)

**The gap:** agent has no way to learn task-specific tricks beyond Gemini's training data. Real Kaggle competitors web-search top writeups for a competition.

**This step is real code, not just prompts.** Implementation:

- [ ] New MCP tool `agent-api/tools/web_search.py` — calls a search API (Brave / Bing / DuckDuckGo) and returns top-K results with snippet text
- [ ] Register in `agent-api/mcp_tools.py`
- [ ] Update `agent-api/prompts/ml_engineer_sdk.md` to mention the tool exists and when to use it (e.g., "when stuck on a task you haven't seen, search for `<task_name> kaggle top solutions`")
- [ ] Document the new env var for the search API key in `start-mlebench.sh`
- [ ] Add a small allowlist of fetched-page sizes so the agent can't blow context

**Predicted lift:** +0.02 to +0.05 on tasks where the agent's training data is thin (especially newer competitions).

**Gate:** mean ↑ on ≥ 3 of 5 tasks; no task regresses by > 0.02. (Higher regression tolerance because web search introduces real variance — search results can mislead.)

**Steps:**

- [ ] Implement + register the tool
- [ ] Add agent-side prompt guidance for when to use it
- [ ] `git commit -m "agent: web_search MCP tool + prompt"`
- [ ] Restart launcher
- [ ] Run 15 measurement runs, tagged `step4-search`
- [ ] Measure against `step3-cv`
- [ ] Check gate

**Decision point before starting Step 4:** if Steps 1-3 already land spaceship-titanic ≥ 0.84, consider deferring Step 4 in favor of starting **INTEL-109 (Kaggle solutions corpus + retrieval index)** from `PRD_MODEL_INTELLIGENCE.md`. That ticket is the controlled-corpus version of this same idea and probably the right long-term answer.

---

## 8. Bundled "secondary" cleanup (ships alongside Step 2)

Too small to measure individually; bundle into the Step 2 commit:

- [ ] Submission filename enforcement (`ml_engineer_sdk.md` Phase 4) — "save exactly `submission.csv`, not `final_submission.csv`"
- [ ] HPO depth: "≥ 50 Optuna trials when budget=thorough"
- [ ] Ensembling specificity: name "rank-average" or "Caruana hill-climbing" explicitly
- [ ] Insight Writer reports the actual optimized metric, not a default

These don't get their own gate. If Step 2's overall gate passes, the bundle ships. If it fails, the bundle is reverted with it and we'd ship them after a fix.

---

## 9. Total budget

| Phase | Time | Cost (Gemini API) |
|---|---|---|
| Step 0 baseline | 1 h | ~$1 |
| Step 1 metric handoff | 1.5 h | ~$1 |
| Step 2 structural EDA | 2 h | ~$1 |
| Step 3 CV strategy auto-pick | 2 h | ~$1 |
| Step 4 web search | 5 h | ~$1-2 |
| Final aggregate report | 1 h | $0 |
| **Total** | **~12-13 h** | **~$5-10** |

---

## 10. Per-step report format

Each step produces a section in `bench/results/STEP<N>.md`:

```
# Step N: <change name>

Baseline: <prior tag>  →  This step: step<N>-<short>

| Task                                | Before       | After         | Δ       | Gate |
|-------------------------------------|--------------|---------------|---------|------|
| spaceship-titanic (acc, higher=↑)   | 0.815 ± 0.012| 0.823 ± 0.009 | +0.008  | PASS |
| nomad2018 (rmsle, lower=↑)          | 0.142 ± 0.005| 0.139 ± 0.004 | −0.003  | PASS |
| tps-may-2022 (auc, higher=↑)        | 0.982 ± 0.003| 0.985 ± 0.002 | +0.003  | PASS |
| detecting-insults (auc, higher=↑)   | 0.873 ± 0.011| 0.871 ± 0.014 | −0.002  | PASS |
| aerial-cactus (auc, higher=↑)       | 0.987 ± 0.005| 0.988 ± 0.003 | +0.001  | PASS |
| **Aggregate**                       |              |               |         | PASS — keep |

Notes:
- <anything surprising>
- <anything that informs the next step>
```

---

## 11. Open decisions

| Question | Default | Status |
|---|---|---|
| N per task at baseline | 3 | revisit after Step 0 if σ > 0.02 anywhere |
| Should Step 4 (web search) ship before INTEL-109 (Kaggle retrieval)? | Yes — ship Step 4 if Steps 1-3 underdeliver on spaceship-titanic; otherwise defer to INTEL-109 | re-evaluate after Step 3 |
| If a step's gate fails by a hair (mean ↑ on exactly 2 of 5), do we keep or revert? | Revert; the gate exists for a reason | confirm with @juanpmcianci on first failure |
| Do we keep the secondary bundle if Step 2 fails? | No — they ride with Step 2's commit and get reverted together | confirm |

---

## 12. Appendix A — proposed text for Step 1 prompt edits

*(To be filled in when Step 1 is implemented. Sketch:)*

**`orchestrator_sdk.md`** — add to delegation guidance:
> When invoking `delegate_data_profiler` or `delegate_ml_engineer`, the `task` field must include the competition's evaluation metric name verbatim (e.g. "the metric is `accuracy`" or "the metric is `mean-column-wise-rmsle`"). Specialists rely on this to override defaults — do not paraphrase.

**`ml_engineer_sdk.md`** — add to Phase 2/3:
> The task description names the **exact evaluation metric**. Use that metric — and only that metric — for cross-validation scoring, model selection, and the final reported number. Do not default to accuracy/AUC/RMSE if a different name is given.

**`insight_writer_sdk.md`** — add to Guidelines:
> All performance claims MUST reference the metric the ML Engineer was given (e.g. "Model achieves AUC-ROC of 0.82 on the held-out set"), not a default metric. If unsure, ask the orchestrator before fabricating a number.

---

## 13. Appendix B — proposed text for Step 2 prompt edit

*(To be filled in when Step 2 is implemented. Sketch:)*

Add a new step to `data_profiler_sdk.md` Procedure (between current "Turn 2" and "Turn 3"):

> **Compositional & structural analysis** (always run; report findings as part of the data profile):
> 1. For each string column, check whether values match a regular pattern (e.g. `\d+_\d+`, `[A-Z]/\d+/[A-Z]`, `\w+\s\w+`). If ≥ 95% of non-null values match, flag the column as **splittable** and propose the sub-columns.
> 2. For each ID-like column (high uniqueness, mostly unique but with some repeats), check whether values share a prefix grouping. If ≥ 10% of rows are in multi-member groups, flag `detected_group_column` and recommend GroupKFold for the ML Engineer.
> 3. For boolean / low-cardinality categorical columns, cross-tab against any numeric column where one side is mostly zero. Flag rows that violate the dominant pattern (e.g. boolean=True but numeric≠0) as "anomaly candidates" — they often carry strong signal for the target.
> 4. Compute `class_balance_ratio` for classification targets. If max(class) / min(class) > 1.5, recommend StratifiedKFold.
> 5. Detect datetime-like columns (parseable dates, or string with date format). If found, flag `time_column` and recommend TimeSeriesSplit if temporal causality matters.
>
> Report all flagged findings in the structured output so the ML Engineer can use them directly.

---

## 14. Status tracker

| Phase | Implementation | Measurement |
|---|---|---|
| Step 0 — baseline locked | ✓ tag field plumbed end-to-end; `bench/stats.py` shared module; `launch-batch.py` + `measure.py` CLI; **bench-service `/batches` `/tags` `/compare` endpoints**; **gateway proxies**; **`/mlebench` Measurements UI** (single-launch tag input, Batch launch card, Tag summaries table, Compare two tags with gate verdict) | ⏳ awaiting operator: launch baseline batch in the studio |
| Step 1 — metric handoff threading | ✓ (4 prompts edited) | ⏳ awaiting operator |
| Step 2 — structural EDA + bundled secondary cleanup | ✓ (`data_profiler_sdk.md` + `ml_engineer_sdk.md`) | ⏳ awaiting operator |
| Step 3 — validation strategy auto-pick | ✓ (`data_profiler_sdk.md` + `ml_engineer_sdk.md`) | ⏳ awaiting operator |
| Step 4 — web search MCP tool *(conditional)* | not started, deferred per §7 decision point | n/a |
| Final aggregate report | n/a | ⏳ after Steps 1-3 measured (Measurements → Compare baseline → step3-cv) |

---

## 15. Execution runbook — frontend-driven

The operator workflow is the **Measurements UI on `/mlebench`**. The CLI
scripts (`launch-batch.py`, `measure.py`) are kept as a headless fallback
(see §15.5) but the primary path is the studio page.

### 15.1 Step 0 — commit infra, restart, capture baseline

Developer-side: commit the harness changes (these don't affect the agent
itself, so they all go in one infra commit). From a terminal:

```bash
git status
git add bench-service/app/models.py bench-service/app/routes.py \
        bench-service/app/store.py \
        frontend/lib/mlebench-types.ts frontend/lib/mlebench-api.ts \
        frontend/app/mlebench/page.tsx \
        gateway/src/routes/bench.ts \
        bench/stats.py bench/scripts/measure.py bench/scripts/launch-batch.py \
        bench/PROMPT_TUNING_PLAN.md
# Sanity check: no prompt edits in this commit
git diff --cached --name-only | grep '^agent-api/prompts/'  # should print NOTHING
git commit -m "bench: tags + batches + measurements UI on /mlebench"
./start-mlebench.sh
```

Then in the **studio at `/mlebench`** → "Batch launch" card:

1. Tag: `baseline`
2. Reps per task (N): `3`
3. Impulse version: `current`
4. Budget: `thorough`
5. Tasks: click "select all" to pick the 5 measurement tasks listed in §2.
6. Click **Launch batch →**.

The studio queues 15 runs. They appear in "Recent runs" with the
`baseline` tag chip. After ~30 min (parallelism = bench-service worker
concurrency), all should be terminal.

In the **Measurements** card → "Tag summaries" — confirm `baseline` shows
`15 / 5 tasks / 15 succeeded`. If a task's per-run σ > 0.02 (check via the
CLI fallback in §15.5), re-launch a 3-run top-up batch with the same tag.

### 15.2 Step 1 — metric handoff threading

Developer-side: commit ONLY Step 1's prompt edits. Per §16, the
`ml_engineer_sdk.md` file is touched by all three steps — use `git add -p`
to stage only Step 1's hunks.

```bash
git add -p agent-api/prompts/orchestrator_sdk.md \
           agent-api/prompts/council/methodology_advisor.md \
           agent-api/prompts/ml_engineer_sdk.md \
           agent-api/prompts/insight_writer_sdk.md
git diff --cached --stat
git commit -m "agent: thread the competition metric through specialists (step 1)"
./start-mlebench.sh
```

Then in the studio:

1. **Batch launch**: tag=`step1-metric`, N=3, same 5 tasks, launch.
2. Wait ~30 min for all to terminate.
3. **Measurements → Compare two tags**: Before=`baseline`, After=`step1-metric`. Click **Compare →**.
4. Read the gate verdict:
   - **PASS** (green badge) → keep the commit, move to Step 2.
   - **FAIL** (red badge) → `git revert HEAD` (or `git reset --hard HEAD~1`), investigate the per-task table to see which task regressed, decide whether to tweak Step 1 or skip it.

### 15.3 Step 2 — structural EDA + secondary bundle

```bash
git add -p agent-api/prompts/data_profiler_sdk.md \
           agent-api/prompts/ml_engineer_sdk.md
git diff --cached --stat
git commit -m "agent: structural EDA in data profiler + submission/HPO/ensemble bundle (step 2)"
./start-mlebench.sh
```

Studio: Batch launch tag=`step2-eda`, N=3. Then Compare `step1-metric` →
`step2-eda`. **This is the highest-leverage step.** If the gate fails,
don't continue to Step 3 before understanding why — structural EDA is the
load-bearing hypothesis.

### 15.4 Step 3 — validation strategy auto-pick

```bash
git add -p agent-api/prompts/data_profiler_sdk.md \
           agent-api/prompts/ml_engineer_sdk.md
git diff --cached --stat
git commit -m "agent: validation strategy auto-pick from data profile (step 3)"
./start-mlebench.sh
```

Studio: Batch launch tag=`step3-cv`, N=3. Then Compare `step2-eda` →
`step3-cv`. Per the plan §6, this step uses the looser `--min-improving 2`
gate — to apply it in the UI, click Compare, and if the verdict is FAIL
with `n_improving = 2`, that's an acceptable PASS for Step 3 specifically.
*(TODO: surface a gate-threshold knob in the UI; for now use the CLI
fallback below if the strict default disagrees with the Step-3-specific
rule.)*

### 15.5 Headless alternative (CI / cron)

Everything above is also reachable from the CLI. The scripts call the
same bench-service endpoints under the hood, so behavior is identical.

```bash
# Step 0
python bench/scripts/launch-batch.py --tag baseline --n 3
python bench/scripts/measure.py --tag baseline

# Step 1
python bench/scripts/launch-batch.py --tag step1-metric --n 3
python bench/scripts/measure.py --before baseline --after step1-metric

# Step 2
python bench/scripts/launch-batch.py --tag step2-eda --n 3
python bench/scripts/measure.py --before step1-metric --after step2-eda

# Step 3 (looser gate)
python bench/scripts/launch-batch.py --tag step3-cv --n 3
python bench/scripts/measure.py --before step2-eda --after step3-cv \
    --min-improving-tasks 2
```

Exit code 0 = gate PASS, 1 = gate FAIL.

### 15.6 Aggregate report

After Step 3 settles, in the studio's Measurements: Compare `baseline` →
`step3-cv`. The headline number is whether spaceship-titanic moved
above 0.84.

- If yes → **defer Step 4** in favor of INTEL-109 (Kaggle solutions
  corpus + retrieval index) from `PRD_MODEL_INTELLIGENCE.md`.
- If no → **proceed to Step 4** (web search MCP tool).

---

## 16. File overlap between steps (commit hygiene)

`agent-api/prompts/ml_engineer_sdk.md` is touched by all three Steps 1-3.
To preserve per-step attribution, use `git add -p` to stage only the
relevant hunks per step:

| File | Step 1 hunks | Step 2 hunks | Step 3 hunks |
|---|---|---|---|
| `orchestrator_sdk.md` | "Passing context to specialists" section | — | — |
| `council/methodology_advisor.md` | `primary_metric` field in JSON | — | — |
| `data_profiler_sdk.md` | — | Procedure step 3 (structural analysis) + Output Format new fields (SPLITTABLE, DETECTED_GROUP_COLUMN, TIME_COLUMN, ANOMALY_CANDIDATES, CLASS_BALANCE_RATIO) | RECOMMENDED_CV_STRATEGY + RECOMMENDED_CV_REASON output lines |
| `ml_engineer_sdk.md` | "Optimization metric" section + Phase 1 metric wording | Phase 2-4 rewrites (FE application, HPO trial counts, ensembling, Phase 4 filename) | "Cross-validation strategy" section before Iteration Protocol |
| `insight_writer_sdk.md` | Guidelines metric-grounding bullets | — | — |

If `git add -p` gets fiddly, an alternative is to land a single mega-commit
("steps 1+2+3 prompt edits") and gate-test cumulatively rather than
per-step. This loses per-step attribution but is faster operationally.
Note that on the plan, **the documented choice is per-step attribution**.
