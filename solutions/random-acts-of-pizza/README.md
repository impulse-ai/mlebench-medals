# random-acts-of-pizza — 🥉 bronze — ~0.692 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/random-acts-of-pizza
- **Medal:** 🥉 bronze — official score **~0.692 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** none dedicated — handled by the generic text/tabular machinery in [`engine/`](../../engine/).

## Approach

A text + metadata fusion approach: TF-IDF features over the request text are combined with the structured request metadata (account age, upvotes/downvotes, etc.) in the agent's generic text/tabular stack. There is no dedicated operator module for this task — the medal comes from the generic machinery.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `result.json`.
- The 2026-07-07 snapshot for this task contains no `grade.json` (only `result.json`); the authoritative evidence is regeneration via `reproduce/agent-run.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
