# mlsp-2013-birds — 🥈 silver — 0.93170 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/mlsp-2013-birds
- **Best verified result:** 🥈 silver — **0.93170 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/audio_multilabel.py`](../../engine/operators/audio_multilabel.py)

## Approach

A general audio multi-label operator: it detects the long-format submission contract (one row per (recording, class) pair with composite integer ids) and the variable-length multi-label annotation file, then fits a multi-label audio classifier on CPU. Re-confirmed autonomously on 2026-07-12; an earlier hand-scripted GPU job (silver, 0.93143) is preserved in `reproduce/gpu/birds/`.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.93170 |
| 2 | silver | 0.93170 |
| 3 | silver | 0.93170 |

**Best verified result:** silver, `0.93170`.
**Method:** public deterministic legacy replay; independent process confirmation.
**Best-grade evidence SHA-256:** `a45a528c1c73a16cc8d8d45d4b9ef4b9c37ac6924484d4cab210cee19f90ba7f`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

GPU lane: the job scripts and the independently re-gradeable `official_grade.json` for this task's GPU runs live in [`reproduce/gpu/birds/`](../../reproduce/gpu/birds/).

---
© 2026 Impulse AI. All rights reserved.
