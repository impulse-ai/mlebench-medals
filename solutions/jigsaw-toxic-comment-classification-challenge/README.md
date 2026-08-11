# jigsaw-toxic-comment-classification-challenge — 🥇 gold — 0.98750 (mean column-wise ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **Best verified result:** 🥇 gold — **0.98750 (mean column-wise ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/text_transformer.py`](../../engine/operators/text_transformer.py)

## Approach

A GPU transformer fine-tune operator: where word/char TF-IDF plateaus (~0.9805 mean AUC here), it fine-tunes pretrained transformer backbones (distilbert / roberta, sigmoid + BCE for multi-label) on a Vertex AI T4 through the Cap-0 seam and blends the transformers. Won autonomously on 2026-07-12, upgrading an earlier hand-scripted bronze (0.98663).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `result.json`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.98723 |
| 2 | gold | 0.98750 |
| 3 | silver | 0.98701 |

**Best verified result:** gold, `0.98750`.
**Method:** public training; provider revision unpinned.
**Best-grade evidence SHA-256:** `44afed4c88531ccb48a15295512e27b466d15866a8978d9573103168f4fe3a49`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

GPU lane: the job scripts and the independently re-gradeable `official_grade.json` for this task's GPU runs live in [`reproduce/gpu/jigsaw/`](../../reproduce/gpu/jigsaw/).

---
© 2026 Impulse AI. All rights reserved.
