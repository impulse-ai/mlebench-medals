# histopathologic-cancer-detection — 🥇 gold — 0.99585 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/histopathologic-cancer-detection
- **Best verified result:** 🥇 gold — **0.99585 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/image_finetune.py`](../../bench/ees_core/operators/image_finetune.py)

## Approach

End-to-end fine-tuning of a torchvision backbone on the histopathology patches, run by the agent itself as a Vertex AI T4 GPU job through the Cap-0 GPU seam (run `repin_histo2`, 2026-07-10). This is one of the 3 medals won via the GPU lane; re-running it needs GCP Vertex access.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.99585 |
| 2 | gold | 0.99578 |
| 3 | gold | 0.99580 |

**Best verified result:** gold, `0.99585`.
**Method:** public-trained image_finetune.
**Best-grade evidence SHA-256:** `33febfd66c506da34a2492347210d67227013968e79a95257984307a8b710d83`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

GPU lane: the job scripts and the independently re-gradeable `official_grade.json` for this task's GPU runs live in [`reproduce/gpu/histo/`](../../reproduce/gpu/histo/).

---
© 2026 Impulse AI. All rights reserved.
