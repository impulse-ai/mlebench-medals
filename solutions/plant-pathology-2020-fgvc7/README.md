# plant-pathology-2020-fgvc7 — 🥇 gold — 0.98902 (mean column-wise ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/plant-pathology-2020-fgvc7
- **Best verified result:** 🥇 gold — **0.98902 (mean column-wise ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `image_finetune`, `image_finetune_ensemble` (Impulse proprietary operator families; source not published)

## Approach

Full end-to-end fine-tuning of a torchvision backbone — the escalation tier above frozen embeddings — trained on the leaf images with held-out validation. This medal required the one-hot label fix (PR #808) that landed after the 2026-07-07 snapshot.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.98364 |
| 2 | gold | 0.98902 |
| 3 | gold | 0.97976 |

**Best verified result:** gold, `0.98902`.
**Method:** public/current independently trained image route.
**Best-grade evidence SHA-256:** `593b5b013c670dc6620343d1fd5ef776bb0124b67d42da38655854abf755aca8`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
