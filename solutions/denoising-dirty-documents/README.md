# denoising-dirty-documents — 🥈 silver — 0.01919 (RMSE)

- **Kaggle competition:** https://www.kaggle.com/c/denoising-dirty-documents
- **Best verified result:** 🥈 silver — **0.01919 (RMSE)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/image_denoise.py`](../../bench/ees_core/operators/image_denoise.py)

## Approach

A CPU image-to-image denoising operator for per-pixel regression: it detects the one-row-per-pixel submission shape (`<imageId>_<row>_<col>`), estimates each page's background with a large-kernel box blur, and reconstructs the cleaned intensities from the paired dirty/clean training images — no torch, no GPU. (The snapshot `submission.csv` is 76 MB and is not committed.)

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.01926 |
| 2 | silver | 0.01928 |
| 3 | silver | 0.01919 |

**Best verified result:** silver, `0.01919`.
**Method:** public_paired_train_images.
**Best-grade evidence SHA-256:** `825dbd94f12f3d99f3d69e0089fe7f01a624bcf05b1456cb943ed44f1289a69b`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
