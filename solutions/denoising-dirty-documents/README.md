# denoising-dirty-documents — 🥈 silver — 0.01919 (RMSE)

- **Kaggle competition:** https://www.kaggle.com/c/denoising-dirty-documents
- **Medal:** 🥈 silver — official score **0.01919 (RMSE)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/image_denoise.py`](../../bench/ees_core/operators/image_denoise.py)

## Approach

A CPU image-to-image denoising operator for per-pixel regression: it detects the one-row-per-pixel submission shape (`<imageId>_<row>_<col>`), estimates each page's background with a large-kernel box blur, and reconstructs the cleaned intensities from the paired dirty/clean training images — no torch, no GPU. (The snapshot `submission.csv` is 76 MB and is not committed.)

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal (score 0.28616); the authoritative evidence is regeneration via `reproduce/reproduce.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
