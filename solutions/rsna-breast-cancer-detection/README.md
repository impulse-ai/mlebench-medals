# rsna-breast-cancer-detection — 🥇 gold — 0.54737

- **Kaggle competition:** https://www.kaggle.com/c/rsna-breast-cancer-detection
- **Result:** 🥇 gold — **0.54737**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Run:** seed 303 of the three-seed evaluation (code tag `eval-high-20261013`); the engine medaled in all three seeds (silver 0.47312, silver 0.47312, gold 0.54737).
- **Solution operator:** `dicom_screening_classification` (Impulse proprietary operator family; source not published)

## Approach

Each DICOM file is decoded once with its own windowing tags, cropped to the breast, normalised to a common side and stored as an image view before training. No table column and no label is read in this step.

ImageNet-pretrained EfficientNetV2 members are trained at high resolution on patient-grouped folds. Each breast's score is the mean over its images, and the final 0/1 flags are set by matching a flagged fraction chosen on out-of-fold predictions. Two optional additions, a flagged fraction per acquisition site and a breast-level second stage over both views, are adopted only when they improve the fold-honest score.

The run's recorded out-of-fold score was 0.395; the graded test score is 0.54737 against a gold line of 0.49.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.54737`.
**Medal lines (gold / silver / bronze):** 0.49 / 0.43 / 0.41; higher is better.
**Method:** Engine run (three-seed evaluation, seed 303).
**LLM:** GPT-6-Luna.
**Run wall time:** 23.08 hours.
**Graded:** 2026-10-16.
**Submission SHA-256:** `cb33ac46b23a5f8bd0e5c88088db721faa01254693723b61e733ac3db2a6b641`.
**Grade report SHA-256:** `d0cddca5de575346d70e522fc99028db5d97f7d4b671ce982bd4af0e0af4deb3`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes, and the [three-seed ledger](../../results/high15-eval-three-seed.json) carries every seed of the evaluation. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
