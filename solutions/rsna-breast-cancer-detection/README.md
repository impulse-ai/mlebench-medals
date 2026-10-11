# rsna-breast-cancer-detection — 🥇 gold — 0.50526

- **Kaggle competition:** https://www.kaggle.com/c/rsna-breast-cancer-detection
- **Result:** 🥇 gold — **0.50526**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `dicom_screening_classification` (Impulse proprietary operator family; source not published)

## Approach

Each DICOM file is converted with its own windowing tags (VOI LUT or window), inverted when stored as MONOCHROME1, cropped to the breast (the largest bright connected component), flipped to a common orientation and resized to a 2048 × 1024 view. No table column and no label is read in this step.

An ImageNet-pretrained EfficientNetV2-S is trained at 1536 × 768 with four patient-grouped folds stratified by site, auxiliary heads on the extra training columns (biopsy, invasive, BIRADS, density, age, difficult negative), an exponential moving average of the weights, and snapshots from the last three epochs.

A breast-level second stage then combines the image scores of all views of the same breast (mean, maximum, minimum, image count, CC and MLO view indicators, site and age). The final 0/1 flags are set by matching the flagged fraction per site, chosen on out-of-fold predictions. Both additions were adopted because they improved the fold-honest score. Out-of-fold probabilistic F1 was 0.42656; the final submission blends three variants of this pipeline built on the same fold models.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.50526`.
**Medal lines (gold / silver / bronze):** 0.49 / 0.43 / 0.41; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 23.11 hours.
**Graded:** 2026-10-10.
**Submission SHA-256:** `e70629ebfcad4238bebd5c0920dc607b9ca9a2d69fb8dd09cbbe4196fa6fb8ee`.
**Grade report SHA-256:** `26f064dc3c07175d2adcc4cd11dbe009571a95fdf517581fb6a44f4cfdaa6f8a`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
