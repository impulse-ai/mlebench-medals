# rsna-miccai-brain-tumor-radiogenomic-classification — 🥉 bronze — 0.57941

- **Kaggle competition:** https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification
- **Result:** 🥉 bronze — **0.57941**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Method:** Impulse ML pipeline: a six-member classifier family with a pre-registered selection rule

## Approach

A family of six classifiers built on EfficientNet-B2. Each member sees a different construction of the MRI input — slice montages, a single middle slice, evenly spaced slices through the volume, max-intensity slabs of the contrast sequence, an attention-pooled stack across the four sequences, and a FLAIR crop centred on the bright region. All six are trained without augmentation (AdamW, 10 fixed epochs) and predict with horizontal-flip test-time augmentation.

Selection used cross-validation on the 526 training patients only: three repeats of stratified five-fold patient-level CV with five seeds per fold, and paired patient bootstrap intervals (2,000 resamples). The adoption rule and a 0.59 out-of-fold AUC bar were written down before the models were trained. The family reached out-of-fold AUC 0.596, a gain of 0.021 (95% interval 0.009 to 0.033) over the augmented family.

Each member's test prediction is the mean over its 75 cross-validation models; the submission is the equal-weight average of the six members' rank-scaled predictions. This is the third submission graded on this 59-patient test set; the two earlier ones scored 0.51882 and 0.56941.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** bronze, `0.57941`.
**Medal lines (gold / silver / bronze):** 0.60096 / 0.5815 / 0.57449; higher is better.
**Method:** Impulse ML pipeline.
**LLM:** not applicable.
**Run wall time:** not applicable.
**Graded:** 2026-10-07.
**Submission SHA-256:** `bed83e4dfc402dd919c1332d591dd07d426c88c221498ca4a5f73942651c8751`.
**Grade report SHA-256:** `7f79bbe88fcb4734aa3e633df7f57ef350a0cd5af373104f5a228854c889be7f`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
