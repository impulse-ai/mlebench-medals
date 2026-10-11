# rsna-miccai-brain-tumor-radiogenomic-classification — 🥈 silver — 0.59

- **Kaggle competition:** https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification
- **Result:** 🥈 silver — **0.59**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `volume_cohort_classification` (Impulse proprietary operator family; source not published)

## Approach

The engine ran its volume-classification operator once and submitted its result. The run first built a compact view from the public file tree: the 348,641 slice files of the 526 training and 59 test patients were reduced to 2,340 arrays, one per patient and MRI sequence (FLAIR, T1w, T1wCE, T2w), holding the largest-area slices, the middle slice, sixteen evenly spaced slices over the central 10–90% of each series and, for the contrast sequence, twelve maximum-intensity slabs.

On that view it trained a family of six classifiers on an ImageNet-pretrained EfficientNet-B2 trunk, each over a different input: (A) the three largest-area slices of every sequence, (B) the middle slice of three sequences, (C) sixteen sampled slices of every sequence, (D) a montage of maximum-intensity slabs of the contrast sequence, (E) six sampled slices of all sequences with attention pooling over slices, and (F) a bright-region-centred crop of the FLAIR sequence. Members train without augmentation for 10 epochs at 224 px and predict with horizontal-flip averaging.

Validation is three repeats of five-fold cross-validation with one row per patient, stratified by the target, and five seeds per fold: 75 models per member. The cross-validation statistic of the family was AUC 0.5932 (95% patient-bootstrap interval 0.5455 to 0.6391, 2,000 resamples), and the AUC of the written out-of-fold predictions was 0.5974. All six members passed the inclusion rule. Each member's test prediction is the mean over its 75 models, rank-scaled; the submission is the equal-weight average of the six members.

This is the fourth submission graded on this 59-patient test set; the three earlier ones scored 0.51882, 0.56941 and 0.57941.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** silver, `0.59`.
**Medal lines (gold / silver / bronze):** 0.60096 / 0.5815 / 0.57449; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 2.94 hours.
**Graded:** 2026-10-11.
**Submission SHA-256:** `140e4068cd8e4d9c3a659122c8d58daa003a7e08563caa53db392e5a070d39ed`.
**Grade report SHA-256:** `5edef3b64d36dcfe85693cb282b76997d2841df46fce44d782283b0e9cd7d94d`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
