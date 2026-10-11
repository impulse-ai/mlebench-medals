# hms-harmful-brain-activity-classification — 🥉 bronze — 0.36573

- **Kaggle competition:** https://www.kaggle.com/c/hms-harmful-brain-activity-classification
- **Result:** 🥉 bronze — **0.36573**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Method:** Impulse ML pipeline: a signal-window CNN ensemble followed by a cross-fitted second stage

## Approach

Stage one is an Impulse `signal_window_cnn` model: EfficientNet-B2 and ConvNeXt-Base networks trained on image renderings of each 50-second EEG window together with the accompanying 10-minute spectrogram, with soft labels from the expert votes and five folds grouped by spectrogram.

Stage two is a CPU-only XGBoost model that takes the stage-one log-probabilities as its starting point and adds label-free features describing how each recording appears in the public index file: rows per EEG, spectrogram and patient, recording lengths, the row's position within its recording, and label spacing. It is trained on soft labels and averaged over five seeds. No test label and no label-derived feature is used.

The recipe, its code hashes and the expected score range were written down before the submission file was produced, and the submission was graded once; two earlier submissions for this competition had been graded before it. Cross-fitted KL divergence on the training rows improved from 0.489 (stage one) to 0.363 with the second stage; the gain comes from the structure of the public index, not from a stronger signal model.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** bronze, `0.36573`.
**Medal lines (gold / silver / bronze):** 0.290807 / 0.353323 / 0.375383; lower is better.
**Method:** Impulse ML pipeline.
**LLM:** not applicable.
**Run wall time:** not applicable.
**Graded:** 2026-10-07.
**Submission SHA-256:** `3c318cd03b1f51b31c7b218746875fb5d1a4d473a5d8812fe0ae6d019ceb433d`.
**Grade report SHA-256:** `4f0f4c7fd57d175678dc93cc984cba75c4d6d27edaaf1c462cf6a25d7c767091`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
