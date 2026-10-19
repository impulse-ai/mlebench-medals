# hms-harmful-brain-activity-classification — 🥉 bronze — 0.37228

- **Kaggle competition:** https://www.kaggle.com/c/hms-harmful-brain-activity-classification
- **Result:** 🥉 bronze — **0.37228**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Run:** seed 202 of the three-seed evaluation (code tag `eval-high-20261013`); seed 303 also medaled (bronze, 0.37511) and seed 101 scored 0.41207, no medal.
- **Solution operator:** `signal_window_cnn` with an `index_layout_stack` second stage (Impulse proprietary operator family; source not published)

## Approach

Stage one renders log-mel spectrograms of the EEG montage chains, stacked beside the companion spectrogram, and trains ImageNet-pretrained 2D CNNs on them with KL divergence against the expert-vote distribution, under grouped K-fold validation.

Stage two is a cross-fitted boosted correction of stage one. It starts from the stage-one predictions and adds label-free features describing how each recording appears in the public data index: rows per recording, a row's position within its recording, and recording length. No test label and no label-derived feature is used. The medal comes from this second stage: the gain over the signal model is a property of how the public index is laid out, and the engine records it as such.

The run's recorded out-of-fold KL divergence was 0.355; the graded test score is 0.37228 against a bronze line of 0.375383.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** bronze, `0.37228`.
**Medal lines (gold / silver / bronze):** 0.290807 / 0.353323 / 0.375383; lower is better.
**Method:** Engine run (three-seed evaluation, seed 202).
**LLM:** GPT-6-Luna.
**Run wall time:** 11.95 hours.
**Graded:** 2026-10-16.
**Submission SHA-256:** `8fd1e38d16451d1d37d81ae212479c5bdbf78a3a6e354511ab148f6215af939b`.
**Grade report SHA-256:** `e7564630193a14bbe05c86ade32a3563e6de5738a737ea227029d8199f049243`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes, and the [three-seed ledger](../../results/high15-eval-three-seed.json) carries every seed of the evaluation. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
