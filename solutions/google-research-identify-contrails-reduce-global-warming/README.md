# google-research-identify-contrails-reduce-global-warming — 🥈 silver — 0.69712

- **Kaggle competition:** https://www.kaggle.com/c/google-research-identify-contrails-reduce-global-warming
- **Result:** 🥈 silver — **0.69712**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `band_record_segmentation` (Impulse proprietary operator family; source not published)

## Approach

The three infrared bands 11, 14 and 15 are combined into a false-colour (ash colour scheme) composite and upsampled to 512 × 512. A six-member U-Net ensemble is trained on it: EfficientNetV2-S, ConvNeXt-Small and ConvNeXt-Base trunks, each trained twice — once plainly and once with offset-corrected mirroring.

Training uses soft labels from annotator agreement, an equal mix of binary cross-entropy and Dice loss, and an exponential moving average of the weights. The six members are averaged and thresholded at 0.42, a value chosen on a 10% holdout of the training split, where the ensemble scored global Dice 0.70544 (best single member 0.69897).

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** silver, `0.69712`.
**Medal lines (gold / silver / bronze):** 0.71059 / 0.6936 / 0.67929; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 12.07 hours.
**Graded:** 2026-10-09.
**Submission SHA-256:** `c924b2ab483685d1d1b27b68a585f3f9b381b4483d1791dbd41f4aed149d18da`.
**Grade report SHA-256:** `eadde884d3e783ce4460f40d5e8fa191133f9aa8b40475c8c6b07a2eccbe4d23`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
