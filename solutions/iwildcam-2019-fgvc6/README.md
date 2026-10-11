# iwildcam-2019-fgvc6 — 🥇 gold — 0.48326

- **Kaggle competition:** https://www.kaggle.com/c/iwildcam-2019-fgvc6
- **Result:** 🥇 gold — **0.48326**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `image_finetune` (Impulse proprietary operator family; source not published)

## Approach

A single pretrained ConvNeXt-Base, fully fine-tuned at 384 px for the 14 classes in the training set. Training used class balancing for the long-tailed label distribution, mixup and CutMix, label smoothing 0.1, stochastic depth 0.2 and horizontal flips; inference used test-time augmentation.

The model trained on 24,000 images with 6,000 held out, and scored macro-F1 0.85848 on the held-out images. The run planned a five-backbone ensemble and put its training time into the ConvNeXt-Base member, which is the submitted model.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.48326`.
**Medal lines (gold / silver / bronze):** 0.212 / 0.129 / 0.114; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 3.02 hours.
**Graded:** 2026-10-09.
**Submission SHA-256:** `506d4a4bdb2d15f1993b5dd1a4c832b28040da6ec005d34891579bf70fd34d33`.
**Grade report SHA-256:** `65de63c3e0e0be46bf175b63d0c90ca15dd28e25ed7b431c99fbb6b783096e32`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
