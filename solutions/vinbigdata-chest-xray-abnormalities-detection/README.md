# vinbigdata-chest-xray-abnormalities-detection — 🥇 gold — 0.4256

- **Kaggle competition:** https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
- **Result:** 🥇 gold — **0.4256**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `object_detection` (Impulse proprietary operator family; source not published)

## Approach

A two-stage detector: torchvision Faster R-CNN with a ResNet-50 FPN v2 backbone, initialised from COCO-pretrained weights and fine-tuned at 800 px for 12 epochs with horizontal flips, batch size 16 and mixed precision on the 13,500 training images.

Because the training table carries several radiologists per image, held-out images were scored against annotator-fused boxes; the model reached 0.49938 mAP at IoU 0.4 there. All predicted boxes per class are kept, and images without detections receive the competition's no-finding row. The run also trained a transformer detector and three Faster R-CNN variants (400 px and longer schedules); the 800 px, 12-epoch model had the best held-out score and is the submission.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.4256`.
**Medal lines (gold / silver / bronze):** 0.289 / 0.257 / 0.243; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 12.09 hours.
**Graded:** 2026-10-09.
**Submission SHA-256:** `7cdac83112a22fdb82445a1fc83948b7487da021b7a0469bcbfca1e6f8b7b3de`.
**Grade report SHA-256:** `8960c9743c3ab7e194672f4523420d6d583fae767a7af5ad8b073379be1f2d84`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
