# 3d-object-detection-for-autonomous-vehicles — 🥇 gold — 0.18128

- **Kaggle competition:** https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles
- **Result:** 🥇 gold — **0.18128**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `lidar_bev_detection` (Impulse proprietary operator family; source not published)

## Approach

A LiDAR-only detector. Each sweep from the primary LiDAR sensor is encoded as pillars on a 512 × 512 bird's-eye-view grid (0.2 m cells, ±51.2 m around the sensor, −5 m to 3 m in height). A centre-heatmap head and a box-regression head predict the nine object classes, and boxes are mapped back to world coordinates for the submission.

The submitted model trained for 10 epochs (36,220 steps) with mixed precision and augmentation on 14,489 samples. 3,654 samples from held-out scenes were kept for validation, where it scored 0.19289 on the competition's mean-average-precision scorer (IoU 0.50–0.95). The run trained four detectors and submitted the one with the best held-out score.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.18128`.
**Medal lines (gold / silver / bronze):** 0.139 / 0.053 / 0.046; higher is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 12.26 hours.
**Graded:** 2026-10-09.
**Submission SHA-256:** `cb0a7265b96b06bfffa392275b0b7a154c330367926b8a5bab9ba065586510a7`.
**Grade report SHA-256:** `043fec519a5d1087f4e5e00f6215d8304622f2dafd3572eb8693a25c2f43ccfd`.

## Verify it yourself

The checked-in [results ledger](../../results/hard15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [HARD evidence index](../../reproduce/EVIDENCE_HARD.md) lists every HARD result.

---
© 2026 Impulse AI. All rights reserved.
