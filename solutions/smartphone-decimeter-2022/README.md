# smartphone-decimeter-2022 — 🥇 gold — 0.000

- **Kaggle competition:** https://www.kaggle.com/c/smartphone-decimeter-2022
- **Result:** 🥇 gold — **0.000**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `artifact_label_leakage` (Impulse proprietary operator family; source not published)

## Approach

Exploit discovery. The run audited the public files and found that the supplemental NMEA track logs shipped with each trip reproduce the ground-truth positions. It decoded the logs, matched them to the label timestamps, and confirmed exact agreement on all 151 training trips (maximum error under 0.00000001 m) before using the same reconstruction for all 37,087 test rows.

No positioning model is trained. This result is listed as exploit discovery in the results ledger and on the High page.

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.000`.
**Medal lines (gold / silver / bronze):** 1.768 / 2.119 / 3.068; lower is better.
**Method:** Engine run (exploit discovery).
**LLM:** GPT-6-Luna.
**Run wall time:** 0.11 hours.
**Graded:** 2026-10-08.
**Submission SHA-256:** `ddf3d46c9d08913013fc94095e6567c4eb515d5620e5510c1c8d4c088b7134f7`.
**Grade report SHA-256:** `c91955f4dbd308e2013a6179774644108d91404f7b3e43af020a8a57f72a2686`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
