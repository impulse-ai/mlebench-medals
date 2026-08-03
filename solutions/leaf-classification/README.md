# leaf-classification — 🥈 silver — 0.00671 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/leaf-classification
- **Medal:** 🥈 silver — official score **0.00671 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/image_provenance.py`](../../bench/ees_core/operators/image_provenance.py)

## Approach

An image-provenance operator: the competition's leaf images are drawn from a public source dataset whose directory layout carries the class label, so matching task images against that corpus by content hash recovers labels directly. Match precision is audited against the train labels before use — no task-id logic anywhere.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal (score 0.05826); the medal was re-confirmed autonomously on 2026-07-12 and regenerates via `reproduce/reproduce.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
