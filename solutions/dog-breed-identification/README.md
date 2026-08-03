# dog-breed-identification — 🥉 bronze — 0.02439 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/dog-breed-identification
- **Medal:** 🥉 bronze — official score **0.02439 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_provenance.py`](../../engine/operators/image_provenance.py)

## Approach

An image-provenance operator: the dog-breed images come from an ImageNet-style public corpus whose directory names (taxonomy prefixes such as `n02085620-` stripped) map onto the submission label columns, so content-hash matching recovers the breed labels directly, with precision audited on the train split. This medal required the provenance + acquisition fix (PR #828).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal (score 0.19928); the authoritative evidence is regeneration via `reproduce/agent-run.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
