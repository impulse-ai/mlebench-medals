# mlsp-2013-birds — 🥈 silver — 0.9128 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/mlsp-2013-birds
- **Medal:** 🥈 silver — official score **0.9128 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/audio_multilabel.py`](../../bench/ees_core/operators/audio_multilabel.py)

## Approach

A general audio multi-label operator: it detects the long-format submission contract (one row per (recording, class) pair with composite integer ids) and the variable-length multi-label annotation file, then fits a multi-label audio classifier on CPU. Re-confirmed autonomously on 2026-07-12; an earlier hand-scripted GPU job (silver, 0.93143) is preserved in `reproduce/gpu/birds/`.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal (score 0.5); the authoritative evidence is regeneration via `reproduce/reproduce.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

GPU lane: the job scripts and the independently re-gradeable `official_grade.json` for this task's GPU runs live in [`reproduce/gpu/birds/`](../../reproduce/gpu/birds/).

---
© 2026 Impulse AI. All rights reserved.
