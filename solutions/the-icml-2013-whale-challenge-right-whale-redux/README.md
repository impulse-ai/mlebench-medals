# the-icml-2013-whale-challenge-right-whale-redux — 🥉 bronze — 0.91633 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux
- **Medal:** 🥉 bronze — official score **0.91633 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/audio_embedding.py`](../../bench/ees_core/operators/audio_embedding.py)

## Approach

A pretrained audio-embedding operator: each whale-call clip (WAV/AIFF) is converted to a log-spectrogram image with per-frequency median denoising, then fed through the shared image-embedding core (frozen pretrained backbone + logistic head). Pure CPU.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- The snapshot `grade.json` in this directory shows the bronze medal (score 0.91633).

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
