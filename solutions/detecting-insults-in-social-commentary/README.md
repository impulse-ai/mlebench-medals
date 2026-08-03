# detecting-insults-in-social-commentary — 🥇 gold — 0.91084 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/detecting-insults-in-social-commentary
- **Medal:** 🥇 gold — official score **0.91084 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/text_tfidf.py`](../../bench/ees_core/operators/text_tfidf.py)

## Approach

A word+character TF-IDF operator: a `FeatureUnion` of word- and char-n-gram TF-IDF features feeding linear and gradient-boosted classifiers, with k-fold out-of-fold predictions used for candidate selection and calibration. Pure CPU, no task-specific configuration.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `result.json`.
- The 2026-07-07 snapshot for this task contains no `grade.json` (only `result.json`); the authoritative evidence is regeneration via `reproduce/reproduce.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
