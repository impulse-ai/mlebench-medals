# nomad2018-predict-transparent-conductors — 🥇 gold — 0.05373 (RMSLE)

- **Kaggle competition:** https://www.kaggle.com/c/nomad2018-predict-transparent-conductors
- **Medal:** 🥇 gold — official score **0.05373 (RMSLE)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/geometry_nomad.py`](../../bench/ees_core/operators/geometry_nomad.py)

## Approach

A geometry-aware operator: it parses each crystal's geometry file (lattice vectors, atomic coordinates, species) into descriptors — cell volume, atomic density, lattice norms, species counts, and pair-distance statistics — and fits a multi-output ExtraTrees regressor over k-fold splits to predict formation energy and bandgap energy jointly.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- The snapshot `grade.json` in this directory shows the gold medal (score 0.05373).

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
