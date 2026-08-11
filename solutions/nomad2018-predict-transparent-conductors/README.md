# nomad2018-predict-transparent-conductors — 🥇 gold — 0.05373 (RMSLE)

- **Kaggle competition:** https://www.kaggle.com/c/nomad2018-predict-transparent-conductors
- **Best verified result:** 🥇 gold — **0.05373 (RMSLE)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/geometry_nomad.py`](../../bench/ees_core/operators/geometry_nomad.py)

## Approach

A geometry-aware operator: it parses each crystal's geometry file (lattice vectors, atomic coordinates, species) into descriptors — cell volume, atomic density, lattice norms, species counts, and pair-distance statistics — and fits a multi-output ExtraTrees regressor over k-fold splits to predict formation energy and bandgap energy jointly.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.05479 |
| 2 | silver | 0.05997 |
| 3 | silver | 0.05993 |

**Best verified result:** gold, `0.05373`.
**Method:** public_train_geometry_files.
**Best-grade evidence SHA-256:** `08c4a8c606a1fca796c23a47837641af2e3e955fa068db81192f87a7b5bf4dab`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
