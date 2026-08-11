# text-normalization-challenge-russian-language — 🥉 bronze — 0.97915 (token-level accuracy)

- **Kaggle competition:** https://www.kaggle.com/c/text-normalization-challenge-russian-language
- **Best verified result:** 🥉 bronze — **0.97915 (token-level accuracy)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/seq2seq_lookup.py`](../../bench/ees_core/operators/seq2seq_lookup.py)

## Approach

The same general seq2seq lookup operator as the English task — exact-match / most-frequent token normalization learned from training pairs, detected purely from data structure with no task-id or language-specific special-casing.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.97906 |
| 2 | bronze | 0.97906 |
| 3 | bronze | 0.97906 |

**Best verified result:** bronze, `0.97915`.
**Method:** public-training deterministic lookup; independent process confirmation.
**Best-grade evidence SHA-256:** `aea2c0e1f8d0cf52b4c49e8c8dde795ba4ea7693ab43923df0ffae1a8c7d376f`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
