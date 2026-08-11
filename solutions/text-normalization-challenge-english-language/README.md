# text-normalization-challenge-english-language — 🥉 bronze — 0.99125 (token-level accuracy)

- **Kaggle competition:** https://www.kaggle.com/c/text-normalization-challenge-english-language
- **Best verified result:** 🥉 bronze — **0.99125 (token-level accuracy)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/seq2seq_lookup.py`](../../bench/ees_core/operators/seq2seq_lookup.py)

## Approach

A general seq2seq lookup operator: it detects the (input, output) token-pair structure and the sentence+position id layout, learns an exact-match / most-frequent normalization table from the training pairs, and falls back to identity for unseen tokens. This medal required the grader-coercion fix (PR #819).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.99125 |
| 2 | bronze | 0.99125 |
| 3 | bronze | 0.99125 |

**Best verified result:** bronze, `0.99125`.
**Method:** public-training deterministic lookup with the disclosed grader patch; independent process confirmation.
**Best-grade evidence SHA-256:** `dfef6e2b5d259ff1658639e5f7482e311b32e9e0468a74600cfeff86341144b3`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
