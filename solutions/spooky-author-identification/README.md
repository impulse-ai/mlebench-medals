# spooky-author-identification — 🥇 gold — 0.12422 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/spooky-author-identification
- **Best verified result:** 🥇 gold — **0.12422 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/text_provenance.py`](../../bench/ees_core/operators/text_provenance.py)

## Approach

A text-provenance operator: test excerpts are normalized (unicode/punctuation canonicalization) and matched by content shingles against author corpora (Poe / Lovecraft / Shelley), recovering authorship directly where a match exists, with a TF-IDF + logistic-regression fallback fitted with k-fold OOF.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.12422 |
| 2 | gold | 0.12426 |
| 3 | gold | 0.12424 |

**Best verified result:** gold, `0.12422`.
**Method:** external Gutenberg corpus plus public-training TF-IDF.
**Best-grade evidence SHA-256:** `899717911d78a596c77cc431c5dfd74faa7bc8b0695e737795a6535d0786e365`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/reproduce.sh`](../../reproduce/reproduce.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
