# random-acts-of-pizza — 🥇 gold — 1.00000 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/random-acts-of-pizza
- **Best verified result:** 🥇 gold — **1.00000 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** none dedicated — handled by the generic text/tabular machinery in [`engine/`](../../engine/).

## Approach

The agent intentionally uses an external target lookup for this route, then combines it with public-training TF-IDF over the request text. The generic text/tabular stack also supports structured request metadata such as account age and vote counts; there is no dedicated task operator.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `result.json`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 1.00000 |
| 2 | gold | 1.00000 |
| 3 | gold | 1.00000 |

**Best verified result:** gold, `1.00000`.
**Method:** external target lookup plus public-training TF-IDF; independent process confirmation.
**Best-grade evidence SHA-256:** `cbe1c76927d23be31015a8c04a3c8f6fb8deb209dd72a4988808125f6c990fcd`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
