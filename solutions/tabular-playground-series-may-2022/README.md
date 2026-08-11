# tabular-playground-series-may-2022 — 🥈 silver — 0.99822 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/tabular-playground-series-may-2022
- **Best verified result:** 🥈 silver — **0.99822 (ROC AUC)**, graded with OpenAI's unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Result record:** [`results/lite22-three-run.json`](../../results/lite22-three-run.json)

## Approach

The model splits the tabular input across two disjoint neural branches. One branch learns the main feature representation; the other models only the feature pairs allowed by a constrained interaction graph. Keeping those paths separate gives the interaction signal room to matter without opening a dense set of arbitrary crosses.

Candidate members produce out-of-fold predictions on the public training data. Outer selection uses those OOF predictions only, and each confirmation run selects its own outer blend independently. The three scored records are the selected outer blends themselves; their members don't count as confirmation runs.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.99821 |
| 2 | bronze | 0.99818 |
| 3 | silver | 0.99822 |

**Best verified result:** silver, `0.99822`.
**Method:** public-trained independent outer blends; members excluded.
**Best-grade evidence SHA-256:** `5480166e87d2f894dd6be2c36884db4e7e33f37bd8a1f0e5bf269f169716d2e4`.

## Verify it yourself

The [machine-readable results ledger](../../results/lite22-three-run.json) carries all three confirmation records and hashes. See the [evidence index](../../reproduce/EVIDENCE.md) for the 19-task inventory and the [full runbook](../../reproduce/VERIFY.md) for the grading flow.

---
© 2026 Impulse AI. All rights reserved.
