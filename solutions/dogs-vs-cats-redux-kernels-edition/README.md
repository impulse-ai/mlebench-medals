# dogs-vs-cats-redux-kernels-edition — 🥇 gold — 0.00597 (log loss)

- **Kaggle competition:** https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
- **Best verified result:** 🥇 gold — **0.00597 (log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Current solution operator:** `image_finetune` (Impulse proprietary operator family; source not published)

## Approach

Current confirmation route: public-training image fine-tuning with three independent GPU seeds. Each run trains the image model on the competition's public training set; the scored confirmations are the three seed runs, while the derived blend stays outside the confirmation set.

The older snapshot used a frozen pretrained embedding backbone with a logistic head and OOF-only backbone/calibration selection. That historical route remains useful reproduction context for the local files below, but it isn't the route represented by the current three-run table.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.00920 |
| 2 | gold | 0.00870 |
| 3 | gold | 0.00974 |

**Best verified result:** gold, `0.00597`.
**Method:** public-training image fine-tuning; three independent GPU seeds; derived blend excluded.
**Best-grade evidence SHA-256:** `6005accddf7b756344e89b0eed9333f37ac59bcc43db047493e26b2437cd2ca9`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
