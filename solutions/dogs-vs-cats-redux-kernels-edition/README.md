# dogs-vs-cats-redux-kernels-edition — 🥇 gold — 0.00597 (log loss)

- **Kaggle competition:** https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
- **Best verified result:** 🥇 gold — **0.00597 (log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_embedding.py`](../../engine/operators/image_embedding.py)

## Approach

A pretrained image-embedding operator: images are embedded with a frozen pretrained backbone, a logistic head is fit on top, and backbone/calibration choices are made using out-of-fold predictions only. The cats-vs-dogs prior in the pretrained features is near-perfect, giving a 0.0098 log loss.

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
**Method:** public-training image fine-tuning with three independent GPU seeds; the derived blend isn't part of the confirmation set.
**Best-grade evidence SHA-256:** `6005accddf7b756344e89b0eed9333f37ac59bcc43db047493e26b2437cd2ca9`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
