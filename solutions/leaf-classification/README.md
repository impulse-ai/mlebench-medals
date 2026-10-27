# leaf-classification — 🥈 silver — 0.00328 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/leaf-classification
- **Best verified result:** 🥈 silver — **0.00328 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Current route:** public training with an ImageNet-pretrained model.

## Approach

An image model trained on the competition's public training labels, starting from ImageNet pretraining.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.00671 |
| 2 | silver | 0.00328 |
| 3 | silver | 0.00470 |

**Best verified result:** silver, `0.00328`.
**Method:** public training and ImageNet-pretrained model.
**Best-grade evidence SHA-256:** `323e52608ae25413ac9482cc53ffba4cbd40e3f7236404710ccbd83e40a76302`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
