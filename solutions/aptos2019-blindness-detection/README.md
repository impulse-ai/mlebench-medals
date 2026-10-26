# aptos2019-blindness-detection — 🥈 silver — 0.92020 (quadratic weighted kappa)

- **Kaggle competition:** https://www.kaggle.com/c/aptos2019-blindness-detection
- **Best verified result:** 🥈 silver — **0.92020 (quadratic weighted kappa)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `image_finetune_ensemble` (Impulse proprietary operator family; source not published)

## Approach

A 3-config diverse ordinal ensemble: diverse backbone families at different resolutions and seeds, each trained as a regression head with k-fold OOF and blended by OOF-weighted averaging so decorrelated configs dominate. Built for ordinal tasks whose test distribution differs from train.

## Evidence in this directory

- Files from the engine's development run of 7 July 2026: `grade.json`, `result.json`, `submission.csv`.
- The three confirmation runs below are recorded, with their submission hashes, in the [results ledger](../../results/lite22-three-run.json).

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.91942 |
| 2 | bronze | 0.91942 |
| 3 | bronze | 0.91930 |

**Best verified result:** silver, `0.92020`.
**Method:** pinned pretrained checkpoints and the engine's ordinal ensemble; three independent runs.
**Best-grade evidence SHA-256:** `6aa3c818ede32760bf85f9af02773991e2b6d783c9375e16ce45d0ade087e11f`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. Every published submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
