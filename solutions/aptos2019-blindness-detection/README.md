# aptos2019-blindness-detection — 🥈 silver — 0.92020 (quadratic weighted kappa)

- **Kaggle competition:** https://www.kaggle.com/c/aptos2019-blindness-detection
- **Best verified result:** 🥈 silver — **0.92020 (quadratic weighted kappa)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_finetune_ensemble.py`](../../engine/operators/image_finetune_ensemble.py)

## Approach

A 3-config diverse ordinal ensemble: diverse backbone families at different resolutions and seeds, each trained as a regression head with k-fold OOF and blended by OOF-weighted averaging so decorrelated configs dominate. Built for ordinal tasks whose test distribution differs from train; won autonomously via the Vertex GPU lane on 2026-07-13 (run `convertA_aptos3`).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.91942 |
| 2 | bronze | 0.91942 |
| 3 | bronze | 0.91930 |

**Best verified result:** silver, `0.92020`.
**Method:** public prepared images + pinned pretrained checkpoints; exact legacy ensemble; independent process confirmation.
**Best-grade evidence SHA-256:** `6aa3c818ede32760bf85f9af02773991e2b6d783c9375e16ce45d0ade087e11f`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
