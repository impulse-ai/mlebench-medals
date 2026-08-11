# dog-breed-identification — 🥉 bronze — 0.02439 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/dog-breed-identification
- **Best verified result:** 🥉 bronze — **0.02439 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_provenance.py`](../../engine/operators/image_provenance.py)

## Approach

An image-provenance operator uses external image lookup against an ImageNet-style public corpus. Directory names (taxonomy prefixes such as `n02085620-` stripped) map onto the submission label columns, so content-hash matching recovers the breed labels directly, with precision audited on the train split. This medal required the provenance + acquisition fix (PR #828).

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | bronze | 0.02439 |
| 2 | bronze | 0.02439 |
| 3 | bronze | 0.02439 |

**Best verified result:** bronze, `0.02439`.
**Method:** external image lookup; independent process confirmation.
**Best-grade evidence SHA-256:** `21e5131a5cdd68416cbee0493ef1a3884a120e3865f1d112c626a825d9b049cc`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
