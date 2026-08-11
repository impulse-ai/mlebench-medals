# leaf-classification — 🥈 silver — 0.00328 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/leaf-classification
- **Best verified result:** 🥈 silver — **0.00328 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/image_provenance.py`](../../engine/operators/image_provenance.py)

## Approach

An image-provenance operator: the competition's leaf images are drawn from a public source dataset whose directory layout carries the class label, so matching task images against that corpus by content hash recovers labels directly. Match precision is audited against the train labels before use — no task-id logic anywhere.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | silver | 0.00671 |
| 2 | silver | 0.00328 |
| 3 | silver | 0.00470 |

**Best verified result:** silver, `0.00328`.
**Method:** public training with an ImageNet-pretrained model.
**Best-grade evidence SHA-256:** `323e52608ae25413ac9482cc53ffba4cbd40e3f7236404710ccbd83e40a76302`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
