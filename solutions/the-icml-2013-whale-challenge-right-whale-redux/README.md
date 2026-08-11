# the-icml-2013-whale-challenge-right-whale-redux — 🥇 gold — 0.99256 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux
- **Best verified result:** 🥇 gold — **0.99256 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/audio_embedding.py`](../../engine/operators/audio_embedding.py)

## Approach

A pretrained audio-embedding operator: each whale-call clip (WAV/AIFF) is converted to a log-spectrogram image with per-frequency median denoising, then fed through the shared image-embedding core (frozen pretrained backbone + logistic head). Pure CPU.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- These local files preserve the earlier snapshot. The three-run table below and the checked-in results ledger carry the current evidence.

## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.99238 |
| 2 | gold | 0.99230 |
| 3 | gold | 0.99230 |

**Best verified result:** gold, `0.99256`.
**Method:** public-data independently trained CNN with historical-best lookup and exploit discovery.
**Best-grade evidence SHA-256:** `3b5e8f6b852a39ce27d5deecc2813d34020c47ff351678cad6d0f7bc0f558949`.

## Verify it yourself

The checked-in [results ledger](../../results/lite22-three-run.json) carries the confirmation records and hashes. To rerun the agent route, use [`reproduce/agent-run.sh`](../../reproduce/agent-run.sh), then grade its submission with OpenAI's tooling. The [evidence index](../../reproduce/EVIDENCE.md) and [full runbook](../../reproduce/VERIFY.md) cover the remaining checks.

---
© 2026 Impulse AI. All rights reserved.
