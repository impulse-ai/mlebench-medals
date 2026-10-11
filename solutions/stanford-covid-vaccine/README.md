# stanford-covid-vaccine — 🥇 gold — 0.32671

- **Kaggle competition:** https://www.kaggle.com/c/stanford-covid-vaccine
- **Result:** 🥇 gold — **0.32671**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** `molecular_gnn` (Impulse proprietary operator family; source not published)

## Approach

A message-passing graph neural network over each RNA molecule's sequence graph: four message-passing rounds with a 128-dimensional hidden state and per-nucleotide outputs for the five targets (reactivity and four degradation rates). Each model trained for 40 epochs with a held-out validation split of the training records.

The final submission is a weighted blend of three models trained with different seeds (weights 0.32 / 0.32 / 0.36).

## Evidence in this directory

- `submission.csv`: the graded submission. Its SHA-256 is listed below and in the results ledger.
- `grade.json`: the `mlebench` grade report for that submission (the grader host's file path is omitted).

## Result

**Result:** gold, `0.32671`.
**Medal lines (gold / silver / bronze):** 0.34728 / 0.35175 / 0.3534; lower is better.
**Method:** Engine run.
**LLM:** GPT-6-Luna.
**Run wall time:** 1.51 hours.
**Graded:** 2026-10-08.
**Submission SHA-256:** `ceda0110fb017c81e5bd17ddfae420c2ea1c920dadcbf213b42d4cf24df0e4e4`.
**Grade report SHA-256:** `e228739b6ffc947d6adf3efc829550bb5a0724924b05cafb64c421216b3cdaa3`.

## Verify it yourself

The checked-in [results ledger](../../results/high15.json) carries this record and its hashes. The submission can be independently re-graded with OpenAI's unmodified `mlebench grade-sample` tooling — no Impulse code required. The [High evidence index](../../reproduce/EVIDENCE_HIGH.md) lists every High result.

---
© 2026 Impulse AI. All rights reserved.
