# text-normalization-challenge-russian-language — 🥉 bronze — 0.97906 (token-level accuracy)

- **Kaggle competition:** https://www.kaggle.com/c/text-normalization-challenge-russian-language
- **Medal:** 🥉 bronze — official score **0.97906 (token-level accuracy)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/seq2seq_lookup.py`](../../engine/operators/seq2seq_lookup.py)

## Approach

The same general seq2seq lookup operator as the English task — exact-match / most-frequent token normalization learned from training pairs, detected purely from data structure with no task-id or language-specific special-casing.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the campaign fixes and shows no medal; the authoritative evidence is regeneration via `reproduce/agent-run.sh`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
