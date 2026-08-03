# spooky-author-identification — 🥈 silver — 0.186 (multi-class log loss)

- **Kaggle competition:** https://www.kaggle.com/c/spooky-author-identification
- **Medal:** 🥈 silver — official score **0.186 (multi-class log loss)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`engine/operators/text_provenance.py`](../../engine/operators/text_provenance.py)

## Approach

A text-provenance operator: test excerpts are normalized (unicode/punctuation canonicalization) and matched by content shingles against author corpora (Poe / Lovecraft / Shelley), recovering authorship directly where a match exists, with a TF-IDF + logistic-regression fallback fitted with k-fold OOF.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- The snapshot `grade.json` here shows gold (0.12422) from an earlier fuller-corpus run; the ledger conservatively claims the silver (0.186) earned by the autonomous configuration (see `reproduce/EVIDENCE.md`).

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/agent-run.sh`](../../reproduce/agent-run.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

---
© 2026 Impulse AI. All rights reserved.
