# Re-grade one published Lite-22 submission

This is the short path through [`VERIFY.md`](VERIFY.md). The published result is
**19 medals on 22 MLE-bench Lite competitions, 86.36% ± 0.00 across three
confirmed runs**, with **11 gold / 5 silver / 3 bronze** as the best-result
breakdown. The 19 tasks include
[Tabular Playground Series May 2022](../solutions/tabular-playground-series-may-2022/).

The engine that produced these submissions is proprietary and is not part of
this repository. You don't need it to check the scores: every published
submission can be independently re-graded with OpenAI's unmodified `mlebench`
grader.

## 1. Set up the grading environment

Run these commands from a clone of this repository:

```bash
git checkout main
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r reproduce/requirements.lock
```

Add your Kaggle API token at `~/.kaggle/kaggle.json`, then accept the rules for
[NOMAD 2018](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/rules)
while signed in to Kaggle. Kaggle rejects data preparation until your account has
accepted that competition's rules.

## 2. Re-grade one published submission

Prepare the competition data, then grade the committed submission:

```bash
mlebench prepare -c nomad2018-predict-transparent-conductors
mlebench grade-sample \
  solutions/nomad2018-predict-transparent-conductors/submission.csv \
  nomad2018-predict-transparent-conductors
```

The published best result for NOMAD is gold at `0.05373`; its three confirmation
scores are `0.05479 / 0.05997 / 0.05993`. Other tasks work the same way: each
solution directory that carries a `submission.csv` can be re-graded with
`mlebench grade-sample <submission.csv> <task>`.

## 3. Check the complete publication

```bash
PYTHONDONTWRITEBYTECODE=1 python reproduce/verify_results.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v tests.test_results
```

The verifier checks the 22-task ledger, the 19 medal pages, both public result
tables, evidence hashes, stale claims, and relative Markdown links. For the full
verification flow, hardware notes, and evidence anchors, continue with
[`VERIFY.md`](VERIFY.md).

## Common failures

- A Kaggle `403` means the signed-in account hasn't accepted the competition rules.
- If preparation produced different bytes, run
  `python reproduce/generate_checksums.py --verify` and inspect the reported file.
- If grading returns no score, confirm that the active environment came from the
  pinned requirement file above.
