# Reproduce and grade one Lite-22 result

This is the short path through [`VERIFY.md`](VERIFY.md). The published result is
**19 medals on 22 MLE-bench Lite competitions, 86.36% ± 0.00 across three
confirmed runs**, with **11 gold / 5 silver / 3 bronze** as the best-result
breakdown. The 19 tasks include
[Tabular Playground Series May 2022](../solutions/tabular-playground-series-may-2022/).

## 1. Set up the repository

Run these commands from a clone of this repository:

```bash
git checkout main
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r reproduce/requirements.lock
pip install -r requirements.txt
```

Add your Kaggle API token at `~/.kaggle/kaggle.json`, then accept the rules for
[NOMAD 2018](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/rules)
while signed in to Kaggle. Kaggle rejects data preparation until your account has
accepted that competition's rules.

## 2. Run one task

```bash
reproduce/agent-run.sh --tasks "nomad2018-predict-transparent-conductors"
```

The command prepares the competition data, runs the agent, grades the submission
with OpenAI's MLE-bench grading logic, and writes a timestamped directory under
`runs/`.

Inspect the newest run:

```text
runs/verify-<timestamp>/medal_table.tsv
runs/verify-<timestamp>/nomad2018-predict-transparent-conductors/grade.json
runs/verify-<timestamp>/nomad2018-predict-transparent-conductors/bundle/
```

You can grade its submission directly as well:

```bash
mlebench grade-sample \
  runs/verify-<timestamp>/nomad2018-predict-transparent-conductors/bundle/submission/submission.csv \
  nomad2018-predict-transparent-conductors
```

The published best result for NOMAD is gold at `0.05373`; its three confirmation
scores are `0.05479 / 0.05997 / 0.05993`.

## 3. Check the complete publication

```bash
PYTHONDONTWRITEBYTECODE=1 python reproduce/verify_results.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v tests.test_results
```

The verifier checks the 22-task ledger, the 19 medal pages, both public result
tables, evidence hashes, stale claims, and relative Markdown links. For the full
run flow, hardware notes, and evidence anchors, continue with [`VERIFY.md`](VERIFY.md).

## Common failures

- A Kaggle `403` means the signed-in account hasn't accepted the competition rules.
- If preparation produced different bytes, run
  `python reproduce/generate_checksums.py --verify` and inspect the reported file.
- If grading returns no score, confirm that the active environment came from both
  pinned requirement files above.
