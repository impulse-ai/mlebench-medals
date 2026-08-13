# Verify the MLE-bench Lite-22 result

Impulse AI earned medals on **19 of 22 MLE-bench Lite competitions: 86.36%
± 0.00 across three confirmed runs**. Each confirmation column contains the same
19 medal tasks. The best verified results are **11 gold / 5 silver / 3 bronze**,
including a silver best result for
[Tabular Playground Series May 2022](../solutions/tabular-playground-series-may-2022/).

The checked-in [results ledger](../results/lite22-three-run.json) holds the exact
scores, methods, and SHA-256 evidence anchors. [`EVIDENCE.md`](EVIDENCE.md)
renders the same records as a readable table.

The engine that produced these results is proprietary and is not part of this
repository; Impulse offers a live re-run under observation on request.

## Check the publication without downloading competition data

Start from the public `main` branch:

```bash
git checkout main
PYTHONDONTWRITEBYTECODE=1 python reproduce/verify_results.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v tests.test_results
```

The first command validates these publication contracts:

- 22 unique tasks, 19 medal tasks, three confirmation runs, and a per-run medal
  count of 19;
- the 11/5/3 best-medal breakdown and all-null records for NYC Taxi Fare,
  RANZCR, and SIIM-ISIC;
- the evidence board and manifest anchors, per-task methods and best-grade hashes;
- exact README, evidence-table, and solution confirmation content; and
- relative Markdown links plus stale-claim scans across public result documents.

The published evidence anchors are:

```text
board    663b9e3a56a12d0c69ac0c547921332d8341ef46bd4812a55a2a9a22bb2680ea
manifest c6d2ad86653719ab55beabb70e549ffdd9e6c674790484dd6ce45af668cf38c1
```

## Prepare a grading environment

Use Python 3.11 and the pinned dependencies:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r reproduce/requirements.lock
```

MLE-bench downloads Kaggle competition data through your Kaggle account. Save an
API token as `~/.kaggle/kaggle.json`, visit the rules page for each competition
you plan to grade against, and accept its rules while signed in. A missing
acceptance usually appears as a `403` during preparation.

To confirm that prepared data matches the recorded inputs:

```bash
python reproduce/generate_checksums.py --verify
```

## Re-grade published submissions

Every published submission can be independently re-graded with OpenAI's
unmodified MLE-bench grader — no Impulse code is required:

```bash
mlebench grade-sample solutions/<task>/submission.csv <task>
```

Committed submissions live next to each solution page that carries one, and the
GPU-lane captures in [`gpu/`](gpu/) include their own `official_grade.json`
outputs. These commands use OpenAI's MLE-bench grading logic. The English
text-normalization route documents its CSV compatibility handling explicitly;
the public method text appears identically in the JSON ledger, evidence table,
and solution page.

## Compute and route notes

The checked-in [`gpu/`](gpu/) material captures the hand-run GPU jobs and their
official grade outputs for the routes that used GPUs, while each solution page
states the method behind its current confirmation record.

Public external data, pretrained checkpoints, web research, and dataset-structure
discovery are enabled product capabilities. The ledger names external target or
image lookups where they occur; it doesn't include private labels, credentials,
raw benchmark data, or model artifacts.

For a hermetic grading environment, build [`Dockerfile`](Dockerfile). For a
quick single-submission re-grade, use [`QUICKSTART.md`](QUICKSTART.md).
