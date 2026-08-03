# Reproduce a medal in ~30 minutes

The 10-minute-read version of [`VERIFY.md`](VERIFY.md). You will run our
autonomous agent on one MLE-bench Lite competition and have **OpenAI's grader**
— not us — tell you it medals. CPU only; a laptop works.

## 1. One-time setup (~10 min)

```bash
# a) clone at the release tag and install the pinned environment
git checkout mlebench-lite22-18medals
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r reproduce/requirements.lock   # pandas 3.0.3 is load-bearing
pip install -r requirements.txt        # installs OpenAI's mlebench grader

# b) Kaggle credentials (the grader downloads competition data through Kaggle)
#    kaggle.com/settings -> "Create New Token" -> save as ~/.kaggle/kaggle.json
```

Then accept the competition rules **while logged in to Kaggle** — this is the
one step no one can automate for you. For the quick run below you only need:

- https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/rules
- https://www.kaggle.com/c/detecting-insults-in-social-commentary/rules

(Full sweep: accept all 15 links in `VERIFY.md` §a.)

## 2. Run the agent on two tasks (~20 min on a laptop)

```bash
reproduce/agent-run.sh --tasks "nomad2018-predict-transparent-conductors detecting-insults-in-social-commentary"
```

This prepares the data, runs the agent with default settings (no
task-specific configuration exists anywhere — same code for every task), and
grades the submission with unmodified `mlebench`.

## 3. Read the verdict

```
runs/verify-<ts>/medal_table.tsv        # the summary table
runs/verify-<ts>/<task>/grade.json      # OpenAI's raw verdict per task
runs/verify-<ts>/<task>/bundle/         # submission + exact code + env + sha256s
```

Expected: **nomad2018 → gold (~0.054)** and **detecting-insults → gold
(~0.911)**. Don't trust our wrapper? Grade the submission yourself:

```bash
mlebench grade-sample runs/verify-<ts>/<task>/bundle/submission/submission.csv <task-id>
```

## 4. The full claim (optional)

- **All 15 CPU medals:** accept all rules, then `reproduce/agent-run.sh`
  (≤4 h/task; image/audio tasks are the slow ones — see `VERIFY.md` §b for
  hardware notes, §d for the honest hardware/time disclosure).
- **The 3 GPU medals** (histopathologic gold, jigsaw silver, aptos silver)
  were won by the agent submitting its own Vertex AI T4 jobs; they need a GCP
  project, so they ship as independently re-gradeable bundles instead — see
  `VERIFY.md` §c/§e and `gpu/`.

## If something breaks

- `403` / rules error during prepare → you skipped the rules-acceptance click
  for that competition. Accept, re-run (`agent-run.sh` is resumable).
- Grades come back `None` → check `pip show pandas` says **3.0.3** (the lock
  file pins it for a reason).
- Data checksums: `python reproduce/generate_checksums.py --verify` confirms
  you prepared the exact bytes we graded against.
