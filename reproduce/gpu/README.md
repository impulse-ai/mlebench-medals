# Class-B GPU / assisted medals — capture + independent-grade evidence

These three medals were **NOT** produced by the one-command autonomous agent
(`agent-run.sh`). They were produced by **manual, hand-scripted Vertex AI GPU
jobs** run during the 2026-07-09 campaign. We mark them **Class-B (assisted)**
and disclose that plainly. The grading is still OpenAI's `mlebench`, unmodified —
you can re-run the grader yourself against the captured submissions (birds and
jigsaw below; histo needs its private data prepared locally).

**Capability 0** (tracked as PR #832) wires the `image_finetune` operator to run these GPU jobs
*through the agent*, which converts histo/birds from Class-B to Class-A. Until it
merges, these are honestly assisted.

| Task | Medal | Official score | Threshold (medal) | Re-gradable here? | Backbone / method |
|---|---|---|---|---|---|
| histopathologic-cancer-detection | 🥇 gold | 0.99154 | bronze 0.9738 | no (private data not prepared locally) | EfficientNet-B0 (`tf_efficientnet_b0_ns`), 96px, T4 |
| mlsp-2013-birds | 🥈 silver | **0.93143** | silver 0.90038 | **yes — verified locally** | ResNet18 spectrogram CNN + classical blend (OOF-selected) |
| jigsaw-toxic-comment-classification-challenge | 🥉 bronze | **0.98663** | bronze 0.98639 | **yes — verified locally** | distilbert + roberta + TF-IDF blend (OOF weights) |

`official_grade.json` in `birds/` and `jigsaw/` were produced by grading the
captured submissions with OpenAI's `mlebench` on this machine (2026-07-09). histo's
0.99154 is the campaign's official grade (see the ledger); it is not re-gradeable
here because its private answers are not in the local cache.

## What's captured, and where the originals live

Source of truth: GCS `gs://engg-ai-experimental-gpu-artifacts/{histo,birds,jigsaw}/`
and the runner VM `mle-bench-runner` (zone `us-central1-a`, home dir).

### histo/  (gold 0.99154)
- `histo_gpu_job_t4.py` — the Vertex GPU training job (EfficientNet-B0, tail-reserve for the test forward pass).
- `histo_submit_t4.py` — the CustomJob submitter.
- `histo_grade.py` — grades the submission via the mlebench python API.
- `holdout_metrics.json` — winning run's internal holdout (AUC 0.99091, 3 epochs, Tesla T4).
- `progress_inference.json` — inference-phase telemetry.
- Winning submission (2.5 MB, not committed): `gs://engg-ai-experimental-gpu-artifacts/histo/out/run_1783629882/submission.csv`
  sha256 `e245957ce7fe79186e474e0936c0695881721a2ce602b116ef68aac825bd105a`

### birds/  (silver 0.93143 — re-verified locally)
- `birds_submission.csv` — the graded silver submission (1216 rows). **Committed** — grade it yourself:
  `mlebench grade-sample reproduce/gpu/birds/birds_submission.csv mlsp-2013-birds`
  sha256 `252c9238572f6d3a10f5547b0ab3d73c1207c18d5a49501b95cf2fbd41be9ad0`
- `official_grade.json` — the mlebench grade output (silver, 0.93143), produced on this machine.
- `METRICS_birds.md` — the campaign's honest method + selection-honesty writeup (OOF-selected blend, test graded once).
- `birds.py`, `embed.py`, `finetune.py`, `final.py`, `verify.py` — the pipeline (features / ResNet50 embeddings / CNN finetune / blend+grade / format-check).

### jigsaw/  (bronze 0.98663 — re-verified locally)
- `jigsaw_gpu_job_t4_v2.py` — distilbert-base-uncased Vertex GPU job (holdout mean-AUC 0.99081).
- `jigsaw_roberta_v3_job.py` — roberta-base Vertex GPU job (adds diversity to the blend).
- `jigsaw_submit_t4_v2.py` — CustomJob submitter.
- `blend3b.py` — **the winning blend**: grid-search on the held-out OOF simplex over
  {distilbert, roberta, TF-IDF} (raw + rank space), applied to test, graded once. Honest OOF selection, no test-tuning.
- `distilbert_metrics.json`, `roberta_metrics.json` — component holdout metrics.
- `grade_jigsaw.py` — the mlebench-API grade script.
- `official_grade.json` — the mlebench grade output (bronze, 0.98663), produced on this machine.
- Winning submission (18 MB, not committed): VM `mle-bench-runner:~/jigsaw_blend3b_submission.csv`
  sha256 `c70cdb612f0a84884e3c7d8e34d4031a8d518de1f64321a1fe5a7d0e26699efa`
  (re-download: `gcloud compute scp mle-bench-runner:~/jigsaw_blend3b_submission.csv . --zone=us-central1-a`)

## Independently re-grade the two committed / re-downloadable submissions

```bash
# birds (committed here):
mlebench grade-sample reproduce/gpu/birds/birds_submission.csv mlsp-2013-birds
#   -> silver_medal: true, score 0.93143

# jigsaw (pull the 18MB submission first, then grade):
gcloud compute scp mle-bench-runner:~/jigsaw_blend3b_submission.csv /tmp/ --zone=us-central1-a
mlebench grade-sample /tmp/jigsaw_blend3b_submission.csv jigsaw-toxic-comment-classification-challenge
#   -> bronze_medal: true, score 0.98663
```

Both require the task's data prepared locally (`mlebench prepare -c <task>`).

## Honesty notes

- **These used GPUs and human orchestration.** The scripts here are the actual
  job code; a verifier reproducing them needs a Vertex AI project + T4 quota +
  Kaggle rule-acceptance, and would run each script by hand. That is exactly why
  we class them B, not A.
- **Selection was honest** (birds/jigsaw): blend weights chosen on held-out OOF,
  the private test set graded once. See `birds/METRICS_birds.md` and `jigsaw/blend3b.py`.
- **The fourth Class-B medal, leaf-classification (silver 0.00671), is NOT a GPU
  medal** — it is operator-gate-proven (the #830 validation-reliability blend on
  the default operator pool) but not yet re-confirmed in a full autonomous sweep,
  so it is not in `agent-run.sh`'s default set. See reproduce/EVIDENCE.md.
