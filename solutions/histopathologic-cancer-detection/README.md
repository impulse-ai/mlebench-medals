# histopathologic-cancer-detection — 🥇 gold — 0.98912 (ROC AUC)

- **Kaggle competition:** https://www.kaggle.com/c/histopathologic-cancer-detection
- **Medal:** 🥇 gold — official score **0.98912 (ROC AUC)**, graded with OpenAI's
  unmodified [`mlebench`](https://github.com/openai/mle-bench) grader.
- **Solution operator:** [`bench/ees_core/operators/image_finetune.py`](../../bench/ees_core/operators/image_finetune.py)

## Approach

End-to-end fine-tuning of a torchvision backbone on the histopathology patches, run by the agent itself as a Vertex AI T4 GPU job through the Cap-0 GPU seam (run `repin_histo2`, 2026-07-10). This is one of the 3 medals won via the GPU lane; re-running it needs GCP Vertex access.

## Evidence in this directory

- Snapshot artifacts from the 2026-07-07 `cloud-harvest-0708` run: `grade.json`, `result.json`, `submission.csv`.
- NOTE: the snapshot `grade.json` here predates the GPU-lane run and shows no medal; the authoritative evidence is the autonomous GPU run recorded in `reproduce/EVIDENCE.md`.

## Verify it yourself

The authoritative per-task evidence is a fresh regeneration: run
[`reproduce/reproduce.sh`](../../reproduce/reproduce.sh) (or follow
[`reproduce/QUICKSTART.md`](../../reproduce/QUICKSTART.md)) and read the
`grade.json` OpenAI's grader emits. See [`reproduce/EVIDENCE.md`](../../reproduce/EVIDENCE.md)
for the full 18-medal ledger and [`reproduce/VERIFY.md`](../../reproduce/VERIFY.md)
for the runbook.

GPU lane: the job scripts and the independently re-gradeable `official_grade.json` for this task's GPU runs live in [`reproduce/gpu/histo/`](../../reproduce/gpu/histo/).

---
© 2026 Impulse AI. All rights reserved.
