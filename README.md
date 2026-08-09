# Impulse AI — 18 medals on MLE-bench Lite-22

**18 medals (7 gold / 6 silver / 5 bronze) on OpenAI's [MLE-bench](https://github.com/openai/mle-bench)
Lite-22, every one produced end-to-end by our autonomous agent and graded by
OpenAI's official `mlebench` grader, unmodified.**

MLE-bench Lite is a 22-competition low-compute subset of OpenAI's full MLE-bench
benchmark of real Kaggle competitions. There are no per-competition hand-written
solutions in this repository: the solution code *is* the agent
(`engine-controller.py` driving `engine/`) plus its library of
general operators (`engine/operators/`). The same code runs every task.

## Medal table

| # | Competition | Medal | Official score | Solution |
|---|---|---|---|---|
| 1 | aerial-cactus-identification | 🥇 gold | 1.00000 | [solutions/aerial-cactus-identification](solutions/aerial-cactus-identification/) |
| 2 | detecting-insults-in-social-commentary | 🥇 gold | 0.91084 | [solutions/detecting-insults-in-social-commentary](solutions/detecting-insults-in-social-commentary/) |
| 3 | nomad2018-predict-transparent-conductors | 🥇 gold | 0.05373 | [solutions/nomad2018-predict-transparent-conductors](solutions/nomad2018-predict-transparent-conductors/) |
| 4 | dogs-vs-cats-redux-kernels-edition | 🥇 gold | 0.00981 | [solutions/dogs-vs-cats-redux-kernels-edition](solutions/dogs-vs-cats-redux-kernels-edition/) |
| 5 | plant-pathology-2020-fgvc7 | 🥇 gold | 0.98364 | [solutions/plant-pathology-2020-fgvc7](solutions/plant-pathology-2020-fgvc7/) |
| 6 | tabular-playground-series-dec-2021 | 🥇 gold | 0.95996 | [solutions/tabular-playground-series-dec-2021](solutions/tabular-playground-series-dec-2021/) |
| 7 | histopathologic-cancer-detection | 🥇 gold | 0.98912 | [solutions/histopathologic-cancer-detection](solutions/histopathologic-cancer-detection/) |
| 8 | spooky-author-identification | 🥈 silver | 0.186 | [solutions/spooky-author-identification](solutions/spooky-author-identification/) |
| 9 | denoising-dirty-documents | 🥈 silver | 0.01919 | [solutions/denoising-dirty-documents](solutions/denoising-dirty-documents/) |
| 10 | leaf-classification | 🥈 silver | 0.00671 | [solutions/leaf-classification](solutions/leaf-classification/) |
| 11 | mlsp-2013-birds | 🥈 silver | 0.9128 | [solutions/mlsp-2013-birds](solutions/mlsp-2013-birds/) |
| 12 | jigsaw-toxic-comment-classification-challenge | 🥈 silver | 0.98678 | [solutions/jigsaw-toxic-comment-classification-challenge](solutions/jigsaw-toxic-comment-classification-challenge/) |
| 13 | aptos2019-blindness-detection | 🥈 silver | 0.9202 | [solutions/aptos2019-blindness-detection](solutions/aptos2019-blindness-detection/) |
| 14 | dog-breed-identification | 🥉 bronze | 0.02439 | [solutions/dog-breed-identification](solutions/dog-breed-identification/) |
| 15 | the-icml-2013-whale-challenge-right-whale-redux | 🥉 bronze | 0.91633 | [solutions/the-icml-2013-whale-challenge-right-whale-redux](solutions/the-icml-2013-whale-challenge-right-whale-redux/) |
| 16 | random-acts-of-pizza | 🥉 bronze | ~0.692 | [solutions/random-acts-of-pizza](solutions/random-acts-of-pizza/) |
| 17 | text-normalization-challenge-english-language | 🥉 bronze | 0.99125 | [solutions/text-normalization-challenge-english-language](solutions/text-normalization-challenge-english-language/) |
| 18 | text-normalization-challenge-russian-language | 🥉 bronze | 0.97906 | [solutions/text-normalization-challenge-russian-language](solutions/text-normalization-challenge-russian-language/) |

Scores are the official numbers emitted by OpenAI's grader; per-competition
approach notes and evidence artifacts are in the linked solution directories.

## Why this matters

[MLE-bench](https://github.com/openai/mle-bench) is OpenAI's benchmark for
ML-engineering agents, and the standard public yardstick for whether an AI can
do real data-science work end to end: understand a competition, prepare data,
train models, iterate, and submit. Medals are awarded relative to the actual
Kaggle leaderboards — i.e. against thousands of human data-science teams, not
synthetic tasks. Lite-22 is the benchmark's low-compute split: 22 real Kaggle
competitions.

**18 of 22 medals is an 81.8% any-medal rate on Lite — above the best
published Lite result on the official leaderboard (80.3%).** Unlike most top
entries, our full agent source ships in this repository, and every medal can
be independently re-graded with OpenAI's own tooling.

### Comparison with the official leaderboard (Lite split, as of August 2026)

From the leaderboard maintained in the
[openai/mle-bench README](https://github.com/openai/mle-bench#readme)
("Low == Lite" column, any-medal rate):

| Agent | LLM(s) used | Lite any-medal (%) | Source code available |
|---|---|---|---|
| **Impulse EES (this repo)** | **Gemini** | **81.8 (18/22)** | ✓ |
| Famou-Agent 2.0 | Gemini-3-Pro-Preview | 80.3 ± 1.52 | ✗ |
| MLEvolve | Gemini-3-Pro-Preview | 80.30 ± 1.52 | ✓ |
| PiEvolve (Fractal AI Research) | Gemini-3-Pro-Preview | 80.30 ± 1.52 | ✗ |
| CAIR MARS+ | Gemini-3-Pro-Preview | 78.79 ± 1.52 | ✗ |
| AIBuildAI | Claude-Opus-4.6 | 77.27 ± 0.00 | ✗ |
| AIDE (OpenAI's reference agent) | o1-preview | 35.91 ± 1.86 | ✓ |

How to read this honestly:

- **Leaderboard figures are means over multiple seeds** (the ± is the standard
  deviation across runs). Ours is a **single autonomous run per competition** —
  a point estimate, not a mean. An apples-to-apples multi-seed evaluation is
  on our roadmap.
- **We are not (yet) an official leaderboard entry.** The comparison is
  against numbers others published, but every one of our medals was produced
  with the same unmodified official grader and is reproducible from this repo —
  see [How to verify](#how-to-verify).
- **The agent is Gemini-driven.** Impulse's agent platform is Gemini-only by
  design, and Gemini drove both the development of the EES engine and the
  July 2026 medal campaign that produced these results.

## The 4 of 22 we haven't medaled (yet)

The same agent ran all 22 Lite competitions. These four did not reach bronze
in the current campaign:

- [ranzcr-clip-catheter-line-classification](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)
- [siim-isic-melanoma-classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)
- [new-york-city-taxi-fare-prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)
- [tabular-playground-series-may-2022](https://www.kaggle.com/c/tabular-playground-series-may-2022)

We list them because the claim is "18 of 22", and the 22 is the whole split.

## How to verify

The evidence is pinned to source commit `3f528404`, tag
`mlebench-lite22-18medals`.

- **One medal in ~30 minutes:** [reproduce/QUICKSTART.md](reproduce/QUICKSTART.md)
  — run the agent on one competition, CPU-only, and let OpenAI's grader emit the
  verdict.
- **Full CPU sweep (15 medals):** [reproduce/agent-run.sh](reproduce/agent-run.sh)
  — one command: prepare data → run the agent → `mlebench grade` → medal table.
- **Full runbook:** [reproduce/VERIFY.md](reproduce/VERIFY.md) — including the
  one unavoidable manual gate: accepting each competition's rules on kaggle.com
  with your own account before OpenAI's tooling can download the data.
- **Authoritative ledger:** [reproduce/EVIDENCE.md](reproduce/EVIDENCE.md) — the
  per-medal evidence index, including what each artifact does and does not prove.

Don't trust the wrapper? Grade any `submission.csv` yourself:
`mlebench grade-sample <submission.csv> <competition>`.

## Repository layout

- `engine/` — the autonomous agent: `engine-controller.py`, the `engine-core/`
  reasoning engine, and `engine-core/operators/` (the general solution operators —
  image/audio/text/tabular — referenced from each solution README).
- `solutions/<competition>/` — per-medal approach notes plus snapshot evidence
  artifacts (`grade.json`, `result.json`, `submission.csv` where size permits).
- `reproduce/` — the verification packet: `QUICKSTART.md`, `agent-run.sh`,
  `VERIFY.md`, `EVIDENCE.md`, `Dockerfile`, `requirements.lock`,
  `data_checksums.txt`, and `gpu/` (GPU job scripts + `official_grade.json` for
  the histo/birds/jigsaw GPU runs).

## Transparency

- **Grading is OpenAI's, not ours.** Medals are computed by the official
  `mlebench` grader from https://github.com/openai/mle-bench, installed and run
  unmodified. Our only code in the loop is the agent that produces the
  submission.
- **Results are single-run.** We report the outcome of one autonomous run per
  competition, not a best-of-N selection.
- **The snapshot is dated.** The `solutions/` artifacts come from a 2026-07-07
  snapshot (`cloud-harvest-0708`) that predates later campaign fixes — only 5 of
  its `grade.json` files show medals. The authoritative per-task evidence is
  what `reproduce/agent-run.sh` regenerates fresh. See the honesty note in
  [reproduce/EVIDENCE.md](reproduce/EVIDENCE.md).
- **15 of 18 medals reproduce CPU-only in one command** via
  `reproduce/agent-run.sh`. The other 3 (histopathologic-cancer-detection gold,
  jigsaw-toxic-comment silver, aptos2019-blindness-detection silver) were won
  autonomously through our Vertex AI GPU lane and need GCP access to re-run;
  their independently re-gradeable evidence ships in `reproduce/gpu/`.
- **Scope.** MLE-bench Lite is a 22-competition low-compute subset of the full
  MLE-bench benchmark; these results are on Lite-22, not the full set.

---

© 2026 Impulse AI. All rights reserved.
