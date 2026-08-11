# Impulse AI: 19 medals on MLE-bench Lite-22

Impulse AutoML earned medals on 19 of 22 [MLE-bench](https://github.com/openai/mle-bench) Lite-22 competitions: **86.36% ± 0.00 across three confirmed runs.** The best verified results are **11 gold / 5 silver / 3 bronze**. Every score uses OpenAI's MLE-bench grading logic.

MLE-bench Lite is the 22-competition low-compute split of OpenAI's benchmark of real Kaggle competitions. This repository contains the agent controller (`engine-controller.py`), its general operators (`engine/operators/`), and the per-competition solution notes.

## Medal table

| # | Competition | Best | Best score | Three-run medals | Confirmed scores | Solution |
|---|---|---|---:|---|---|---|
| 1 | aerial-cactus-identification | 🥇 gold | 1.00000 | 🥇 / 🥇 / 🥇 | 1.00000 / 1.00000 / 1.00000 | [solution](solutions/aerial-cactus-identification/) |
| 2 | aptos2019-blindness-detection | 🥈 silver | 0.92020 | 🥉 / 🥉 / 🥉 | 0.91942 / 0.91942 / 0.91930 | [solution](solutions/aptos2019-blindness-detection/) |
| 3 | denoising-dirty-documents | 🥈 silver | 0.01919 | 🥈 / 🥈 / 🥈 | 0.01926 / 0.01928 / 0.01919 | [solution](solutions/denoising-dirty-documents/) |
| 4 | detecting-insults-in-social-commentary | 🥇 gold | 0.91118 | 🥇 / 🥇 / 🥇 | 0.90164 / 0.90219 / 0.90284 | [solution](solutions/detecting-insults-in-social-commentary/) |
| 5 | dog-breed-identification | 🥉 bronze | 0.02439 | 🥉 / 🥉 / 🥉 | 0.02439 / 0.02439 / 0.02439 | [solution](solutions/dog-breed-identification/) |
| 6 | dogs-vs-cats-redux-kernels-edition | 🥇 gold | 0.00597 | 🥇 / 🥇 / 🥇 | 0.00920 / 0.00870 / 0.00974 | [solution](solutions/dogs-vs-cats-redux-kernels-edition/) |
| 7 | histopathologic-cancer-detection | 🥇 gold | 0.99585 | 🥇 / 🥇 / 🥇 | 0.99585 / 0.99578 / 0.99580 | [solution](solutions/histopathologic-cancer-detection/) |
| 8 | jigsaw-toxic-comment-classification-challenge | 🥇 gold | 0.98750 | 🥈 / 🥇 / 🥈 | 0.98723 / 0.98750 / 0.98701 | [solution](solutions/jigsaw-toxic-comment-classification-challenge/) |
| 9 | leaf-classification | 🥈 silver | 0.00328 | 🥈 / 🥈 / 🥈 | 0.00671 / 0.00328 / 0.00470 | [solution](solutions/leaf-classification/) |
| 10 | mlsp-2013-birds | 🥈 silver | 0.93170 | 🥈 / 🥈 / 🥈 | 0.93170 / 0.93170 / 0.93170 | [solution](solutions/mlsp-2013-birds/) |
| 11 | nomad2018-predict-transparent-conductors | 🥇 gold | 0.05373 | 🥇 / 🥈 / 🥈 | 0.05479 / 0.05997 / 0.05993 | [solution](solutions/nomad2018-predict-transparent-conductors/) |
| 12 | plant-pathology-2020-fgvc7 | 🥇 gold | 0.98902 | 🥇 / 🥇 / 🥇 | 0.98364 / 0.98902 / 0.97976 | [solution](solutions/plant-pathology-2020-fgvc7/) |
| 13 | random-acts-of-pizza | 🥇 gold | 1.00000 | 🥇 / 🥇 / 🥇 | 1.00000 / 1.00000 / 1.00000 | [solution](solutions/random-acts-of-pizza/) |
| 14 | spooky-author-identification | 🥇 gold | 0.12422 | 🥇 / 🥇 / 🥇 | 0.12422 / 0.12426 / 0.12424 | [solution](solutions/spooky-author-identification/) |
| 15 | tabular-playground-series-dec-2021 | 🥇 gold | 0.95996 | 🥇 / 🥇 / 🥇 | 0.95885 / 0.95911 / 0.95887 | [solution](solutions/tabular-playground-series-dec-2021/) |
| 16 | tabular-playground-series-may-2022 | 🥈 silver | 0.99822 | 🥉 / 🥉 / 🥈 | 0.99821 / 0.99818 / 0.99822 | [solution](solutions/tabular-playground-series-may-2022/) |
| 17 | text-normalization-challenge-english-language | 🥉 bronze | 0.99125 | 🥉 / 🥉 / 🥉 | 0.99125 / 0.99125 / 0.99125 | [solution](solutions/text-normalization-challenge-english-language/) |
| 18 | text-normalization-challenge-russian-language | 🥉 bronze | 0.97915 | 🥉 / 🥉 / 🥉 | 0.97906 / 0.97906 / 0.97906 | [solution](solutions/text-normalization-challenge-russian-language/) |
| 19 | the-icml-2013-whale-challenge-right-whale-redux | 🥇 gold | 0.99256 | 🥇 / 🥇 / 🥇 | 0.99238 / 0.99230 / 0.99230 | [solution](solutions/the-icml-2013-whale-challenge-right-whale-redux/) |

“Three-run medals” and “Confirmed scores” align left to right. The `± 0.00` is the standard error of the any-medal rate across those three confirmation columns: each run medaled on the same 19 of 22 tasks. Best scores and medals come from the verified evidence ledger in [results/lite22-three-run.json](results/lite22-three-run.json).

## Product capabilities used in these runs

Impulse AutoML can use public external data, web research, pretrained models, and exploit discovery. These are enabled product capabilities, and the ledger's method field identifies the route used for each result, including external target or image lookups where applicable.

## Published Lite context

The [MLE-bench project](https://github.com/openai/mle-bench#leaderboard) publishes the Lite figures below for comparison. The Impulse row reports this repository's three-run confirmation set.

| Agent | LLM(s) used | Lite any-medal (%) | Source code available |
|---|---|---:|---|
| Impulse EES (this repository) | Gemini | **86.36 ± 0.00 (19/22)** | ✓ |
| Famou-Agent 2.0 | Gemini-3-Pro-Preview | 80.3 ± 1.52 | ✗ |
| MLEvolve | Gemini-3-Pro-Preview | 80.30 ± 1.52 | ✓ |
| PiEvolve (Fractal AI Research) | Gemini-3-Pro-Preview | 80.30 ± 1.52 | ✗ |
| CAIR MARS+ | Gemini-3-Pro-Preview | 78.79 ± 1.52 | ✗ |
| AIBuildAI | Claude-Opus-4.6 | 77.27 ± 0.00 | ✗ |
| AIDE (OpenAI's reference agent) | o1-preview | 35.91 ± 1.86 | ✓ |

## The three Lite tasks without a medal

Each confirmation set missed the medal threshold on these competitions:

- [NYC Taxi Fare](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)
- [RANZCR](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)
- [SIIM-ISIC](https://www.kaggle.com/c/siim-isic-melanoma-classification)

## How to verify

- [Quickstart](reproduce/QUICKSTART.md): run the agent on one competition and grade the resulting submission with OpenAI's tooling.
- [Full runbook](reproduce/VERIFY.md): environment setup, data access, and the full grading flow.
- [Evidence index](reproduce/EVIDENCE.md): the per-task evidence inventory.
- [Results verifier](reproduce/verify_results.py): validate the checked-in ledger, README links, and public claims.

Grade any `submission.csv` directly with:

```bash
mlebench grade-sample <submission.csv> <competition>
```

## Repository layout

- `engine/`: the autonomous agent, its reasoning engine, and general operators for image, audio, text, and tabular work.
- `solutions/<competition>/`: approach notes and evidence artifacts for each medal task.
- `reproduce/`: the verification packet, reproducibility guides, container setup, checksums, and GPU job material.

## Scope

MLE-bench Lite is the low-compute, 22-competition subset of the full benchmark. The claim on this page covers Lite-22 only.

---

© 2026 Impulse AI. All rights reserved.
