# Impulse AI: 11 medals on MLE-bench HARD

Impulse AI holds medals on 11 of the 15 competitions in the HARD split of [MLE-bench](https://github.com/openai/mle-bench), OpenAI's benchmark of real Kaggle competitions: **7 gold / 1 silver / 3 bronze**. Nine of them come from single end-to-end runs of the Impulse engine: **60.00% on HARD, against 46.67% for the top entry on the MLE-bench main leaderboard.** Every score uses OpenAI's MLE-bench grading logic.

HARD is the 15-competition high-complexity split: 3D detection from LiDAR, chest X-ray localisation, satellite contrail segmentation, screening mammography, mRNA degradation, EEG, game video with wearable tracking. This repository contains the results ledger, per-competition solution notes, graded submissions, grading artifacts and verification tooling. The engine itself is proprietary and is not public; a live re-run under observation is available on request.

## Medal table

| # | Competition | Medal | Score | Gold / silver / bronze line | Method | Solution |
|---|---|---|---:|---|---|---|
| 1 | 3d-object-detection-for-autonomous-vehicles | 🥇 gold | 0.18128 | 0.139 / 0.053 / 0.046 | Engine run | [solution](solutions/3d-object-detection-for-autonomous-vehicles/) |
| 2 | iwildcam-2019-fgvc6 | 🥇 gold | 0.48326 | 0.212 / 0.129 / 0.114 | Engine run | [solution](solutions/iwildcam-2019-fgvc6/) |
| 3 | predict-volcanic-eruptions-ingv-oe | 🥇 gold | 2,740,004 | 3,971,366 / 4,808,057 / 4,999,330 | Engine run | [solution](solutions/predict-volcanic-eruptions-ingv-oe/) |
| 4 | rsna-breast-cancer-detection | 🥇 gold | 0.50526 | 0.49 / 0.43 / 0.41 | Engine run | [solution](solutions/rsna-breast-cancer-detection/) |
| 5 | smartphone-decimeter-2022 | 🥇 gold | 0.000 | 1.768 / 2.119 / 3.068 | Engine run (exploit discovery) | [solution](solutions/smartphone-decimeter-2022/) |
| 6 | stanford-covid-vaccine | 🥇 gold | 0.32671 | 0.34728 / 0.35175 / 0.3534 | Engine run | [solution](solutions/stanford-covid-vaccine/) |
| 7 | vinbigdata-chest-xray-abnormalities-detection | 🥇 gold | 0.4256 | 0.289 / 0.257 / 0.243 | Engine run | [solution](solutions/vinbigdata-chest-xray-abnormalities-detection/) |
| 8 | google-research-identify-contrails-reduce-global-warming | 🥈 silver | 0.69712 | 0.71059 / 0.6936 / 0.67929 | Engine run | [solution](solutions/google-research-identify-contrails-reduce-global-warming/) |
| 9 | nfl-player-contact-detection | 🥉 bronze | 0.71956 | 0.78167 / 0.72203 / 0.69652 | Engine run | [solution](solutions/nfl-player-contact-detection/) |
| 10 | hms-harmful-brain-activity-classification | 🥉 bronze | 0.36573 | 0.290807 / 0.353323 / 0.375383 | Impulse ML pipeline | [solution](solutions/hms-harmful-brain-activity-classification/) |
| 11 | rsna-miccai-brain-tumor-radiogenomic-classification | 🥉 bronze | 0.57941 | 0.60096 / 0.5815 / 0.57449 | Impulse ML pipeline | [solution](solutions/rsna-miccai-brain-tumor-radiogenomic-classification/) |

Lower is better for rows 3, 5, 6 and 10; higher is better elsewhere. Engine runs are single end-to-end runs inside the benchmark's 24-hour limit, each scored once. Scores and medals come from the evidence ledger in [results/hard15.json](results/hard15.json).

## Published HARD context

The [MLE-bench project](https://github.com/openai/mle-bench#leaderboard) publishes the High-split figures below on its main leaderboard. Its 2026-04-24 update states that the leaderboard is not taking new submissions, so these results are published here with their evidence.

| Agent | LLM(s) used | High any-medal (%) | Source code available |
|---|---|---:|---|
| **Impulse AI, all verified medals (this repository)** | GPT-6-Luna (engine runs) | **73.33 (11/15)** | ✗ |
| **Impulse EES, engine runs (this repository)** | GPT-6-Luna | **60.00 (9/15)** | ✗ |
| AIBuildAI | Claude-Opus-4.6 | 46.67 ± 0.00 | ✗ |
| CAIR MARS+ | Gemini-3-Pro-Preview | 44.44 ± 2.22 | ✗ |
| Famou-Agent 2.0 | Gemini-3-Pro-Preview | 42.22 ± 2.22 | ✗ |
| MLEvolve | Gemini-3-Pro-Preview | 42.22 ± 2.22 | ✓ |
| ML-Master 2.0 | Deepseek-V3.2-Speciale | 42.22 ± 2.22 | ✗ |
| PiEvolve (Fractal AI Research) | Gemini-3-Pro-Preview | 40.0 ± 0.00 | ✗ |
| AIDE (OpenAI's reference agent) | o1-preview | 11.67 ± 1.27 | ✓ |

The project also lists additional submissions that it marks as not directly comparable to the main leaderboard; the highest High figure there is 71.11 ± 2.22. They are not included in this table.

## What Impulse built

- **3D object detection (LiDAR):** pillar encoding into a bird's-eye-view grid with centre-based heads.
- **Wildlife camera traps (iWildCam):** a pretrained ConvNeXt-Base fine-tuned at 384 px with class balancing for the long-tailed label set, mixup and CutMix, and test-time augmentation.
- **Volcanic eruption timing:** 245 features engineered from each segment's sensor signals, with a weighted blend of LightGBM regressors.
- **Screening mammography:** per-file DICOM windowing, breast crop, high-resolution EfficientNetV2 with a breast-level second stage over both views.
- **Smartphone GNSS:** reconstruction of ground-truth positions from logs shipped in the public data.
- **mRNA degradation (OpenVaccine):** message-passing graph networks over each molecule's sequence graph, three seeds blended.
- **Chest X-ray abnormalities (VinBig):** COCO-pretrained two-stage detector fine-tuned at 800 px.
- **Contrails:** false-colour composite from infrared bands, six-member U-Net ensemble with offset-corrected mirroring in three members.
- **NFL player contact:** pairwise tracking features with a temporal second stage along each player pair's timeline.
- **Brain-tumour MRI (MICCAI radiogenomics):** a family of six EfficientNet-B2 classifiers, each over a different construction of the MRI sequences, trained without augmentation and selected by repeated patient-level cross-validation with bootstrap intervals; rank-averaged across members.
- **EEG harmful brain activity (HMS):** CNNs over EEG-window and spectrogram images, with a cross-fitted second stage built on the recording-level structure of the data index.

## Product capabilities used in these runs

Impulse's agent can use public external data, web research, pretrained models, and exploit discovery. These are enabled product capabilities, and the ledger's method field identifies the route used for each result.

## The four HARD tasks without a medal

- [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection)
- [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection)
- [Vesuvius Challenge: Ink Detection](https://www.kaggle.com/c/vesuvius-challenge-ink-detection)
- [BMS Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation)

## How to verify

- [HARD evidence index](reproduce/EVIDENCE_HARD.md): per-task results, methods and SHA-256 hashes.
- [Results verifier](reproduce/verify_results.py): validate the checked-in ledgers, graded-submission hashes, links, and public claims.

Every published submission can be independently re-graded with OpenAI's unmodified `mlebench` grader; no Impulse code is required:

```bash
mlebench grade-sample <submission.csv> <competition>
```

## Scope

This page covers the 15-competition HARD (High) split. The Lite-22 results are on the [main page](README.md).

---

© 2026 Impulse AI. All rights reserved.
