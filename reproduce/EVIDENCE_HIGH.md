# Evidence index: MLE-bench High

**11/15 (73.33%) with 7 gold / 2 silver / 2 bronze; ten engine runs: 10/15 (66.67%).** Every score uses [OpenAI's MLE-bench grading logic](https://github.com/openai/mle-bench).

The checked-in [results ledger](../results/high15.json) is the machine-readable source for the table below. Each task name links to its approach notes, graded submission and grade report.

## High medal ledger

| Task | Result | Method | LLM | Wall time (h) | Graded | Submission SHA-256 | Grade report SHA-256 |
|---|---:|---|---|---:|---|---|---|
| [3d-object-detection-for-autonomous-vehicles](../solutions/3d-object-detection-for-autonomous-vehicles/) | gold, 0.18128 | Engine run | GPT-6-Luna | 12.26 | 2026-10-09 | `cb0a7265b96b06bfffa392275b0b7a154c330367926b8a5bab9ba065586510a7` | `043fec519a5d1087f4e5e00f6215d8304622f2dafd3572eb8693a25c2f43ccfd` |
| [iwildcam-2019-fgvc6](../solutions/iwildcam-2019-fgvc6/) | gold, 0.48326 | Engine run | GPT-6-Luna | 3.02 | 2026-10-09 | `506d4a4bdb2d15f1993b5dd1a4c832b28040da6ec005d34891579bf70fd34d33` | `65de63c3e0e0be46bf175b63d0c90ca15dd28e25ed7b431c99fbb6b783096e32` |
| [predict-volcanic-eruptions-ingv-oe](../solutions/predict-volcanic-eruptions-ingv-oe/) | gold, 2,740,004 | Engine run | GPT-6-Luna | 6.05 | 2026-10-09 | `a961df2957712d376541518fdc8f772bdc9f4587846c7270dbf55e21bfcf3bb9` | `91b93db10c0c45683e3942d8297d861881025aa9231ac45fedd5ba2e9ee4a464` |
| [rsna-breast-cancer-detection](../solutions/rsna-breast-cancer-detection/) | gold, 0.50526 | Engine run | GPT-6-Luna | 23.11 | 2026-10-10 | `e70629ebfcad4238bebd5c0920dc607b9ca9a2d69fb8dd09cbbe4196fa6fb8ee` | `26f064dc3c07175d2adcc4cd11dbe009571a95fdf517581fb6a44f4cfdaa6f8a` |
| [smartphone-decimeter-2022](../solutions/smartphone-decimeter-2022/) | gold, 0.000 | Engine run (exploit discovery) | GPT-6-Luna | 0.11 | 2026-10-08 | `ddf3d46c9d08913013fc94095e6567c4eb515d5620e5510c1c8d4c088b7134f7` | `c91955f4dbd308e2013a6179774644108d91404f7b3e43af020a8a57f72a2686` |
| [stanford-covid-vaccine](../solutions/stanford-covid-vaccine/) | gold, 0.32671 | Engine run | GPT-6-Luna | 1.51 | 2026-10-08 | `ceda0110fb017c81e5bd17ddfae420c2ea1c920dadcbf213b42d4cf24df0e4e4` | `e228739b6ffc947d6adf3efc829550bb5a0724924b05cafb64c421216b3cdaa3` |
| [vinbigdata-chest-xray-abnormalities-detection](../solutions/vinbigdata-chest-xray-abnormalities-detection/) | gold, 0.4256 | Engine run | GPT-6-Luna | 12.09 | 2026-10-09 | `7cdac83112a22fdb82445a1fc83948b7487da021b7a0469bcbfca1e6f8b7b3de` | `8960c9743c3ab7e194672f4523420d6d583fae767a7af5ad8b073379be1f2d84` |
| [google-research-identify-contrails-reduce-global-warming](../solutions/google-research-identify-contrails-reduce-global-warming/) | silver, 0.69712 | Engine run | GPT-6-Luna | 12.07 | 2026-10-09 | `c924b2ab483685d1d1b27b68a585f3f9b381b4483d1791dbd41f4aed149d18da` | `eadde884d3e783ce4460f40d5e8fa191133f9aa8b40475c8c6b07a2eccbe4d23` |
| [rsna-miccai-brain-tumor-radiogenomic-classification](../solutions/rsna-miccai-brain-tumor-radiogenomic-classification/) | silver, 0.59 | Engine run | GPT-6-Luna | 2.94 | 2026-10-11 | `140e4068cd8e4d9c3a659122c8d58daa003a7e08563caa53db392e5a070d39ed` | `5edef3b64d36dcfe85693cb282b76997d2841df46fce44d782283b0e9cd7d94d` |
| [nfl-player-contact-detection](../solutions/nfl-player-contact-detection/) | bronze, 0.71956 | Engine run | GPT-6-Luna | 12.33 | 2026-10-08 | `970d3ce25a44e2ecb2d7d111783e14b687b969cad9812b6cd5d56d30c534d6b5` | `92985c78e5d4c360316931b9d69d3ed5652ba48a8eb61495dc4b9b35b7d9deb6` |
| [hms-harmful-brain-activity-classification](../solutions/hms-harmful-brain-activity-classification/) | bronze, 0.36573 | Impulse ML pipeline | n/a | n/a | 2026-10-07 | `3c318cd03b1f51b31c7b218746875fb5d1a4d473a5d8812fe0ae6d019ceb433d` | `4f0f4c7fd57d175678dc93cc984cba75c4d6d27edaaf1c462cf6a25d7c767091` |

## Checking the files

Each graded submission is committed as `solutions/<competition>/submission.csv`. Check all eleven against the published hashes from the repository root:

```bash
shasum -a 256 -c reproduce/high15_submission_checksums.txt
```

The results verifier checks the High ledger, every graded-submission and grade-report hash, the High page tables and the public claims:

```bash
PYTHONDONTWRITEBYTECODE=1 python reproduce/verify_results.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v tests.test_high_results
```

`grade.json` in each solution directory is the `mlebench` grade report for that submission, with the grader host's file path omitted. To reproduce a grade, prepare the competition with `mlebench prepare -c <competition>` and run `mlebench grade-sample solutions/<competition>/submission.csv <competition>`.

## Capability and scope

Public external data, web research, pretrained models, and exploit discovery are intentional capabilities of Impulse's agent. The method column records the route used for each result. The Smartphone Decimeter result is an exploit-discovery result: the run reconstructed the target positions from logs shipped in the public data.

The claim covers the 15-competition High split. SIIM-FISABIO-RSNA COVID-19 Detection, RSNA 2022 Cervical Spine Fracture Detection, Vesuvius Challenge Ink Detection, and BMS Molecular Translation have no medal. This repository publishes hashes, scores, methods, graded submissions and notes; it doesn't contain private labels, credentials, raw benchmark datasets, or model blobs.
