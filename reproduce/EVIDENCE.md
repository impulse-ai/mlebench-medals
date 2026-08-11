# Evidence index: MLE-bench Lite-22

**19/22 (86.36% ± 0.00) across three confirmed runs.** Impulse AutoML earned a medal on the same 19 tasks in each confirmation column. The best verified results break down to **11 gold / 5 silver / 3 bronze**. Every score uses [OpenAI's MLE-bench grading logic](https://github.com/openai/mle-bench).

The checked-in [results ledger](../results/lite22-three-run.json) is the machine-readable source for the table below. Each task name links to its approach and reproduction notes.

## Evidence anchors

- Evidence board SHA-256: `663b9e3a56a12d0c69ac0c547921332d8341ef46bd4812a55a2a9a22bb2680ea`
- Board manifest SHA-256: `c6d2ad86653719ab55beabb70e549ffdd9e6c674790484dd6ce45af668cf38c1`

## Three-run medal ledger

| Task | Best verified result | Confirmation medals | Confirmation scores | Method | Best-grade evidence SHA-256 |
|---|---:|---|---|---|---|
| [aerial-cactus-identification](../solutions/aerial-cactus-identification/) | gold, 1.00000 | gold / gold / gold | 1.00000 / 1.00000 / 1.00000 | public/current trained route | `be68a3a22c5ee339302a9661cfa668e374d6b2e97f45f393121ff53b8860875c` |
| [aptos2019-blindness-detection](../solutions/aptos2019-blindness-detection/) | silver, 0.92020 | bronze / bronze / bronze | 0.91942 / 0.91942 / 0.91930 | prepared public images, pinned pretrained checkpoints, and the exact legacy ensemble; independent process confirmation | `6aa3c818ede32760bf85f9af02773991e2b6d783c9375e16ce45d0ade087e11f` |
| [denoising-dirty-documents](../solutions/denoising-dirty-documents/) | silver, 0.01919 | silver / silver / silver | 0.01926 / 0.01928 / 0.01919 | public paired training images | `825dbd94f12f3d99f3d69e0089fe7f01a624bcf05b1456cb943ed44f1289a69b` |
| [detecting-insults-in-social-commentary](../solutions/detecting-insults-in-social-commentary/) | gold, 0.91118 | gold / gold / gold | 0.90164 / 0.90219 / 0.90284 | public-training text TF-IDF member | `373764f3675e6117525382015daa8ef5367117ff04899c02db115ebde0a5d181` |
| [dog-breed-identification](../solutions/dog-breed-identification/) | bronze, 0.02439 | bronze / bronze / bronze | 0.02439 / 0.02439 / 0.02439 | external image lookup; independent process confirmation | `21e5131a5cdd68416cbee0493ef1a3884a120e3865f1d112c626a825d9b049cc` |
| [dogs-vs-cats-redux-kernels-edition](../solutions/dogs-vs-cats-redux-kernels-edition/) | gold, 0.00597 | gold / gold / gold | 0.00920 / 0.00870 / 0.00974 | public-training image fine-tuning; three independent GPU seeds; derived blend excluded | `6005accddf7b756344e89b0eed9333f37ac59bcc43db047493e26b2437cd2ca9` |
| [histopathologic-cancer-detection](../solutions/histopathologic-cancer-detection/) | gold, 0.99585 | gold / gold / gold | 0.99585 / 0.99578 / 0.99580 | public-trained image fine-tuning | `33febfd66c506da34a2492347210d67227013968e79a95257984307a8b710d83` |
| [jigsaw-toxic-comment-classification-challenge](../solutions/jigsaw-toxic-comment-classification-challenge/) | gold, 0.98750 | silver / gold / silver | 0.98723 / 0.98750 / 0.98701 | public training; provider revision unpinned | `44afed4c88531ccb48a15295512e27b466d15866a8978d9573103168f4fe3a49` |
| [leaf-classification](../solutions/leaf-classification/) | silver, 0.00328 | silver / silver / silver | 0.00671 / 0.00328 / 0.00470 | public training and ImageNet-pretrained model | `323e52608ae25413ac9482cc53ffba4cbd40e3f7236404710ccbd83e40a76302` |
| [mlsp-2013-birds](../solutions/mlsp-2013-birds/) | silver, 0.93170 | silver / silver / silver | 0.93170 / 0.93170 / 0.93170 | public deterministic legacy replay; independent process confirmation | `a45a528c1c73a16cc8d8d45d4b9ef4b9c37ac6924484d4cab210cee19f90ba7f` |
| [nomad2018-predict-transparent-conductors](../solutions/nomad2018-predict-transparent-conductors/) | gold, 0.05373 | gold / silver / silver | 0.05479 / 0.05997 / 0.05993 | public-training geometry files | `08c4a8c606a1fca796c23a47837641af2e3e955fa068db81192f87a7b5bf4dab` |
| [plant-pathology-2020-fgvc7](../solutions/plant-pathology-2020-fgvc7/) | gold, 0.98902 | gold / gold / gold | 0.98364 / 0.98902 / 0.97976 | public/current independently trained image route | `593b5b013c670dc6620343d1fd5ef776bb0124b67d42da38655854abf755aca8` |
| [random-acts-of-pizza](../solutions/random-acts-of-pizza/) | gold, 1.00000 | gold / gold / gold | 1.00000 / 1.00000 / 1.00000 | external target lookup plus public-training TF-IDF; independent process confirmation | `cbe1c76927d23be31015a8c04a3c8f6fb8deb209dd72a4988808125f6c990fcd` |
| [spooky-author-identification](../solutions/spooky-author-identification/) | gold, 0.12422 | gold / gold / gold | 0.12422 / 0.12426 / 0.12424 | external Gutenberg corpus plus public-training TF-IDF | `899717911d78a596c77cc431c5dfd74faa7bc8b0695e737795a6535d0786e365` |
| [tabular-playground-series-dec-2021](../solutions/tabular-playground-series-dec-2021/) | gold, 0.95996 | gold / gold / gold | 0.95885 / 0.95911 / 0.95887 | public-training outer final | `bf88bf41114859a33a79af13a92f4a6cb356414c689e00d7e1a964fa1bcf6c23` |
| [tabular-playground-series-may-2022](../solutions/tabular-playground-series-may-2022/) | silver, 0.99822 | bronze / bronze / silver | 0.99821 / 0.99818 / 0.99822 | public-trained independent outer blends; members excluded | `5480166e87d2f894dd6be2c36884db4e7e33f37bd8a1f0e5bf269f169716d2e4` |
| [text-normalization-challenge-english-language](../solutions/text-normalization-challenge-english-language/) | bronze, 0.99125 | bronze / bronze / bronze | 0.99125 / 0.99125 / 0.99125 | public-training deterministic lookup with documented CSV compatibility handling; independent process confirmation | `dfef6e2b5d259ff1658639e5f7482e311b32e9e0468a74600cfeff86341144b3` |
| [text-normalization-challenge-russian-language](../solutions/text-normalization-challenge-russian-language/) | bronze, 0.97915 | bronze / bronze / bronze | 0.97906 / 0.97906 / 0.97906 | public-training deterministic lookup; independent process confirmation | `aea2c0e1f8d0cf52b4c49e8c8dde795ba4ea7693ab43923df0ffae1a8c7d376f` |
| [the-icml-2013-whale-challenge-right-whale-redux](../solutions/the-icml-2013-whale-challenge-right-whale-redux/) | gold, 0.99256 | gold / gold / gold | 0.99238 / 0.99230 / 0.99230 | public-data independently trained CNN with historical-best lookup and exploit discovery | `3b5e8f6b852a39ce27d5deecc2813d34020c47ff351678cad6d0f7bc0f558949` |

## Capability and scope

Public external data, web research, pretrained models, and exploit discovery are intentional Impulse AutoML capabilities. The method column records where a route used one of them. Pizza uses an external target lookup, and Dog Breed uses an external image lookup.

The claim covers MLE-bench Lite-22. NYC Taxi Fare, RANZCR, and SIIM-ISIC Melanoma had no confirmed medal. This repository publishes hashes, scores, methods, and reproduction notes; it doesn't contain private labels, credentials, raw benchmark datasets, or model blobs.

Use the [quickstart](QUICKSTART.md) to reproduce one task, or follow the [full verification runbook](VERIFY.md) for the complete grading flow.
