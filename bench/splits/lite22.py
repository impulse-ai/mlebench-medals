"""Official MLE-bench Lite-22 task list.

The installed ``mlebench`` CLI in this environment does not expose the
``list-comps --split lite`` command that the original harness expected. The
task list below mirrors the official low split from ``bench/splits/all.yaml``
with the tutorial-only ``spaceship-titanic`` entry removed.
"""

from __future__ import annotations


_LITE22_TASKS = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]


def get_lite22_tasks() -> list[str]:
    """Return the official Lite-22 tasks in deterministic run order."""
    return list(_LITE22_TASKS)
