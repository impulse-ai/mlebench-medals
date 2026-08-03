"""Full fine-tune image operator: the escalation tier above frozen embeddings.

`image_embedding.py` freezes a pretrained backbone and only fits a logistic
head on top -- fast, but it plateaus below medal bars on several image tasks
(aerial/aptos/dog-breed/histopathologic/ranzcr/siim). This operator fine-tunes
ALL layers of a torchvision backbone end to end, at real compute cost, as the
next rung on the escalation ladder. It reuses `image_embedding.load_image_task`
for id/label/path resolution (including zip-backed members) rather than
re-deriving it, and reuses the shared embedding-core helpers (zip
materialization, backbone weight lookup, classifier head swap, stratified
subsample, metric scoring, prediction-frame assembly) so none of that logic is
duplicated.

Validation is HONEST: a single stratified 20% holdout (not k-fold OOF, because
each epoch is an expensive training step) is used both to early-stop/select
the best epoch and as the exported validation artifact (`validation_kind=
"holdout"`), so the reported score is exactly the score of what's exported --
nothing here is train-in-sample.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from bench.ees_core.candidates import CandidateResult
from bench.ees_core.operators.embedding_core import (
    _oof_metric,
    assemble_prediction_frames,
    resolve_media_paths,
    scorer_for_metric,
    stratified_subsample,
    swap_classifier_head,
    torchvision_weight_for,
)
from bench.ees_core.operators.image_embedding import load_image_task
from bench.ees_core.prediction_artifacts import write_prediction_artifacts

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_normalized(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return (arr - _IMAGENET_MEAN) / _IMAGENET_STD


@dataclass(frozen=True)
class _TimePlan:
    """Wall-clock plan that guarantees the operator finishes BEFORE any outside kill.

    July-7 aerial gate post-mortem: the controller hard-kills the operator
    subprocess at EES_OPERATOR_BUDGET_SECONDS with `no_score_reason=
    "operator_timeout"` and ZERO artifacts -- a kill from outside never lets the
    graceful best-so-far path run. Two rules follow:

    - `wall_seconds`: never train past 85% of a positive controller budget,
      regardless of the configured max_seconds (the gate had max_seconds=10800
      against a 3600s controller budget and died with nothing to show). A
      controller budget <= 0 means "no outside kill" (execution.py runs the
      operator with no timeout), so only max_seconds applies then.
    - `train_deadline_seconds`: the tail AFTER the training loop (one full
      test-set forward pass, plus a possible epoch-end holdout eval) is not
      interruptible, so the training loop must stop early enough to leave room
      for it. The reserve is sized from the measured epoch-0 holdout eval rate
      scaled to the test-set size with a 1.5x safety factor -- large test sets
      (histopathologic: 57k, ranzcr: 35k) can need far more than a fixed slack.
    """

    wall_seconds: float
    tail_reserve_seconds: float
    train_deadline_seconds: float


def _compute_time_plan(
    *,
    max_seconds: float,
    controller_budget_seconds: float,
    epoch0_eval_seconds: float,
    n_holdout: int,
    n_test: int,
) -> _TimePlan:
    if controller_budget_seconds > 0:
        wall = min(max_seconds, 0.85 * controller_budget_seconds)
    else:
        wall = max_seconds
    per_image_seconds = epoch0_eval_seconds / max(1, n_holdout)
    # one safety-padded test pass + one epoch-end holdout eval + fixed margin
    tail_reserve = 1.5 * per_image_seconds * n_test + epoch0_eval_seconds + 5.0
    train_deadline = max(0.0, wall - tail_reserve)
    return _TimePlan(
        wall_seconds=wall,
        tail_reserve_seconds=tail_reserve,
        train_deadline_seconds=train_deadline,
    )


def _finetune_use_vertex() -> bool:
    """Route the operator to the Vertex GPU lane instead of in-process CPU.

    Requires BOTH the escalation opt-in (``EES_ENABLE_FINETUNE=1``, which the
    controller already gates on) AND the explicit GPU-offload flag
    (``EES_FINETUNE_USE_VERTEX=1``). Default is the in-process path, which stays
    the fallback whenever the offload flag is unset.
    """
    return (
        os.environ.get("EES_FINETUNE_USE_VERTEX") == "1"
        and os.environ.get("EES_ENABLE_FINETUNE") == "1"
    )


def _build_finetune_job_code(
    *,
    backbone: str,
    max_epochs: int,
    max_seconds: float,
    max_train_rows: int,
    image_size: int,
    task_id: str | None,
    metric: str | None,
    random_state: int,
) -> str:
    """In-container training script: run the SAME operator on the staged data.

    The GPU container runs this operator in-process (CPU/GPU inside the box) with
    the offload flag cleared (no recursion) and the controller-kill budget
    disabled (the container's own wall budget governs). It reads the extracted
    data from cwd and writes artifacts to cwd; the launcher's upload tail ships
    the NEW files (submission.csv, validation_*.csv, metrics.json) back to the
    unique GCS output path, which the seam pulls into the operator's output_dir.
    """
    # Thread the GPU-lane throughput knobs into the container (the container env
    # only carries what submit_vertex_job sets, so bake the host values in).
    passthru = {
        k: os.environ[k]
        for k in (
            "EES_FINETUNE_NUM_WORKERS",
            "EES_FINETUNE_BATCH_SIZE",
            "EES_FINETUNE_EVAL_BATCH_SIZE",
        )
        if os.environ.get(k)
    }
    env_lines = "".join(
        f"os.environ[{k!r}] = {v!r}\n" for k, v in passthru.items()
    )
    # The base gpu-sandbox image ships CUDA torch but NOT torchvision; install a
    # torch-version-matched torchvision (from the cu121 wheel index) before the
    # operator imports it, else _build_finetune_model raises ModuleNotFoundError.
    torchvision_bootstrap = (
        "import subprocess as _sp, sys as _sys, importlib as _il\n"
        "try:\n"
        "    import torchvision  # noqa\n"
        "except Exception:\n"
        "    import torch as _t\n"
        "    _tvmap = {'2.1':'0.16.2','2.2':'0.17.2','2.3':'0.18.1','2.4':'0.19.1',"
        "'2.5':'0.20.1','2.6':'0.21.0','2.7':'0.22.0','2.8':'0.23.0'}\n"
        "    _mm = '.'.join(_t.__version__.split('+')[0].split('.')[:2])\n"
        "    _ver = _tvmap.get(_mm)\n"
        "    _pkg = ('torchvision==' + _ver) if _ver else 'torchvision'\n"
        "    print('[finetune-bootstrap] installing', _pkg, 'for torch', _t.__version__, flush=True)\n"
        "    _sp.run([_sys.executable, '-m', 'pip', 'install', '--quiet',\n"
        "             '--extra-index-url', 'https://download.pytorch.org/whl/cu121', _pkg], check=True)\n"
        "    _il.invalidate_caches()\n"
    )
    return (
        "import os\n"
        "os.environ.pop('EES_FINETUNE_USE_VERTEX', None)\n"
        "os.environ['EES_OPERATOR_BUDGET_SECONDS'] = '0'\n"
        + env_lines
        + torchvision_bootstrap
        + "from pathlib import Path\n"
        "from bench.ees_core.operators.image_finetune import run_image_finetune_operator\n"
        "run_image_finetune_operator(\n"
        "    Path('.'), Path('.'),\n"
        f"    backbone={backbone!r},\n"
        f"    max_epochs={int(max_epochs)},\n"
        f"    max_seconds={float(max_seconds)},\n"
        f"    max_train_rows={int(max_train_rows)},\n"
        f"    image_size={int(image_size)},\n"
        f"    task_id={task_id!r},\n"
        f"    metric={metric!r},\n"
        f"    random_state={int(random_state)},\n"
        ")\n"
    )


def _run_via_vertex(
    data_dir: Path,
    output_dir: Path,
    *,
    backbone: str,
    max_epochs: int,
    max_seconds: float,
    max_train_rows: int,
    image_size: int,
    task_id: str | None,
    metric: str | None,
    random_state: int,
    started: float,
    seam=None,
) -> CandidateResult:
    """Offload the fine-tune to a Vertex GPU job and rebuild the CandidateResult.

    Emits the SAME OOF/holdout artifact contract as the in-process path (the
    in-container run wrote them; the seam pulled them into ``output_dir``), so
    the registry blends deep + classical candidates honestly.
    """
    if seam is None:
        from bench.ees_core import vertex_gpu_execution as seam
    job_code = _build_finetune_job_code(
        backbone=backbone,
        max_epochs=max_epochs,
        max_seconds=max_seconds,
        max_train_rows=max_train_rows,
        image_size=image_size,
        task_id=task_id,
        metric=metric,
        random_state=random_state,
    )
    job = seam.run_gpu_training_job(
        job_code,
        data_dir,
        task_id=task_id or "image_finetune",
        output_dir=output_dir,
        timeout_min=max(1, int(round(max_seconds / 60.0))),
    )
    if not job.success:
        return CandidateResult(
            candidate_id="image_finetune",
            recipe_id="image_finetune",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source=f"ees_core:image_finetune:vertex:{backbone}",
            no_score_reason=f"vertex_gpu:{job.state}:{job.error}",
        )

    metrics_path = Path(output_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["execution_lane"] = "vertex_gpu"
    metrics["vertex_job_id"] = job.job_id
    metrics["vertex_run_ts"] = job.run_ts
    metrics["seconds"] = round(time.time() - started, 3)
    metrics["runtime_seconds"] = metrics["seconds"]

    # Repoint the artifact paths at the pulled local files (the metrics.json the
    # container wrote carries container-relative paths).
    artifacts = metrics.get("prediction_artifacts")
    if isinstance(artifacts, dict):
        for key, fname in (
            ("test_prediction_path", "test_predictions.csv"),
            ("validation_prediction_path", "validation_predictions.csv"),
            ("validation_target_path", "validation_targets.csv"),
        ):
            local = Path(output_dir) / fname
            if local.exists():
                artifacts[key] = str(local)

    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    score = validation.get("score")
    direction = validation.get("direction")
    submission_path = job.submission_path or (Path(output_dir) / "submission.csv")
    return CandidateResult(
        candidate_id="image_finetune",
        recipe_id="image_finetune",
        success=True,
        submission_path=submission_path,
        validation={"execution_lane": "vertex_gpu"},
        score=score,
        medal="none",
        final_artifact_source=f"ees_core:image_finetune:vertex:{backbone}",
        artifact_paths=job.artifact_paths or [str(submission_path)],
        metrics=metrics,
        score_direction=direction if direction in ("minimize", "maximize") else None,
    )


def _build_finetune_model(backbone: str, n_classes: int):
    """Load a pretrained torchvision backbone and replace its head for n_classes.

    Tries pretrained ImageNet weights first (matching `torchvision_backbones`'
    weight choice per name so embedding and fine-tune stay on the same
    pretraining); if the weight download is blocked (no network / offline
    sandbox), falls back to random init (`weights=None`) rather than failing
    the whole operator. Returns (model, pretrained: bool).
    """
    import torchvision as tv
    from torch import nn

    pretrained = True
    try:
        weight = torchvision_weight_for(tv, backbone)
        model = getattr(tv.models, backbone)(weights=weight)
    except Exception:
        pretrained = False
        model = getattr(tv.models, backbone)(weights=None)
    swap_classifier_head(model, backbone, lambda in_features: nn.Linear(in_features, n_classes))
    return model, pretrained


class _ImageIdDataset:
    """Lazily decodes+normalizes images by path; optional horizontal-flip augmentation."""

    def __init__(self, paths: list[Path], labels: "np.ndarray | None", image_size: int, augment: bool):
        self.paths = paths
        self.labels = labels
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        arr = _load_normalized(self.paths[index], self.image_size)
        if self.augment and random.random() < 0.5:
            arr = arr[:, ::-1, :].copy()
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))
        if self.labels is None:
            return tensor
        return tensor, int(self.labels[index])


def _evaluate(model, loader, n_classes: int) -> np.ndarray:
    import torch

    model.eval()
    # DataLoader batches always come off the loader on cpu; the model may live on
    # cuda. Move every batch to the model's own device (the first real GPU run
    # died with FloatTensor-vs-cuda.FloatTensor here on the epoch-0 baseline pass).
    device = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device)
            logits = model(xb)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, n_classes), dtype=float)


def _derive_ordinal_task_meta(data_dir: Path) -> dict:
    """Read id/label columns, image dirs, extension and bin count from the data.

    Pure pandas/filesystem -- no torch -- so the diverse orchestrator can derive
    the container-trainer parameters generically (no task-specific constants).
    """
    import pandas as pd

    data_dir = Path(data_dir)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    id_col = sample.columns[0]
    target_col = sample.columns[1]
    train = pd.read_csv(data_dir / "train.csv")
    label_col = target_col if target_col in train.columns else train.columns[1]
    n_bins = int(pd.to_numeric(train[label_col], errors="coerce").dropna().astype(int).max()) + 1

    def _find_images_dir(*names: str) -> tuple[str, str]:
        for name in names:
            d = data_dir / name
            if d.is_dir():
                sample_file = next((p for p in d.iterdir() if p.is_file()), None)
                ext = sample_file.suffix if sample_file else ".png"
                return name, ext
        return names[0], ".png"

    train_dir, ext = _find_images_dir("train_images", "train", "images")
    test_dir, _ = _find_images_dir("test_images", "test", "images")
    return {
        "id_col": id_col,
        "target_col": target_col,
        "label_col": label_col,
        "n_bins": n_bins,
        "train_images_dir": train_dir,
        "test_images_dir": test_dir,
        "image_ext": ext,
    }


def run_diverse_ensemble(
    data_dir: Path,
    output_dir: Path,
    *,
    task_id: str | None,
    metric: str | None,
    max_seconds: float,
    seam=None,
    storage_client=None,
) -> CandidateResult:
    """Diverse multi-backbone ordinal ensemble (the proven medal-gap path).

    Fires N decorrelated configs (backbone family x resolution x seed) from the
    general default set (env-overridable), trains each k-fold on the Vertex GPU
    seam (or pulls a cached ``reuse_prefix``), then OOF-weighted-blends them and
    fits ONE pooled OptimizedRounder. Emits a single candidate with OOF
    artifacts. Selection is OOF-only; the test set is never consulted.
    """
    import pandas as pd

    from bench.ees_core.operators import image_finetune_ensemble as ens

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        meta = _derive_ordinal_task_meta(data_dir)
        configs = ens.resolve_ensemble_configs()
        n_bins = meta["n_bins"]

        # Resolve the seam + storage client ONCE up front so the per-config
        # workers below are thread-safe (no shared lazy-import races).
        if any(not c.get("reuse_prefix") for c in configs) and seam is None:
            from bench.ees_core import vertex_gpu_execution as seam
        if any(c.get("reuse_prefix") for c in configs) and storage_client is None:
            from bench.ees_core import vertex_gpu_execution as _seam_mod
            storage_client = _seam_mod._load_storage_client()

        def _process_config(cfg: dict) -> tuple[ens.ConfigPreds | None, dict]:
            # A single config's GPU job can be spot-preempted or time out; that
            # must NOT sink the whole ensemble. On failure we return (None, entry
            # with error) and the caller aggregates the surviving configs — a
            # single strong config still medals.
            nfolds = int(cfg.get("nfolds", 5))
            reuse_prefix = cfg.get("reuse_prefix")
            try:
                if reuse_prefix:
                    folds = ens.pull_reuse_fold_preds(reuse_prefix, nfolds, storage_client=storage_client)
                    entry = {"name": cfg["name"], "source": "reuse", "reuse_prefix": reuse_prefix}
                else:
                    job_code = ens.build_config_job_code(
                        cfg,
                        task_id=task_id,
                        id_col=meta["id_col"],
                        label_col=meta["label_col"],
                        train_images_dir=meta["train_images_dir"],
                        test_images_dir=meta["test_images_dir"],
                        image_ext=meta["image_ext"],
                        n_bins=n_bins,
                        max_seconds=max_seconds,
                    )
                    cfg_dir = output_dir / cfg["name"]
                    job = seam.run_gpu_training_job(
                        job_code, Path(data_dir),
                        task_id=f"{task_id or 'image_finetune'}:{cfg['name']}",
                        output_dir=cfg_dir,
                        timeout_min=max(1, int(round(max_seconds / 60.0))),
                    )
                    if not job.success:
                        return None, {"name": cfg["name"], "source": "vertex",
                                      "job_id": job.job_id, "failed": f"{job.state}:{job.error}"}
                    folds = ens.load_fold_arrays_from_dir(cfg_dir, nfolds)
                    entry = {"name": cfg["name"], "source": "vertex", "job_id": job.job_id}
                return ens.config_preds_from_folds(cfg["name"], folds), entry
            except Exception as exc:  # noqa: BLE001 — isolate one config's failure
                return None, {"name": cfg["name"], "failed": f"{type(exc).__name__}: {exc}"}

        # Run each config concurrently: the per-config Vertex GPU jobs are
        # independent, so submitting them together turns N sequential ~90-min
        # jobs into one ~90-min wall (they train on separate T4s in parallel).
        # Sequential execution was the cause of the aptos operator_timeout —
        # config 2 never got to run before the operator budget expired.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, len(configs))) as pool:
            results = list(pool.map(_process_config, configs))
        config_preds = [cp for cp, _ in results if cp is not None]
        exec_log = [entry for _, entry in results]
        failed = [e for cp, e in results if cp is None]
        if not config_preds:
            raise RuntimeError(f"all {len(configs)} ensemble configs failed: {failed}")
        # Aggregate the surviving configs. If some were preempted/timed out the
        # blend just has fewer members (a single config → its own pooled rounder);
        # one preemption no longer forfeits the medal.

        agg = ens.aggregate_ordinal_ensemble(
            config_preds, id_col=meta["id_col"], target_col=meta["target_col"], n_bins=n_bins,
        )

        id_col, target_col = meta["id_col"], meta["target_col"]
        submission = ens.align_submission_to_sample(
            agg.submission, data_dir / "sample_submission.csv", id_col, target_col
        )
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)

        oof_round = ens.OptimizedRounder(n_bins=n_bins)
        oof_round.coef_ = np.asarray(agg.rounder_coef, dtype=float)
        val_pred = pd.DataFrame({id_col: agg.common_oof_ids, target_col: oof_round.predict(agg.oof_ensemble_cont).astype(int)})
        val_targets = pd.DataFrame({id_col: agg.common_oof_ids, target_col: agg.oof_labels.astype(int)})
        test_pred = submission[[id_col, target_col]].copy()
        val_pred = val_pred.sort_values(id_col, kind="stable").reset_index(drop=True)
        val_targets = val_targets.sort_values(id_col, kind="stable").reset_index(drop=True)
        artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=test_pred,
            validation_predictions=val_pred,
            validation_targets=val_targets,
            id_col=id_col,
            target_cols=[target_col],
            validation_kind="oof",
        )

        metrics = {
            "operator": "image_finetune",
            "mode": "diverse_ensemble",
            "task_id": task_id,
            "configs": [c["name"] for c in configs],
            "config_execution": exec_log,
            "weights": agg.weights,
            "chosen_blend": agg.chosen,
            "equal_mean_oof_qwk": agg.equal_mean_oof_qwk,
            "weighted_oof_qwk": agg.weighted_oof_qwk,
            "ensemble_oof_qwk": agg.ensemble_oof_qwk,
            "config_self_oof_qwk": agg.config_self_oof_qwk,
            "oof_corr": agg.oof_corr,
            "rounder_coef": agg.rounder_coef,
            "n_bins": n_bins,
            "n_oof": len(agg.common_oof_ids),
            "n_test": len(agg.common_test_ids),
            "prediction_artifacts": artifacts,
            "validation": {
                "validation_kind": "oof",
                "score": agg.ensemble_oof_qwk,
                "objective": "qwk",
                "direction": "maximize",
            },
            "seconds": round(time.time() - started, 3),
            "runtime_seconds": round(time.time() - started, 3),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="image_finetune",
            recipe_id="image_finetune_diverse_ensemble",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=agg.ensemble_oof_qwk,
            medal="none",
            final_artifact_source="ees_core:image_finetune:diverse_ensemble",
            artifact_paths=[str(submission_path), str(output_dir / "metrics.json")],
            metrics=metrics,
            score_direction="maximize",
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="image_finetune",
            recipe_id="image_finetune_diverse_ensemble",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:image_finetune:diverse_ensemble",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )


def run_image_finetune_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    backbone: str = "resnet18",
    max_epochs: int = int(os.environ.get("EES_FINETUNE_MAX_EPOCHS", 8)),
    max_seconds: float = float(os.environ.get("EES_FINETUNE_MAX_SECONDS", 3600)),
    max_train_rows: int = int(os.environ.get("EES_FINETUNE_MAX_TRAIN_ROWS", 30000)),
    image_size: int = int(os.environ.get("EES_FINETUNE_IMAGE_SIZE", 160)),
    task_id: str | None = None,
    metric: str | None = None,
    random_state: int = 101,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Diverse-ensemble escalation: on an ordinal (QWK/kappa) task a single
    # backbone underfits the train->test shift; the proven fix is an OOF-weighted
    # blend of DECORRELATED backbone/resolution configs with a pooled ordinal
    # rounder. Gated (EES_FINETUNE_DIVERSE_ENSEMBLE=1|auto + ordinal metric);
    # single-model stays the default below.
    from bench.ees_core.operators.image_finetune_ensemble import diverse_ensemble_enabled
    if diverse_ensemble_enabled(metric):
        return run_diverse_ensemble(
            data_dir, output_dir,
            task_id=task_id, metric=metric, max_seconds=max_seconds,
        )
    # GPU offload lane: when opted in (EES_ENABLE_FINETUNE=1 + EES_FINETUNE_USE_VERTEX=1)
    # train on a real Vertex GPU through the seam instead of the in-process CPU
    # host, emitting the same OOF/holdout artifacts. In-process stays the default
    # fallback below.
    if _finetune_use_vertex():
        return _run_via_vertex(
            data_dir,
            output_dir,
            backbone=backbone,
            max_epochs=max_epochs,
            max_seconds=max_seconds,
            max_train_rows=max_train_rows,
            image_size=image_size,
            task_id=task_id,
            metric=metric,
            random_state=random_state,
            started=started,
        )
    # See _TimePlan: self-cap under the controller's outside-kill budget and
    # reserve tail time for the untimed post-training test pass. The plan is
    # computed after the epoch-0 holdout eval (which measures the per-image
    # eval rate the tail reserve is sized from).
    controller_budget_seconds = float(os.environ.get("EES_OPERATOR_BUDGET_SECONDS", "3600"))
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        task = load_image_task(data_dir)
        n_classes = len(task.classes)
        if n_classes < 2:
            raise ValueError("image_finetune requires >= 2 classes")

        train_ids, y = task.train_ids, task.y
        if len(train_ids) > max_train_rows:
            keep = stratified_subsample(y, max_train_rows, np.random.default_rng(random_state))
            train_ids = [train_ids[i] for i in keep]
            y = y[keep]

        labels = list(range(n_classes))
        scorer = scorer_for_metric(metric) if metric else ("auc" if n_classes == 2 else "logloss")
        maximize = scorer in ("auc", "qwk")

        idx = np.arange(len(train_ids))
        fit_idx, holdout_idx = train_test_split(idx, test_size=0.2, random_state=random_state, stratify=y)

        all_train_paths = resolve_media_paths(train_ids, task.train_paths)
        fit_paths = [all_train_paths[i] for i in fit_idx]
        fit_y = y[fit_idx]
        holdout_paths = [all_train_paths[i] for i in holdout_idx]
        holdout_y = y[holdout_idx]
        holdout_ids = [train_ids[i] for i in holdout_idx]
        test_paths = resolve_media_paths(task.test_ids, task.test_paths)

        # GPU-lane throughput knobs (env-driven; defaults preserve the original
        # in-process behavior). num_workers>0 parallelizes PIL decode -- the
        # dominant cost on large image sets (histo test = 45k) that otherwise
        # starves the GPU and blows the container hard-kill budget.
        cfg_batch = int(os.environ.get("EES_FINETUNE_BATCH_SIZE", "32"))
        cfg_eval_batch = int(os.environ.get("EES_FINETUNE_EVAL_BATCH_SIZE", "64"))
        num_workers = int(os.environ.get("EES_FINETUNE_NUM_WORKERS", "0"))
        pin_memory = torch.cuda.is_available()
        persistent = num_workers > 0
        batch_size = min(cfg_batch, max(1, len(fit_paths)))
        generator = torch.Generator()
        generator.manual_seed(random_state)
        fit_loader = DataLoader(
            _ImageIdDataset(fit_paths, fit_y, image_size, augment=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            generator=generator,
        )
        eval_batch_size = cfg_eval_batch
        holdout_loader = DataLoader(
            _ImageIdDataset(holdout_paths, None, image_size, augment=False),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        test_loader = DataLoader(
            _ImageIdDataset(test_paths, None, image_size, augment=False),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model, pretrained = _build_finetune_model(backbone, n_classes)
        model.to(device)

        def evaluate_holdout() -> np.ndarray:
            return _evaluate(model, holdout_loader, n_classes)

        # Baseline (epoch 0) checkpoint, evaluated BEFORE any training step, so a
        # max_seconds wall breached before/inside the very first epoch still has a
        # "best-so-far" checkpoint to fall back to (never an unset/crashing state).
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        epoch0_eval_started = time.time()
        best_holdout_proba = evaluate_holdout()
        epoch0_eval_seconds = time.time() - epoch0_eval_started
        best_score = _oof_metric(scorer, holdout_y, best_holdout_proba, n_classes, labels)

        time_plan = _compute_time_plan(
            max_seconds=max_seconds,
            controller_budget_seconds=controller_budget_seconds,
            epoch0_eval_seconds=epoch0_eval_seconds,
            n_holdout=len(holdout_paths),
            n_test=len(test_paths),
        )
        max_seconds_effective = time_plan.wall_seconds

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))
        criterion = nn.CrossEntropyLoss()

        epochs_run = 0
        timed_out = time.time() - started >= time_plan.train_deadline_seconds
        for epoch in range(1, max_epochs + 1):
            if timed_out:
                break
            model.train()
            for xb, yb in fit_loader:
                if time.time() - started >= time_plan.train_deadline_seconds:
                    timed_out = True
                    break
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            if timed_out:
                break
            scheduler.step()
            epochs_run = epoch
            holdout_proba = evaluate_holdout()
            score = _oof_metric(scorer, holdout_y, holdout_proba, n_classes, labels)
            improved = score > best_score if maximize else score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_holdout_proba = holdout_proba

        model.load_state_dict(best_state)
        test_proba = _evaluate(model, test_loader, n_classes)

        val_pred, val_targets = assemble_prediction_frames(
            holdout_ids, best_holdout_proba, holdout_y,
            id_col=task.id_col, classes=task.classes, target_cols=task.target_cols,
        )
        test_pred, _ = assemble_prediction_frames(
            task.test_ids, test_proba, None,
            id_col=task.id_col, classes=task.classes, target_cols=task.target_cols,
        )
        # native-id sort so cross-operator OOF/holdout records align (registry
        # uses ordered id equality) -- same convention as image_embedding/audio_embedding.
        val_pred = val_pred.sort_values(task.id_col, kind="stable").reset_index(drop=True)
        val_targets = val_targets.sort_values(task.id_col, kind="stable").reset_index(drop=True)

        submission_path = output_dir / "submission.csv"
        test_pred.to_csv(submission_path, index=False)
        artifacts = write_prediction_artifacts(
            output_dir,
            test_predictions=test_pred,
            validation_predictions=val_pred,
            validation_targets=val_targets,
            id_col=task.id_col,
            target_cols=task.target_cols,
            validation_kind="holdout",
        )
        metrics = {
            "operator": "image_finetune",
            "task_id": task_id,
            "backbone": backbone,
            "pretrained": pretrained,
            "device": device.type,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "max_epochs": max_epochs,
            "max_seconds": max_seconds,
            "max_seconds_effective": max_seconds_effective,
            "controller_budget_seconds": controller_budget_seconds,
            "tail_reserve_seconds": round(time_plan.tail_reserve_seconds, 3),
            "train_deadline_seconds": round(time_plan.train_deadline_seconds, 3),
            "epoch0_eval_seconds": round(epoch0_eval_seconds, 3),
            "timed_out": timed_out,
            "max_train_rows": max_train_rows,
            "image_size": image_size,
            "scorer": scorer,
            "n_classes": n_classes,
            "fit_train_rows": int(len(fit_paths)),
            "holdout_rows": int(len(holdout_paths)),
            "prediction_artifacts": artifacts,
            "validation": {
                "validation_kind": "holdout",
                "score": best_score,
                "objective": scorer,
                "direction": "maximize" if maximize else "minimize",
            },
            "seconds": round(time.time() - started, 3),
            "runtime_seconds": round(time.time() - started, 3),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CandidateResult(
            candidate_id="image_finetune",
            recipe_id="image_finetune",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(test_pred)), "columns": list(test_pred.columns)},
            score=best_score,
            medal="none",
            final_artifact_source=f"ees_core:image_finetune:{backbone}",
            artifact_paths=[str(submission_path), str(output_dir / "metrics.json")],
            metrics=metrics,
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="image_finetune",
            recipe_id="image_finetune",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:image_finetune",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )
