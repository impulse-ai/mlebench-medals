"""Zip-backed image operator for Aerial Cactus Identification."""
from __future__ import annotations

import json
import os
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from bench.ees_core.candidates import CandidateResult
from bench.ees_core.prediction_artifacts import write_prediction_artifacts


def _zip_member_map(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {
            Path(name).name: name
            for name in zf.namelist()
            if not name.endswith("/")
        }


def _load_zip_images(zip_path: Path, ids: list[str], *, size: int = 32) -> np.ndarray:
    arr = np.empty((len(ids), 3, size, size), dtype=np.float32)
    with zipfile.ZipFile(zip_path) as zf:
        members = {
            Path(name).name: name
            for name in zf.namelist()
            if not name.endswith("/")
        }
        for index, image_id in enumerate(ids):
            member = members.get(image_id)
            if member is None:
                raise FileNotFoundError(f"{image_id} not found in {zip_path}")
            with zf.open(member) as handle:
                image = Image.open(handle).convert("RGB").resize((size, size))
                arr[index] = np.transpose(np.asarray(image, dtype=np.float32) / 255.0, (2, 0, 1))
    return arr


def _holdout_validation_frames(
    train: pd.DataFrame,
    valid_idx: np.ndarray,
    ensemble_probs: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """Assemble holdout validation artifacts from the SAME train rows.

    One frame convention: predictions and targets are sliced from the same
    frame in the same order with native ids, so their rows are aligned by
    construction. Returns the ensemble ROC AUC on that holdout (None when the
    holdout is single-class).
    """
    rows = train.iloc[np.asarray(valid_idx)]
    validation_predictions = pd.DataFrame(
        {
            "id": rows["id"].to_numpy(),
            "has_cactus": np.clip(np.asarray(ensemble_probs, dtype=float), 0, 1),
        }
    )
    validation_targets = rows[["id", "has_cactus"]].reset_index(drop=True)
    if rows["has_cactus"].nunique() > 1:
        auc = float(roc_auc_score(validation_targets["has_cactus"].to_numpy(), validation_predictions["has_cactus"].to_numpy()))
    else:
        auc = None
    return validation_predictions, validation_targets, auc


def _choose_torch_device(torch_module):
    if (
        platform.system() != "Darwin"
        or os.environ.get("EES_ENABLE_MPS") == "1"
    ) and hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps"), "mps"
    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), "cuda"
    return torch_module.device("cpu"), "cpu"


def run_aerial_image_operator(
    data_dir: Path,
    output_dir: Path,
    *,
    random_state: int = 42,
    max_epochs: int = 40,
    portfolio_seeds: tuple[int, ...] | None = None,
) -> CandidateResult:
    started = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = pd.read_csv(data_dir / "train.csv")
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        train_zip = data_dir / "train.zip"
        test_zip = data_dir / "test.zip"
        if not train_zip.exists() or not test_zip.exists():
            raise FileNotFoundError("train.zip and test.zip are required")

        if len(train) < 20:
            positive_rate = float(train["has_cactus"].mean())
            test_probs = np.full(len(sample), positive_rate, dtype=float)
            # No fitted model to validate; the constant is computed on train
            # itself, so these artifacts stay on the untrusted audit kind.
            validation_predictions = pd.DataFrame(
                {"id": train["id"], "has_cactus": np.full(len(train), positive_rate, dtype=float)}
            )
            validation_targets = train[["id", "has_cactus"]].copy()
            validation_kind = "train_mean_fallback_audit"
            metrics = {
                "operator": "image_aerial",
                "metric": "roc_auc",
                "validation_roc_auc": None,
                "fallback": "mean_label_small_data",
                "torch_backend": None,
                "train_rows": int(len(train)),
                "test_rows": int(len(sample)),
                "runtime_seconds": round(time.time() - started, 3),
            }
        else:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            seeds = portfolio_seeds or (7,)
            device, torch_backend = _choose_torch_device(torch)
            torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

            x = _load_zip_images(train_zip, train["id"].astype(str).tolist())
            y = train["has_cactus"].to_numpy(dtype=np.float32)

            class SmallCNN(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(3, 24, 3, padding=1),
                        nn.BatchNorm2d(24),
                        nn.ReLU(),
                        nn.Conv2d(24, 24, 3, padding=1),
                        nn.BatchNorm2d(24),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(24, 48, 3, padding=1),
                        nn.BatchNorm2d(48),
                        nn.ReLU(),
                        nn.Conv2d(48, 48, 3, padding=1),
                        nn.BatchNorm2d(48),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(48, 96, 3, padding=1),
                        nn.BatchNorm2d(96),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Dropout(0.15),
                        nn.Linear(96, 1),
                    )

                def forward(self, batch):
                    return self.net(batch)

            criterion = nn.BCEWithLogitsLoss()

            def _augment(batch):
                if torch.rand((), device=batch.device) < 0.5:
                    batch = torch.flip(batch, dims=[3])
                if torch.rand((), device=batch.device) < 0.5:
                    batch = torch.flip(batch, dims=[2])
                return batch

            def _predict(model, array: np.ndarray) -> np.ndarray:
                model.eval()
                chunks = []
                for start in range(0, len(array), 512):
                    xb = torch.from_numpy(array[start : start + 512]).to(device)
                    variants = [
                        xb,
                        torch.flip(xb, dims=[3]),
                        torch.flip(xb, dims=[2]),
                        torch.flip(xb, dims=[2, 3]),
                    ]
                    with torch.no_grad():
                        probs = [
                            torch.sigmoid(model(variant)).squeeze(1).detach().cpu().numpy()
                            for variant in variants
                        ]
                    chunks.append(np.mean(probs, axis=0))
                return np.concatenate(chunks)

            # One SHARED stratified holdout across all seeds. With per-seed splits,
            # every "validation" row was in-sample for some ensemble member, so the
            # exported ensemble artifacts were train_in_sample_audit (untrusted).
            # True OOF is impractical for this operator: each member is a torch CNN
            # early-stopped per epoch on a live validation set, so K-fold OOF would
            # multiply the training cost by K and need per-fold epoch selection.
            # A shared honest holdout (trusted kind) is the minimal upgrade; seeds
            # still differ in init and batch order, preserving portfolio diversity.
            train_idx, valid_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=random_state,
                stratify=y,
            )
            portfolio = []
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                generator = torch.Generator()
                generator.manual_seed(seed)
                loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(x[train_idx]),
                        torch.from_numpy(y[train_idx]).unsqueeze(1),
                    ),
                    batch_size=256,
                    shuffle=True,
                    num_workers=0,
                    generator=generator,
                )
                model = SmallCNN().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                history = []
                best_auc = -1.0
                best_state = None
                best_epoch = 1
                valid_y = y[valid_idx]
                for epoch in range(1, max_epochs + 1):
                    model.train()
                    losses = []
                    for xb, yb in loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(model(_augment(xb)), yb)
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    probs = _predict(model, x[valid_idx])
                    auc = float(roc_auc_score(valid_y, probs))
                    rec = {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "validation_roc_auc": auc,
                        "validation_accuracy_at_0_5": float(accuracy_score(valid_y, probs >= 0.5)),
                        "validation_log_loss": float(log_loss(valid_y, np.clip(probs, 1e-6, 1 - 1e-6))),
                    }
                    history.append(rec)
                    if auc > best_auc:
                        best_auc = auc
                        best_epoch = epoch
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if best_state is not None:
                    model.load_state_dict(best_state)
                portfolio.append(
                    {
                        "candidate_id": f"aerial_cnn_seed_{seed}",
                        "seed": int(seed),
                        "model": model,
                        "validation_roc_auc": best_auc,
                        "best_epoch": best_epoch,
                        "history": history,
                    }
                )

            portfolio.sort(key=lambda item: item["validation_roc_auc"], reverse=True)
            test_x = _load_zip_images(test_zip, sample["id"].astype(str).tolist())
            top_k = portfolio[: min(3, len(portfolio))]
            test_probs = np.mean([_predict(item["model"], test_x) for item in top_k], axis=0)
            # Honest ensemble evidence: the exported artifacts are the top-k mean
            # on the SHARED holdout, and the candidate score is the AUC of exactly
            # those artifacts (canonically reproducible).
            ensemble_valid_probs = np.mean([_predict(item["model"], x[valid_idx]) for item in top_k], axis=0)
            validation_predictions, validation_targets, ensemble_auc = _holdout_validation_frames(
                train, valid_idx, ensemble_valid_probs
            )
            validation_kind = "holdout"
            metrics = {
                "operator": "image_aerial",
                "metric": "roc_auc",
                "validation_roc_auc": ensemble_auc,
                "holdout_rows": int(len(valid_idx)),
                "top_k_mean_seed_roc_auc": float(np.mean([item["validation_roc_auc"] for item in top_k])),
                "selected_candidate_id": top_k[0]["candidate_id"],
                "prediction_strategy": "validation_portfolio_tta",
                "torch_backend": torch_backend,
                "portfolio_candidates": [
                    {
                        "candidate_id": item["candidate_id"],
                        "seed": item["seed"],
                        "validation_roc_auc": item["validation_roc_auc"],
                        "best_epoch": item["best_epoch"],
                        "history": item["history"],
                    }
                    for item in portfolio
                ],
                "top_k": len(top_k),
                "train_rows": int(len(train)),
                "test_rows": int(len(sample)),
                "runtime_seconds": round(time.time() - started, 3),
            }

        submission = pd.DataFrame(
            {
                "id": sample["id"],
                "has_cactus": np.clip(test_probs, 0, 1),
            }
        )
        submission_path = output_dir / "submission.csv"
        metrics_path = output_dir / "metrics.json"
        submission.to_csv(submission_path, index=False)
        metrics["prediction_artifacts"] = write_prediction_artifacts(
            output_dir,
            test_predictions=submission,
            validation_predictions=validation_predictions,
            validation_targets=validation_targets,
            id_col="id",
            target_cols=["has_cactus"],
            validation_kind=validation_kind,
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return CandidateResult(
            candidate_id="aerial_image",
            recipe_id="metadata_audit",
            success=True,
            submission_path=submission_path,
            validation={"rows": int(len(submission)), "columns": list(submission.columns)},
            score=metrics.get("validation_roc_auc"),
            medal="none",
            final_artifact_source="ees_core:image_aerial",
            artifact_paths=[str(submission_path), str(metrics_path)],
            metrics=metrics,
            score_direction="maximize",  # score is validation ROC AUC
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id="aerial_image",
            recipe_id="metadata_audit",
            success=False,
            submission_path=None,
            validation={},
            score=None,
            medal="none",
            final_artifact_source="ees_core:image_aerial",
            no_score_reason=f"{type(exc).__name__}: {exc}",
        )
