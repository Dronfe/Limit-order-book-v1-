"""Training and evaluation for the dual-stream ice detector.

The dataset reads co-registered patches stored as ``.npy`` or GeoTIFF
(``.tif`` / ``.tiff``) files. Optimisation uses Adam with step-wise learning
rate decay and a combined Binary Cross-Entropy + Focal Loss objective, which
emphasises the sparse ice-positive pixels typical of lunar PSR mapping.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

try:
    from .models import DualStreamIceDetector
except ImportError:  # Allow `python src/train.py`
    from models import DualStreamIceDetector

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".npy", ".tif", ".tiff"}
DEFAULT_RADAR_CHANNELS = 4


def _load_array(path: Path) -> np.ndarray:
    """Load a patch from NumPy or GeoTIFF into a float32 array."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            array = src.read()
    else:
        raise ValueError(f"Unsupported patch format: {path}")
    return np.asarray(array, dtype=np.float32)


def _ensure_chw(array: np.ndarray) -> np.ndarray:
    """Convert ``(H, W)`` labels/images to ``(C, H, W)``."""
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected 2-D or 3-D array, got shape {array.shape}")


class IcePatchDataset(Dataset):
    """Dataset of preprocessed DFSAR / OHRC patches and ice masks.

    Directory layout (default)::

        root/
          radar/    HH-HV-VH-VV cubes  (C=4, H, W)
          optical/  OHRC patches       (C>=1, H, W)
          labels/   binary ice masks   (H, W) or (1, H, W)

    Files are paired by stem. A single stacked ``.npy`` / ``.tif`` under
    ``stacked/`` is also accepted: the first four channels are treated as
    radar and the remaining channels as optical.

    Args:
        root: Dataset root directory.
        split: Optional split subdirectory (e.g. ``train``, ``val``).
        radar_channels: Number of leading channels assigned to DFSAR.
        transform: Optional callable applied to ``(radar, optical, mask)``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        radar_channels: int = DEFAULT_RADAR_CHANNELS,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root) if split is None else Path(root) / split
        self.radar_channels = radar_channels
        self.transform = transform
        self.samples = self._index_samples()
        if not self.samples:
            raise FileNotFoundError(f"No patches found under {self.root}")

    def _list_stems(self, folder: Path) -> dict[str, Path]:
        files = {}
        if not folder.is_dir():
            return files
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                files[path.stem] = path
        return files

    def _index_samples(self) -> list[dict[str, Path]]:
        stacked_dir = self.root / "stacked"
        radar_dir = self.root / "radar"
        optical_dir = self.root / "optical"
        label_dir = self.root / "labels"

        samples: list[dict[str, Path]] = []
        labels = self._list_stems(label_dir)
        if not labels:
            raise FileNotFoundError(f"Expected label patches in {label_dir}")

        stacked = self._list_stems(stacked_dir)
        radar = self._list_stems(radar_dir)
        optical = self._list_stems(optical_dir)

        for stem, label_path in labels.items():
            item: dict[str, Path] = {"label": label_path}
            if stem in stacked:
                item["stacked"] = stacked[stem]
            elif stem in radar and stem in optical:
                item["radar"] = radar[stem]
                item["optical"] = optical[stem]
            else:
                logger.warning("Skipping unpaired label patch '%s'", stem)
                continue
            samples.append(item)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.samples[index]

        if "stacked" in item:
            cube = _ensure_chw(_load_array(item["stacked"]))
            if cube.shape[0] <= self.radar_channels:
                raise ValueError(
                    f"{item['stacked']} has {cube.shape[0]} channels; "
                    f"need > {self.radar_channels} to split radar/optical"
                )
            radar = cube[: self.radar_channels]
            optical = cube[self.radar_channels :]
        else:
            radar = _ensure_chw(_load_array(item["radar"]))
            optical = _ensure_chw(_load_array(item["optical"]))

        mask = _load_array(item["label"])
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.float32)

        if self.transform is not None:
            radar, optical, mask = self.transform(radar, optical, mask)

        return {
            "radar": torch.from_numpy(np.ascontiguousarray(radar)),
            "optical": torch.from_numpy(np.ascontiguousarray(optical)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0),
            "name": item["label"].stem,
        }


class FocalLoss(nn.Module):
    """Binary focal loss for highly imbalanced ice / background pixels.

    ``FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)``
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedBCEFocalLoss(nn.Module):
    """Weighted sum of BCE-with-logits and Focal Loss.

    BCE provides a stable dense gradient; Focal Loss up-weights the sparse
    ice-positive pixels that would otherwise be drowned out by the lunar
    regolith background.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )
        focal = self.focal(logits, targets)
        return self.bce_weight * bce + self.focal_weight * focal


@dataclass
class TrainConfig:
    """Hyper-parameters for a training run."""

    data_root: Path
    output_dir: Path
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_step: int = 15
    lr_gamma: float = 0.5
    num_workers: int = 4
    optical_channels: int = 1
    radar_channels: int = 4
    base_channels: int = 32
    num_heads: int = 4
    dropout: float = 0.1
    threshold: float = 0.5
    seed: int = 42
    device: str | None = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flatten_probs(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y_true = masks.detach().cpu().numpy().reshape(-1)
    y_true = (y_true > 0.5).astype(np.uint8)
    return y_true, probs


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision, recall, F1, and ROC-AUC on flattened pixels."""
    y_pred = (y_prob >= threshold).astype(np.uint8)
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # ROC-AUC is undefined if a batch contains only one class.
    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run a full validation pass and aggregate pixel-wise metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(optical, radar)
        loss = criterion(logits, masks)
        total_loss += float(loss.item())
        n_batches += 1

        y_true, y_prob = _flatten_probs(logits, masks)
        y_true_all.append(y_true)
        y_prob_all.append(y_prob)

    y_true_cat = np.concatenate(y_true_all)
    y_prob_cat = np.concatenate(y_prob_all)
    metrics = compute_metrics(y_true_cat, y_prob_cat, threshold=threshold)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def _log_metrics(phase: str, epoch: int, metrics: dict[str, float]) -> None:
    logger.info(
        "%s epoch %03d | loss=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        phase,
        epoch,
        metrics.get("loss", float("nan")),
        metrics.get("precision", float("nan")),
        metrics.get("recall", float("nan")),
        metrics.get("f1", float("nan")),
        metrics.get("roc_auc", float("nan")),
    )


def train(config: TrainConfig) -> DualStreamIceDetector:
    """Train ``DualStreamIceDetector`` and save the best F1 checkpoint."""
    set_seed(config.seed)
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = IcePatchDataset(config.data_root, split="train", radar_channels=config.radar_channels)
    val_root = config.data_root / "val"
    val_loader = None
    if val_root.is_dir() and (val_root / "labels").is_dir():
        val_set = IcePatchDataset(config.data_root, split="val", radar_channels=config.radar_channels)
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) >= config.batch_size,
    )

    model = DualStreamIceDetector(
        optical_channels=config.optical_channels,
        radar_channels=config.radar_channels,
        base_channels=config.base_channels,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)

    criterion = CombinedBCEFocalLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)

    best_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        y_true_all: list[np.ndarray] = []
        y_prob_all: list[np.ndarray] = []

        for batch in train_loader:
            radar = batch["radar"].to(device, non_blocking=True)
            optical = batch["optical"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(optical, radar)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
            y_true, y_prob = _flatten_probs(logits, masks)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

        scheduler.step()
        train_metrics = compute_metrics(
            np.concatenate(y_true_all),
            np.concatenate(y_prob_all),
            threshold=config.threshold,
        )
        train_metrics["loss"] = running_loss / max(n_batches, 1)
        _log_metrics("train", epoch, train_metrics)

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, threshold=config.threshold)
            _log_metrics("val  ", epoch, val_metrics)
            current_f1 = val_metrics["f1"]
        else:
            current_f1 = train_metrics["f1"]

        if current_f1 >= best_f1:
            best_f1 = current_f1
            ckpt_path = config.output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "f1": best_f1,
                    "config": vars(config),
                },
                ckpt_path,
            )
            logger.info("Saved checkpoint to %s (F1=%.4f)", ckpt_path, best_f1)

    final_path = config.output_dir / "last_model.pt"
    torch.save({"model_state": model.state_dict(), "epoch": config.epochs}, final_path)
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lunar ice detector")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root with train/val splits")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optical-channels", type=int, default=1)
    parser.add_argument("--radar-channels", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        num_workers=args.num_workers,
        optical_channels=args.optical_channels,
        radar_channels=args.radar_channels,
        base_channels=args.base_channels,
        num_heads=args.num_heads,
        dropout=args.dropout,
        threshold=args.threshold,
        seed=args.seed,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
