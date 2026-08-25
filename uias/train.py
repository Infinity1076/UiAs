"""Leakage-resistant two-stage training entry point.

Held-out groups are excluded from training and evaluated separately by
``uias.test`` after the final checkpoints have been saved.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataload import make_group_disjoint_folds
from .model import UIASNet


LOSS_NAMES = (
    "total",
    "classification",
    "alignment",
    "orthogonality",
    "reconstruction",
    "contrastive",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters shared by the two training stages."""

    stage1_epochs: int = 100
    stage2_epochs: int = 200
    stage1_lr: float = 1e-4
    stage2_lr: float = 1e-4
    weight_decay: float = 1e-6
    align_weight: float = 1e-3
    orthogonality_weight: float = 10.0
    reconstruction_weight: float = 1e-2
    classification_weight: float = 1.0
    contrastive_weight: float = 1.0
    anchor_temperature: float = 0.1
    anchor_ema_momentum: float = 0.7
    anchor_condition_id: int = 0
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _inputs(
    batch: Mapping[str, torch.Tensor],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["input_2d"].to(device).float(),
        batch["input_ra"].to(device).float(),
        batch["acoustic"].to(device).float(),
        batch["label"].to(device).float(),
        batch["condition_id"].to(device).long(),
    )


@torch.no_grad()
def estimate_skin_anchor(
    model: UIASNet,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: str | torch.device,
    anchor_condition_id: int,
) -> torch.Tensor:
    """Average projected features from the configured clean anchor condition."""

    model.eval()
    features = []
    for batch in dataloader:
        input_2d, input_ra, acoustic, labels, conditions = _inputs(batch, device)
        output = model(input_2d, input_ra, acoustic, stage=1)
        mask = (labels > 0.5) & (conditions == anchor_condition_id)
        if bool(mask.any()):
            features.append(output["projected"][mask])

    if not features:
        raise RuntimeError(
            "no clean uncovered genuine-face samples were supplied for the anchor"
        )
    return torch.cat(features).mean(0)


def _train_epoch(
    model: UIASNet,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    config: TrainingConfig,
    stage: int,
) -> dict[str, float]:
    if stage == 1:
        model.train()
    else:
        model.eval()
        model.project_head.train()

    criterion = nn.BCEWithLogitsLoss()
    totals = {name: 0.0 for name in LOSS_NAMES}
    samples = 0

    for batch in dataloader:
        input_2d, input_ra, acoustic, labels, conditions = _inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_2d,
            input_ra,
            acoustic,
            labels=labels,
            condition_ids=conditions,
            stage=stage,
            anchor_condition_id=config.anchor_condition_id,
            temperature=config.anchor_temperature,
        )
        classification = criterion(output["logits"], labels)
        if stage == 1:
            total = (
                config.classification_weight * classification
                + config.align_weight * output["alignment"]
                + config.orthogonality_weight * output["orthogonality"]
                + config.reconstruction_weight * output["reconstruction"]
            )
        else:
            total = (
                config.classification_weight * classification
                + config.contrastive_weight * output["contrastive"]
            )

        total.backward()
        optimizer.step()

        count = int(labels.numel())
        samples += count
        values = {
            "total": total,
            "classification": classification,
            "alignment": output["alignment"],
            "orthogonality": output["orthogonality"],
            "reconstruction": output["reconstruction"],
            "contrastive": output["contrastive"],
        }
        for name, value in values.items():
            totals[name] += float(value.detach()) * count

    return {name: value / max(samples, 1) for name, value in totals.items()}


def _freeze_for_stage2(model: UIASNet) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.project_head.parameters():
        parameter.requires_grad = True


def fit_two_stage(
    model: UIASNet,
    stage1_loader: Iterable[Mapping[str, torch.Tensor]],
    stage2_loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: str | torch.device,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, object]:
    """Train both stages and save their final checkpoints."""

    if config.stage1_epochs < 1 or config.stage2_epochs < 1:
        raise ValueError("both training stages require at least one epoch")

    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    initial_anchor = estimate_skin_anchor(
        model,
        stage1_loader,
        device,
        config.anchor_condition_id,
    )
    model.initialize_skin_anchor(initial_anchor)

    stage1_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.stage1_lr,
        weight_decay=config.weight_decay,
    )
    history = {"stage1": [], "stage2": []}
    for epoch in range(1, config.stage1_epochs + 1):
        losses = _train_epoch(
            model,
            stage1_loader,
            stage1_optimizer,
            device,
            config,
            stage=1,
        )
        history["stage1"].append({"epoch": epoch, **losses})
    torch.save(model.state_dict(), output_dir / "stage1_final.pt")

    _freeze_for_stage2(model)
    stage2_optimizer = torch.optim.Adam(
        model.project_head.parameters(),
        lr=config.stage2_lr,
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, config.stage2_epochs + 1):
        losses = _train_epoch(
            model,
            stage2_loader,
            stage2_optimizer,
            device,
            config,
            stage=2,
        )
        fresh_anchor = estimate_skin_anchor(
            model,
            stage1_loader,
            device,
            config.anchor_condition_id,
        )
        with torch.no_grad():
            model.skin_anchor.mul_(config.anchor_ema_momentum).add_(
                fresh_anchor,
                alpha=1.0 - config.anchor_ema_momentum,
            )
        history["stage2"].append({"epoch": epoch, **losses})
    torch.save(model.state_dict(), output_dir / "model_final.pt")

    result = {
        "config": asdict(config),
        "history": history,
        "selection_policy": "fixed_epochs_no_test_selection",
    }
    (output_dir / "training_log.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def load_adapter_factory(specification: str) -> Callable[[], object]:
    """Resolve an adapter factory written as ``module:function``."""

    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--adapter must use module:function syntax")

    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"adapter factory is not callable: {specification}")
    return factory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="private module:factory")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--test-groups-per-fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    adapter = load_adapter_factory(args.adapter)()
    folds = make_group_disjoint_folds(
        adapter.group_ids,
        folds=args.folds,
        test_groups_per_fold=args.test_groups_per_fold,
        seed=args.seed,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }

    for fold in folds:
        stage1_dataset = adapter.stage1_dataset_for_groups(fold.train_groups)
        stage2_dataset = adapter.dataset_for_groups(fold.train_groups)
        fit_two_stage(
            UIASNet(),
            DataLoader(stage1_dataset, shuffle=True, **loader_options),
            DataLoader(stage2_dataset, shuffle=True, **loader_options),
            device=args.device,
            output_dir=args.output / f"fold_{fold.index + 1}",
            config=TrainingConfig(seed=args.seed + fold.index),
        )


if __name__ == "__main__":
    main()
