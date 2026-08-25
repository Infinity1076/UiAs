"""Evaluate trained UIAS checkpoints on held-out groups using accuracy."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataload import make_group_disjoint_folds
from .model import UIASNet
from .train import load_adapter_factory


@torch.no_grad()
def evaluate_accuracy(
    model: UIASNet,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: str | torch.device,
) -> float:
    """Return classification accuracy at a zero-logit threshold."""

    model.eval()
    correct = 0
    samples = 0
    for batch in dataloader:
        output = model(
            batch["input_2d"].to(device).float(),
            batch["input_ra"].to(device).float(),
            batch["acoustic"].to(device).float(),
            stage=1,
        )
        labels = batch["label"].to(device).float()
        predictions = output["logits"] >= 0.0
        correct += int((predictions == (labels > 0.5)).sum())
        samples += int(labels.numel())

    if samples == 0:
        raise ValueError("held-out dataset is empty")
    return correct / samples


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="private module:factory")
    parser.add_argument("--checkpoints", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
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

    fold_results = []
    for fold in folds:
        checkpoint = args.checkpoints / f"fold_{fold.index + 1}" / "model_final.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        model = UIASNet().to(args.device)
        state_dict = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(state_dict)

        held_out_dataset = adapter.dataset_for_groups(fold.test_groups)
        accuracy = evaluate_accuracy(
            model,
            DataLoader(held_out_dataset, shuffle=False, **loader_options),
            args.device,
        )
        fold_results.append(
            {
                "fold": fold.index + 1,
                "accuracy": accuracy,
            }
        )

    result = {
        "folds": fold_results,
        "mean_accuracy": sum(item["accuracy"] for item in fold_results)
        / len(fold_results),
    }
    output = args.output or args.checkpoints / "test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
