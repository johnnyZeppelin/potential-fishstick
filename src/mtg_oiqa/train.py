from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

from mtg_oiqa.config import Config
from mtg_oiqa.data.cviq_dataset import CVIQDataset
from mtg_oiqa.eval import EvaluationResult, evaluate_predictions
from mtg_oiqa.model import HeadConfig, MultiTaskGuidedOIQA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-task Guided OIQA on CVIQ")
    parser.add_argument("--annotations", type=Path, default=Config().annotations_csv)
    parser.add_argument("--data-root", type=Path, default=Config().data_root)
    parser.add_argument("--viewports-root", type=Path, default=Config().viewports_root)
    parser.add_argument("--batch-size", type=int, default=Config().batch_size)
    parser.add_argument("--epochs", type=int, default=Config().max_epochs)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, dataloader: DataLoader) -> EvaluationResult:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["global_image"], batch["viewports"])
            predictions.extend(outputs["quality"].cpu().numpy().tolist())
            targets.extend(batch["mos"].cpu().numpy().tolist())
    model.train()
    return evaluate_predictions(predictions, targets)


def main() -> None:
    args = parse_args()
    config = Config()
    set_seed(args.seed)

    dataset = CVIQDataset(
        annotations_csv=args.annotations,
        data_root=args.data_root,
        viewports_root=args.viewports_root,
        image_size=config.image_size,
        viewport_size=config.viewport_size,
        num_viewports=config.num_viewports,
    )
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * args.train_split)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.num_workers
    )

    model = MultiTaskGuidedOIQA(
        HeadConfig(
            num_compression_types=config.num_compression_types,
            num_distortion_levels=config.num_distortion_levels,
        )
    )
    model.train()

    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch["global_image"], batch["viewports"])
            quality_loss = mse_loss(outputs["quality"], batch["mos"])
            distortion_loss = ce_loss(outputs["distortion_logits"], batch["distortion_level"])
            compression_loss = ce_loss(outputs["compression_logits"], batch["compression_type"])
            loss = (
                config.quality_loss_weight * quality_loss
                + config.distortion_loss_weight * distortion_loss
                + config.compression_loss_weight * compression_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, test_loader)
        print(
            "Epoch {}/{} - Loss: {:.4f} - PLCC: {:.4f} - SRCC: {:.4f} - RMSE: {:.4f}".format(
                epoch + 1,
                args.epochs,
                total_loss / len(train_loader),
                metrics.plcc,
                metrics.srcc,
                metrics.rmse,
            )
        )


if __name__ == "__main__":
    main()
