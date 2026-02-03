from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from mtg_oiqa.config import Config
from mtg_oiqa.data.cviq_dataset import CVIQDataset
from mtg_oiqa.model import HeadConfig, MultiTaskGuidedOIQA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-task Guided OIQA on CVIQ")
    parser.add_argument("--annotations", type=Path, default=Config().annotations_csv)
    parser.add_argument("--data-root", type=Path, default=Config().data_root)
    parser.add_argument("--viewports-root", type=Path, default=Config().viewports_root)
    parser.add_argument("--batch-size", type=int, default=Config().batch_size)
    parser.add_argument("--epochs", type=int, default=Config().max_epochs)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    dataset = CVIQDataset(
        annotations_csv=args.annotations,
        data_root=args.data_root,
        viewports_root=args.viewports_root,
        image_size=config.image_size,
        viewport_size=config.viewport_size,
        num_viewports=config.num_viewports,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=config.num_workers)

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
        for batch in dataloader:
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

        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    main()
