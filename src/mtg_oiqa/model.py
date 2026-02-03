from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass
class HeadConfig:
    feature_dim: int = 2048
    hidden_dim: int = 512
    num_compression_types: int = 3
    num_distortion_levels: int = 5


class GlobalFeatureExtractor(nn.Module):
    """Placeholder global branch (VMamba can be swapped in later)."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return feats.flatten(1)


class LocalFeatureExtractor(nn.Module):
    """Shared local feature extractor for viewports."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return feats.flatten(1)


class MultiTaskGuidedOIQA(nn.Module):
    """Simplified implementation aligned with the paper's multi-task head."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()

        self.global_extractor = GlobalFeatureExtractor()
        self.local_extractor = LocalFeatureExtractor()

        fused_dim = self.config.feature_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.quality_head = nn.Linear(self.config.hidden_dim, 1)
        self.distortion_head = nn.Linear(self.config.hidden_dim, self.config.num_distortion_levels)
        self.compression_head = nn.Linear(self.config.hidden_dim, self.config.num_compression_types)

    def forward(self, global_image: torch.Tensor, viewports: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_viewports = viewports.shape[:2]
        viewports = viewports.view(batch_size * num_viewports, *viewports.shape[2:])
        local_feats = self.local_extractor(viewports)
        local_feats = local_feats.view(batch_size, num_viewports, -1).mean(dim=1)

        global_feats = self.global_extractor(global_image)
        fused = torch.cat([local_feats, global_feats], dim=1)
        fused = self.fusion(fused)

        return {
            "quality": self.quality_head(fused).squeeze(1),
            "distortion_logits": self.distortion_head(fused),
            "compression_logits": self.compression_head(fused),
        }
