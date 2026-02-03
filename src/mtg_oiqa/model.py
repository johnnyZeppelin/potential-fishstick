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
    """Global branch backbone (ResNet50 placeholder for VMamba)."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f2, f3, f4


class LocalFeatureExtractor(nn.Module):
    """Shared local feature extractor for viewports."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f2, f3, f4


class BidirectionalPseudoReference(nn.Module):
    """Generate restoration/degradation predictions and error maps."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.restoration = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )
        self.degradation = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        restored = self.restoration(x)
        degraded = self.degradation(x)
        error_map = torch.abs(restored - degraded)
        return restored, error_map


class BSMSFA(nn.Module):
    """Bi-stream multi-scale feature aggregation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.local_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.global_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_proj(local_feat)
        global_feat = self.global_proj(global_feat)
        fused = torch.cat([local_feat, global_feat], dim=1)
        fused = self.fusion(fused)
        return fused.flatten(1)


class MultiTaskGuidedOIQA(nn.Module):
    """Expanded implementation aligned with the paper's multi-task head."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()

        self.pseudo_reference = BidirectionalPseudoReference()
        self.global_extractor = GlobalFeatureExtractor()
        self.local_extractor = LocalFeatureExtractor()
        self.bsmsfa2 = BSMSFA(in_channels=512, out_channels=256)
        self.bsmsfa3 = BSMSFA(in_channels=1024, out_channels=256)
        self.bsmsfa4 = BSMSFA(in_channels=2048, out_channels=256)

        fused_dim = 256 * 3
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
        _, error_map = self.pseudo_reference(viewports)
        local_f2, local_f3, local_f4 = self.local_extractor(error_map)
        local_f2 = local_f2.view(batch_size, num_viewports, *local_f2.shape[1:]).mean(dim=1)
        local_f3 = local_f3.view(batch_size, num_viewports, *local_f3.shape[1:]).mean(dim=1)
        local_f4 = local_f4.view(batch_size, num_viewports, *local_f4.shape[1:]).mean(dim=1)

        global_f2, global_f3, global_f4 = self.global_extractor(global_image)
        fused2 = self.bsmsfa2(local_f2, global_f2)
        fused3 = self.bsmsfa3(local_f3, global_f3)
        fused4 = self.bsmsfa4(local_f4, global_f4)
        fused = torch.cat([fused2, fused3, fused4], dim=1)
        fused = self.fusion(fused)

        return {
            "quality": self.quality_head(fused).squeeze(1),
            "distortion_logits": self.distortion_head(fused),
            "compression_logits": self.compression_head(fused),
        }
