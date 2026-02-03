from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Experiment configuration for CVIQ reproduction."""

    data_root: Path = Path("data/CVIQ")
    viewports_root: Path = Path("data/view_ports")
    annotations_csv: Path = Path("data/cviq_annotations.csv")
    batch_size: int = 4
    num_workers: int = 2
    image_size: int = 224
    viewport_size: int = 256
    num_viewports: int = 20
    num_compression_types: int = 3
    num_distortion_levels: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    max_epochs: int = 300
    quality_loss_weight: float = 1.0
    distortion_loss_weight: float = 0.1
    compression_loss_weight: float = 0.1
