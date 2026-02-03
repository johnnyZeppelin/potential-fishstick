from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


COMPRESSION_TYPES = {
    "JPEG": 0,
    "AVC": 1,
    "HEVC": 2,
}


@dataclass
class CVIQSample:
    image_id: str
    mos: float
    compression_type: int
    distortion_level: int


class CVIQDataset(Dataset):
    """Dataset loader for CVIQ with pre-extracted viewports."""

    def __init__(
        self,
        annotations_csv: Path,
        data_root: Path,
        viewports_root: Path,
        image_size: int = 224,
        viewport_size: int = 256,
        num_viewports: int = 20,
        transform: Callable | None = None,
        viewport_transform: Callable | None = None,
    ) -> None:
        self.annotations_csv = Path(annotations_csv)
        self.data_root = Path(data_root)
        self.viewports_root = Path(viewports_root)
        self.image_size = image_size
        self.viewport_size = viewport_size
        self.num_viewports = num_viewports
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.viewport_transform = viewport_transform or transforms.Compose(
            [
                transforms.Resize((viewport_size, viewport_size)),
                transforms.ToTensor(),
            ]
        )

        self._annotations = self._load_annotations()

    def _load_annotations(self) -> List[CVIQSample]:
        df = pd.read_csv(self.annotations_csv)
        samples: List[CVIQSample] = []
        for _, row in df.iterrows():
            image_id = f"{int(row['image_id']):03d}" if str(row["image_id"]).isdigit() else str(row["image_id"])
            compression = str(row["compression_type"]).upper()
            compression_idx = COMPRESSION_TYPES.get(compression)
            if compression_idx is None:
                raise ValueError(f"Unknown compression type: {compression}")
            samples.append(
                CVIQSample(
                    image_id=image_id,
                    mos=float(row["mos"]),
                    compression_type=compression_idx,
                    distortion_level=int(row["distortion_level"]),
                )
            )
        return samples

    def _load_image(self, path: Path, transform: Callable) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return transform(img)

    def _load_viewports(self, image_id: str, compression_folder: str) -> torch.Tensor:
        viewport_tensors: List[torch.Tensor] = []
        for idx in range(1, self.num_viewports + 1):
            viewport_path = self.viewports_root / compression_folder / f"{image_id}_fov{idx}.png"
            viewport_tensors.append(self._load_image(viewport_path, self.viewport_transform))
        return torch.stack(viewport_tensors, dim=0)

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int | float]:
        sample = self._annotations[index]
        image_path = self.data_root / f"{sample.image_id}.png"
        compression_folder = [k for k, v in COMPRESSION_TYPES.items() if v == sample.compression_type][0]
        global_image = self._load_image(image_path, self.transform)
        viewports = self._load_viewports(sample.image_id, compression_folder)

        return {
            "image_id": sample.image_id,
            "global_image": global_image,
            "viewports": viewports,
            "mos": torch.tensor(sample.mos, dtype=torch.float32),
            "compression_type": torch.tensor(sample.compression_type, dtype=torch.long),
            "distortion_level": torch.tensor(sample.distortion_level, dtype=torch.long),
        }
