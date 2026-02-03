from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CVIQ dataset layout")
    parser.add_argument("--data-root", type=Path, default=Path("data/CVIQ"))
    parser.add_argument("--viewports-root", type=Path, default=Path("data/view_ports"))
    parser.add_argument("--num-images", type=int, default=544)
    parser.add_argument("--num-viewports", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise SystemExit(f"Missing data root: {args.data_root}")
    if not args.viewports_root.exists():
        raise SystemExit(f"Missing viewports root: {args.viewports_root}")

    missing_images = []
    for idx in range(1, args.num_images + 1):
        image_id = f"{idx:03d}"
        image_path = args.data_root / f"{image_id}.png"
        if not image_path.exists():
            missing_images.append(image_path)

    if missing_images:
        print("Missing ERP images:")
        for path in missing_images[:10]:
            print(f"  - {path}")
        raise SystemExit(f"Missing {len(missing_images)} ERP images.")

    for compression in ["JPEG", "AVC", "HEVC"]:
        missing_viewports = []
        folder = args.viewports_root / compression
        if not folder.exists():
            raise SystemExit(f"Missing compression folder: {folder}")
        for idx in range(1, args.num_images + 1):
            image_id = f"{idx:03d}"
            for fov in range(1, args.num_viewports + 1):
                viewport_path = folder / f"{image_id}_fov{fov}.png"
                if not viewport_path.exists():
                    missing_viewports.append(viewport_path)
        if missing_viewports:
            print(f"Missing viewports for {compression}:")
            for path in missing_viewports[:10]:
                print(f"  - {path}")
            raise SystemExit(f"Missing {len(missing_viewports)} viewport images for {compression}.")

    print("CVIQ dataset layout looks correct.")


if __name__ == "__main__":
    main()
