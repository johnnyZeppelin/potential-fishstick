from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack viewport tool zip from main branch")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("twentyviewportstool.zip"),
        help="Path to twentyviewportstool.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/twenty_viewports"),
        help="Directory to extract the tool contents",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip_path.exists():
        raise SystemExit(f"Missing zip file: {args.zip_path}")

    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {args.output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "r") as zip_ref:
        zip_ref.extractall(args.output_dir)

    print(f"Extracted {args.zip_path} -> {args.output_dir}")


if __name__ == "__main__":
    main()
