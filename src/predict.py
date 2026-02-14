"""Run YOLO26 inference on image directories.

Usage:
    uv run src/predict.py \
        --weights runs/yolo26n-v3/weights/best.pt \
        --source data/mh0_filtered/images

    uv run src/predict.py \
        --weights best.pt --source data/ --all
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 prediction on image directories"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image directory or parent data directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all subdirectories under --source",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size (default: 1024)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run on (default: '0')",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection labels as .txt files",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="runs/predict",
        help="Base output directory (default: runs/predict)",
    )
    return parser.parse_args()


def find_image_dirs(root: Path) -> list[Path]:
    """Find all 'images/' subdirectories under root."""
    dirs = sorted(root.glob("*/images"))
    return [d for d in dirs if d.is_dir()]


def predict_dir(
    model: YOLO,
    source: Path,
    name: str,
    args: argparse.Namespace,
) -> None:
    """Run prediction on a single image directory."""
    n_images = len(list(source.glob("*.[jJ][pP][gG]")))
    n_images += len(list(source.glob("*.[pP][nN][gG]")))
    print(f"\n{'=' * 50}")
    print(f"Dataset: {name} ({n_images} images)")
    print(f"Source:  {source}")
    print(f"{'=' * 50}")

    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=str(Path(args.export_dir).resolve()),
        name=name,
        exist_ok=True,
    )


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    source = Path(args.source)

    if args.all:
        img_dirs = find_image_dirs(source)
        if not img_dirs:
            print(f"No */images/ dirs found under {source}")
            return
        print(f"Found {len(img_dirs)} dataset(s):")
        for d in img_dirs:
            print(f"  - {d.parent.name}/images/")
        for d in img_dirs:
            predict_dir(model, d, d.parent.name, args)
    else:
        name = source.parent.name if source.name == "images" else source.name
        predict_dir(model, source, name, args)


if __name__ == "__main__":
    main()
