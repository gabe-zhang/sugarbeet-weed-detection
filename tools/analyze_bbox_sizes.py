"""Analyze bounding box sizes for each class in a dataset split.

Usage:
    uv run tools/analyze_bbox_sizes.py --split val
    uv run tools/analyze_bbox_sizes.py --split train --cls 1
"""

import argparse
from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "sugarbeet", 1: "weed"}
IMG_SIZE = 1024  # PhenoBench native resolution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze bbox sizes per class")
    p.add_argument(
        "--data-path",
        type=str,
        default="data/PhenoBench",
    )
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--cls",
        type=int,
        default=None,
        help="Filter to a single class (0=sugarbeet, 1=weed)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=IMG_SIZE,
        help="Image size for pixel conversion",
    )
    return p.parse_args()


def load_all_boxes(lbl_dir: Path, imgsz: int) -> dict[int, np.ndarray]:
    """Load all boxes grouped by class.

    Returns dict mapping class_id to array of [w_px, h_px].
    """
    boxes: dict[int, list[list[float]]] = {}
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        for line in lbl_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            w_px = float(parts[3]) * imgsz
            h_px = float(parts[4]) * imgsz
            boxes.setdefault(cls, []).append([w_px, h_px])

    return {k: np.array(v) for k, v in boxes.items()}


def print_stats(name: str, sizes: np.ndarray, imgsz: int) -> None:
    """Print size statistics for a set of boxes."""
    widths = sizes[:, 0]
    heights = sizes[:, 1]
    max_dim = np.maximum(widths, heights)
    area = widths * heights

    print(f"\n{'=' * 50}")
    print(f"  {name}: {len(sizes)} annotations")
    print(f"{'=' * 50}")

    # Width/height stats
    print(
        f"\n  {'':>12s}  {'Width':>8s}  {'Height':>8s}"
        f"  {'MaxDim':>8s}  {'Area':>10s}"
    )
    print(
        f"  {'':>12s}  {'-----':>8s}  {'------':>8s}"
        f"  {'------':>8s}  {'----':>10s}"
    )
    for label, fn in [
        ("Min", np.min),
        ("5th pct", lambda x: np.percentile(x, 5)),
        ("10th pct", lambda x: np.percentile(x, 10)),
        ("25th pct", lambda x: np.percentile(x, 25)),
        ("Median", np.median),
        ("Mean", np.mean),
        ("75th pct", lambda x: np.percentile(x, 75)),
        ("95th pct", lambda x: np.percentile(x, 95)),
        ("Max", np.max),
    ]:
        print(
            f"  {label:>12s}  {fn(widths):8.1f}  "
            f"{fn(heights):8.1f}  {fn(max_dim):8.1f}  "
            f"{fn(area):10.1f}"
        )

    # Small object counts
    thresholds = [10, 15, 20, 32, 64]
    print("\n  Small object counts (both dims < threshold):")
    for t in thresholds:
        count = int(np.sum(max_dim < t))
        pct = count / len(sizes) * 100
        print(f"    < {t:3d}px: {count:6d} ({pct:5.1f}%)")

    # Histogram by max dimension
    bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, imgsz]
    hist, _ = np.histogram(max_dim, bins=bins)
    print("\n  Size distribution (max dimension):")
    for i in range(len(bins) - 1):
        bar = "#" * int(hist[i] / max(hist) * 30)
        pct = hist[i] / len(sizes) * 100
        print(
            f"    {bins[i]:4d}-{bins[i + 1]:<4d}px: "
            f"{hist[i]:5d} ({pct:5.1f}%) {bar}"
        )


def main() -> None:
    args = parse_args()
    lbl_dir = Path(args.data_path) / "labels" / args.split

    if not lbl_dir.exists():
        print(f"Label dir not found: {lbl_dir}")
        return

    all_boxes = load_all_boxes(lbl_dir, args.imgsz)

    if not all_boxes:
        print("No annotations found.")
        return

    classes = [args.cls] if args.cls is not None else sorted(all_boxes.keys())

    total = sum(len(v) for v in all_boxes.values())
    print(f"Dataset: {args.data_path} / {args.split}")
    print(f"Total annotations: {total}")

    for cls in classes:
        if cls not in all_boxes:
            print(f"\nClass {cls}: no annotations found")
            continue
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        print_stats(name, all_boxes[cls], args.imgsz)


if __name__ == "__main__":
    main()
