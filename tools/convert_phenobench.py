"""Convert PhenoBench instance + semantic annotations to YOLO format.

For each image, extracts per-plant annotations from instance masks
and assigns class labels from semantic masks:
  - Semantic class 1 (crop/sugarbeet) -> YOLO class 0
  - Semantic class 2 (weed)           -> YOLO class 1

Supports two output formats:
  --format bbox    : bounding box labels (default)
  --format polygon : polygon segment labels for copy-paste aug

Output: YOLO-format .txt files and symlinks to source images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEMANTIC_TO_YOLO = {1: 0, 2: 1}

# Downsample contours with many points to keep label files
# small.  Value chosen so the polygon still closely matches
# the original mask boundary.
MAX_POLYGON_POINTS = 50


def mask_to_polygon(
    mask: np.ndarray,
    img_height: int,
    img_width: int,
) -> list[float] | None:
    """Extract the largest contour from a binary mask.

    Returns normalized [x1, y1, x2, y2, ...] or None if no
    valid contour is found.
    """
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    # Downsample if too many points
    if len(contour) > MAX_POLYGON_POINTS:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        while len(contour) > MAX_POLYGON_POINTS:
            contour = cv2.approxPolyDP(contour, epsilon, True)
            epsilon *= 1.5
        if len(contour) < 3:
            return None

    # Normalize coordinates
    coords = []
    for point in contour:
        x, y = point[0]
        coords.append(x / img_width)
        coords.append(y / img_height)
    return coords


def convert_image_bbox(
    instance_path: Path,
    semantic_path: Path,
    label_path: Path,
    img_height: int,
    img_width: int,
) -> int:
    """Convert annotations to YOLO bounding box format."""
    instances = np.array(Image.open(instance_path))
    semantics = np.array(Image.open(semantic_path))

    lines = []
    for plant_id in np.unique(instances):
        if plant_id == 0:
            continue

        mask = instances == plant_id
        sem_class = int(semantics[mask][0])
        yolo_class = SEMANTIC_TO_YOLO.get(sem_class)
        if yolo_class is None:
            continue

        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height

        lines.append(
            f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        )

    label_path.write_text("\n".join(lines) + "\n" if lines else "")
    return len(lines)


def convert_image_polygon(
    instance_path: Path,
    semantic_path: Path,
    label_path: Path,
    img_height: int,
    img_width: int,
) -> int:
    """Convert annotations to YOLO polygon segment format."""
    instances = np.array(Image.open(instance_path))
    semantics = np.array(Image.open(semantic_path))

    lines = []
    for plant_id in np.unique(instances):
        if plant_id == 0:
            continue

        mask = instances == plant_id
        sem_class = int(semantics[mask][0])
        yolo_class = SEMANTIC_TO_YOLO.get(sem_class)
        if yolo_class is None:
            continue

        coords = mask_to_polygon(mask, img_height, img_width)
        if coords is None:
            continue

        coord_str = " ".join(f"{c:.6f}" for c in coords)
        lines.append(f"{yolo_class} {coord_str}")

    label_path.write_text("\n".join(lines) + "\n" if lines else "")
    return len(lines)


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    fmt: str,
) -> None:
    """Convert one data split (train or val)."""
    src_images = src_root / split / "images"
    src_instances = src_root / split / "plant_instances"
    src_semantics = src_root / split / "semantics"

    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    convert_fn = (
        convert_image_polygon if fmt == "polygon" else convert_image_bbox
    )

    image_files = sorted(src_images.glob("*.png"))
    total_objects = 0

    for img_file in image_files:
        stem = img_file.stem

        # Symlink image
        link = dst_images / img_file.name
        if not link.exists():
            link.symlink_to(img_file.resolve())

        # Get image dimensions
        img = Image.open(img_file)
        w, h = img.size

        # Convert annotations
        instance_path = src_instances / f"{stem}.png"
        semantic_path = src_semantics / f"{stem}.png"
        label_path = dst_labels / f"{stem}.txt"

        n = convert_fn(instance_path, semantic_path, label_path, h, w)
        total_objects += n

    print(f"  {split}: {len(image_files)} images, {total_objects} objects")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Convert PhenoBench to YOLO detection format")
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/home/zy/repos/sugarbeet-weed-segmentation/data/PhenoBench"
        ),
        help="Path to PhenoBench dataset root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/PhenoBench"),
        help="Output directory for YOLO-format dataset",
    )
    parser.add_argument(
        "--format",
        choices=["bbox", "polygon"],
        default="bbox",
        help="Label format: bbox or polygon (default: bbox)",
    )
    args = parser.parse_args()

    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Format: {args.format}")

    for split in ("train", "val"):
        convert_split(args.src, args.dst, split, args.format)

    print("Done.")


if __name__ == "__main__":
    main()
