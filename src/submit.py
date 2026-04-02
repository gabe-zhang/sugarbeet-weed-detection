"""Generate PhenoBench detection submission + overlay.

Runs best single model on val or test split, saves:
  - plant_bboxes/*.txt  (PhenoBench 6-col format)
  - overlay/*.png       (boxes drawn on images)

For val: also prints mAP metrics via model.val().
For test: packages plant_bboxes/ into a zip.

PhenoBench bbox format (per line, space-separated):
  <class> <cx> <cy> <w> <h> <confidence>
  class: 1=crop, 2=weed  (YOLO 0->1, 1->2)
  cx/cy/w/h: normalized [0,1]

Usage:
    # Val (metrics + overlay):
    uv run src/submit.py \
        --weights runs/yolo26x-p2-v1/weights/best.pt \
        --conf 0.30 --split val

    # Test (submission zip + overlay):
    uv run src/submit.py \
        --weights runs/yolo26x-p2-v1/weights/best.pt \
        --conf 0.30 --split test
"""

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# YOLO class -> PhenoBench class (0->1 crop, 1->2 weed)
CLS_MAP = {0: 1, 1: 2}
# Colors for overlay: crop=green, weed=red (BGR)
CLS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
CLS_NAMES = {0: "crop", 1: "weed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhenoBench detection submission"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to best.pt weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/PhenoBench/data.yaml",
        help="Data YAML for val metrics.",
    )
    parser.add_argument(
        "--test-img-dir",
        type=str,
        default="data/PhenoBench/test/images",
        help="Test images directory.",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Submission name (default: from weights).",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save overlay visualizations.",
    )
    return parser.parse_args()


def draw_boxes(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw bounding boxes on image with transparency."""
    overlay = img.copy()
    for box, cls_id, conf in zip(boxes_xyxy, cls_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        color = CLS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{CLS_NAMES.get(cls_id, '?')} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(
            overlay,
            (x1, y1 - th - 4),
            (x1 + tw, y1),
            color,
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def xyxy_to_cxcywh_norm(
    boxes: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    """Convert xyxy pixel boxes to normalized cxcywh."""
    x1, y1, x2, y2 = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=1)


def run_predict(
    model: YOLO,
    img_files: list[Path],
    imgsz: int,
    device: str,
    conf: float,
    bbox_dir: Path,
    overlay_dir: Path | None,
    augment: bool = False,
) -> None:
    """Run prediction, save bboxes and optionally overlays."""
    for img_path in tqdm(img_files, desc="Predicting"):
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            device=device,
            conf=conf,
            augment=augment,
            verbose=False,
            save=False,
        )
        r = results[0]
        h, w = r.orig_shape

        boxes = r.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            if overlay_dir:
                img = cv2.imread(str(img_path))
                vis = draw_boxes(img, xyxy, cls_ids, confs, alpha=1)
                cv2.imwrite(
                    str(overlay_dir / img_path.name),
                    vis,
                )

            # Save PhenoBench format bbox txt
            cxcywh = xyxy_to_cxcywh_norm(xyxy, w, h)
            pb_cls = np.array([CLS_MAP[int(c)] for c in cls_ids])
            lines = []
            for j in range(len(pb_cls)):
                lines.append(
                    f"{pb_cls[j]} "
                    f"{cxcywh[j, 0]:.6f} "
                    f"{cxcywh[j, 1]:.6f} "
                    f"{cxcywh[j, 2]:.6f} "
                    f"{cxcywh[j, 3]:.6f} "
                    f"{confs[j]:.6f}"
                )
            txt_name = img_path.stem + ".txt"
            (bbox_dir / txt_name).write_text("\n".join(lines) + "\n")
        else:
            if overlay_dir:
                img = cv2.imread(str(img_path))
                cv2.imwrite(
                    str(overlay_dir / img_path.name),
                    img,
                )
            txt_name = img_path.stem + ".txt"
            (bbox_dir / txt_name).write_text("")


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    # Derive submission name
    if args.name:
        sub_name = args.name
    else:
        sub_name = Path(args.weights).parts[-3]

    out_dir = Path(args.export_dir) / sub_name / f"submission_{args.split}"
    bbox_dir = out_dir / "plant_bboxes"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir: Path | None = None
    if args.plot:
        overlay_dir = out_dir / "overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {args.weights}")
    print(f"Split:  {args.split}")
    print(f"Conf:   {args.conf}")
    print(f"Output: {out_dir}")

    if args.split == "val":
        # Run model.val() for metrics
        data_cfg = yaml.safe_load(Path(args.data).read_text())
        data_path = Path(data_cfg["path"])
        val_imgs = data_path / data_cfg["val"]

        results = model.val(
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            augment=args.tta,
            conf=args.conf,
            plots=False,
            verbose=False,
            project=str(out_dir),
            name="val",
            exist_ok=True,
        )

        names = results.names
        ap50 = results.box.ap50
        ap = results.box.ap
        print(f"\n{'=' * 40}")
        print("  Val Results")
        print(f"{'=' * 40}")
        for i, name in names.items():
            print(f"  {name:>12s}:  AP50={ap50[i]:.4f}  AP50-95={ap[i]:.4f}")
        print(f"\n  {'mAP50':>12s}:  {results.box.map50:.4f}")
        print(f"  {'mAP50-95':>12s}:  {results.box.map:.4f}")
        print(f"{'=' * 40}")

        # Generate bboxes (+ overlays if --plot)
        img_files = sorted(val_imgs.glob("*.[jJpP][pPnN][gG]"))
        print(f"\nGenerating bboxes for {len(img_files)} val images...")
        run_predict(
            model,
            img_files,
            args.imgsz,
            args.device,
            args.conf,
            bbox_dir,
            overlay_dir,
            augment=args.tta,
        )

    elif args.split == "test":
        test_dir = Path(args.test_img_dir)
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test images not found: {test_dir}")
        img_files = sorted(test_dir.glob("*.[jJpP][pPnN][gG]"))
        print(f"Found {len(img_files)} test images in {test_dir}")

        run_predict(
            model,
            img_files,
            args.imgsz,
            args.device,
            args.conf,
            bbox_dir,
            overlay_dir,
            augment=args.tta,
        )

        # Create submission zip
        zip_path = out_dir / f"{sub_name}_bboxes.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.mkdir("plant_bboxes")
            for txt in sorted(bbox_dir.glob("*.txt")):
                zf.write(txt, f"plant_bboxes/{txt.name}")
        print(f"\nSubmission zip: {zip_path}")
        print(f"  Contains {len(list(bbox_dir.glob('*.txt')))} txt files")

    print(f"\nBboxes:   {bbox_dir}")
    if overlay_dir:
        print(f"Overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
