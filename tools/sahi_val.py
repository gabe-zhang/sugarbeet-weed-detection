"""Evaluate YOLO26 with SAHI (Sliced Aided Hyper Inference).

Runs sliced inference on the val set and evaluates using
Ultralytics' own AP calculation so results are directly
comparable to standard model.val() output.

Usage:
    uv run tools/sahi_val.py \
        --weights runs/yolo26x-v1/weights/best.pt \
        --data-path data/PhenoBench \
        --slice-size 512 --overlap 0.2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics, box_iou


CLASS_NAMES = {0: "sugarbeet", 1: "weed"}
# Same IoU thresholds as Ultralytics (0.50:0.05:0.95)
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAHI sliced inference evaluation")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data-path", type=str, default="data/PhenoBench")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--slice-size", type=int, default=512)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
    """Load YOLO labels as [cls, x1, y1, x2, y2] tensor."""
    if not label_path.exists():
        return torch.zeros((0, 5))

    boxes = []
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h
        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2
        boxes.append([cls, x1, y1, x2, y2])

    if not boxes:
        return torch.zeros((0, 5))
    return torch.tensor(boxes)


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
) -> torch.Tensor:
    """Match predictions to GT using Ultralytics' method.

    Returns bool tensor of shape (n_preds, 10) indicating
    TP at each IoU threshold.
    """
    n_pred = pred_boxes.shape[0]
    correct = torch.zeros((n_pred, len(IOU_THRESHOLDS)), dtype=torch.bool)

    if gt_boxes.shape[0] == 0 or n_pred == 0:
        return correct

    iou = box_iou(gt_boxes, pred_boxes)

    # Zero out IoU for class mismatches
    correct_class = gt_cls[:, None] == pred_cls[None, :]
    iou = iou * correct_class.float()

    for ti, threshold in enumerate(IOU_THRESHOLDS):
        matches = torch.nonzero(iou >= threshold, as_tuple=False)
        if matches.shape[0] == 0:
            continue

        # Get IoU values for each match
        match_ious = iou[matches[:, 0], matches[:, 1]]

        # Sort by IoU descending
        order = match_ious.argsort(descending=True)
        matches = matches[order]

        # Greedy unique assignment
        seen_gt = set()
        seen_pred = set()
        for gt_idx, pred_idx in matches.tolist():
            if gt_idx in seen_gt or pred_idx in seen_pred:
                continue
            seen_gt.add(gt_idx)
            seen_pred.add(pred_idx)
            correct[pred_idx, ti] = True

    return correct


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)

    print(f"Weights: {args.weights}")
    print(
        f"Slice: {args.slice_size}x{args.slice_size}, overlap: {args.overlap}"
    )
    print(f"Conf threshold: {args.conf}")

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=f"cuda:{args.device}",
        image_size=args.imgsz,
    )

    img_dir = data_path / "images" / args.split
    lbl_dir = data_path / "labels" / args.split
    img_files = sorted(img_dir.glob("*.png"))

    print(f"Evaluating {len(img_files)} images...")

    metrics = DetMetrics(names=CLASS_NAMES)

    for img_idx, img_file in enumerate(tqdm(img_files, desc="SAHI inference")):
        img = Image.open(img_file)
        img_w, img_h = img.size

        # Load GT
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        gt = load_gt_boxes(lbl_file, img_w, img_h)
        gt_cls = gt[:, 0] if gt.shape[0] > 0 else torch.zeros(0)
        gt_boxes = gt[:, 1:5] if gt.shape[0] > 0 else gt

        # Run SAHI
        result = get_sliced_prediction(
            str(img_file),
            det_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )

        preds = result.object_prediction_list

        if not preds:
            # No predictions — still record GT classes
            metrics.update_stats(
                {
                    "tp": np.zeros((0, 10), dtype=bool),
                    "conf": np.zeros(0),
                    "pred_cls": np.zeros(0),
                    "target_cls": gt_cls.numpy(),
                    "target_img": np.full(len(gt_cls), img_idx),
                }
            )
            continue

        # Collect predictions
        pred_boxes = []
        pred_scores = []
        pred_classes = []
        for pred in preds:
            bbox = pred.bbox
            pred_boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            pred_scores.append(pred.score.value)
            pred_classes.append(pred.category.id)

        pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float)
        pred_scores_t = torch.tensor(pred_scores, dtype=torch.float)
        pred_classes_t = torch.tensor(pred_classes, dtype=torch.float)

        # Match predictions to GT
        tp = match_predictions(pred_boxes_t, pred_classes_t, gt_boxes, gt_cls)

        metrics.update_stats(
            {
                "tp": tp.numpy(),
                "conf": pred_scores_t.numpy(),
                "pred_cls": pred_classes_t.numpy(),
                "target_cls": gt_cls.numpy(),
                "target_img": np.full(len(gt_cls), img_idx),
            }
        )

    metrics.process()
    results = metrics.results_dict

    # Print results matching val.py format
    print("\n=== Per-Class Results ===")
    ap50_all = metrics.box.ap50
    ap_all = metrics.box.ap
    for i, name in CLASS_NAMES.items():
        print(
            f"  {name:>12s}:  "
            f"AP50 = {ap50_all[i]:.4f}  "
            f"AP50-95 = {ap_all[i]:.4f}"
        )

    print(f"\n  {'mAP50':>12s}:  {results['metrics/mAP50(B)']:.4f}")
    print(f"  {'mAP50-95':>12s}:  {results['metrics/mAP50-95(B)']:.4f}")
    print(f"  {'Precision':>12s}:  {results['metrics/precision(B)']:.4f}")
    print(f"  {'Recall':>12s}:  {results['metrics/recall(B)']:.4f}")


if __name__ == "__main__":
    main()
