"""Sweep confidence thresholds for best val performance.

Runs validation on one or more models across a grid of
confidence thresholds, logging results to a text file.

Usage:
    uv run tools/sweep_conf.py \
        --weights runs/yolo26x-p2-v1/weights/best.pt \
        --weights runs/yolo26x-v1/weights/best.pt \
        --data data/PhenoBench/data.yaml \
        --out runs/sweep_results.txt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep conf for best val performance"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        action="append",
        help="Path to weights (repeat for multiple models).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/PhenoBench/data.yaml",
        help="Path to data YAML.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image size (default: 1024).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (default: '0').",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation.",
    )
    parser.add_argument(
        "--conf-range",
        type=float,
        nargs=3,
        default=[0.3, 0.7, 0.025],
        help="Conf sweep: start stop step (default: 0.3 0.7 0.025)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/sweep_results.txt",
        help="Output file (default: runs/sweep_results.txt).",
    )
    return parser.parse_args()


def frange(start: float, stop: float, step: float):
    """Float range generator."""
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    confs = frange(*args.conf_range)

    models = []
    for w in args.weights:
        models.append((w, YOLO(w)))

    with open(out_path, "w") as f:
        header = (
            f"{'weights':<50s}  {'conf':>5s}  "
            f"{'mAP50':>7s}  {'mAP50-95':>8s}  "
            f"{'AP50_crop':>9s}  {'AP50-95_crop':>12s}  "
            f"{'AP50_weed':>9s}  {'AP50-95_weed':>12s}  "
            f"{'P':>7s}  {'R':>7s}  {'TTA':>3s}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        f.flush()
        print(header)
        print("-" * len(header))

        for w_path, model in models:
            w_name = Path(w_path).parts[-3]
            for conf in confs:
                results = model.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    device=args.device,
                    augment=args.tta,
                    conf=conf,
                    plots=False,
                    verbose=False,
                )
                ap50 = results.box.ap50
                ap = results.box.ap
                mp = results.box.mp
                mr = results.box.mr
                line = (
                    f"{w_name:<50s}  {conf:>5.2f}  "
                    f"{results.box.map50:>7.4f}  "
                    f"{results.box.map:>8.4f}  "
                    f"{ap50[0]:>9.4f}  {ap[0]:>12.4f}  "
                    f"{ap50[1]:>9.4f}  {ap[1]:>12.4f}  "
                    f"{mp:>7.4f}  {mr:>7.4f}  "
                    f"{'Y' if args.tta else 'N':>3s}"
                )
                f.write(line + "\n")
                f.flush()
                print(line)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
