"""Validate a YOLO26 model on PhenoBench data.

Usage:
    uv run src/val.py --config config/yolo26n.yaml \
        --weights runs/yolo26n-v3/weights/best.pt
"""

import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO26 for sugarbeet/weed detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load configuration from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def write_data_yaml(data_cfg: dict) -> str:
    """Write the data section to a temp YAML for Ultralytics."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data_cfg, tmp)
    tmp.close()
    return tmp.name


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    val_cfg = cfg.get("val", {})

    data_yaml = write_data_yaml(cfg["data"])

    # Derive save dir from weights path:
    # .../yolo26n-v1/weights/best.pt → project=.../yolo26n-v1, name=val
    weights_path = Path(args.weights).resolve()
    run_dir = weights_path.parent.parent
    project = str(run_dir)
    name = "val"

    model = YOLO(args.weights)
    results = model.val(
        data=data_yaml,
        imgsz=val_cfg.get("imgsz", 1024),
        device=val_cfg.get("device", "0"),
        project=project,
        name=name,
        exist_ok=True,
    )

    # Print per-class results
    names = results.names
    ap50 = results.box.ap50
    ap = results.box.ap
    print("\n=== Per-Class Results ===")
    for i, name in names.items():
        print(f"  {name:>12s}:  AP50 = {ap50[i]:.4f}  AP50-95 = {ap[i]:.4f}")
    print(f"\n  {'mAP50':>12s}:  {results.box.map50:.4f}")
    print(f"  {'mAP50-95':>12s}:  {results.box.map:.4f}")


if __name__ == "__main__":
    main()
