"""Train a YOLO26 object detection model on PhenoBench data.

Usage:
    uv run src/train.py --config config/yolo26n.yaml
"""

import argparse
import tempfile
from pathlib import Path

import wandb
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26 for sugarbeet/weed detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="runs",
        help="Base directory for training outputs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load training configuration from a YAML file."""
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


def log_gradient_stats(trainer) -> None:
    """Log mean per-parameter gradient norm to W&B."""
    if wandb.run is None:
        return
    total_sq = 0.0
    n_params = 0
    for param in trainer.model.parameters():
        if param.grad is not None:
            total_sq += param.grad.data.norm(2).item() ** 2
            n_params += param.numel()
    if n_params > 0:
        wandb.log(
            {"gradients/mean_norm": (total_sq / n_params) ** 0.5},
            commit=False,
        )


def log_per_class_ap(trainer) -> None:
    """Log per-class AP metrics to W&B after validation."""
    if wandb.run is None:
        return
    validator = trainer.validator
    if validator is None:
        return
    names = validator.metrics.names
    ap50 = validator.metrics.box.ap50
    ap = validator.metrics.box.ap
    print(f"Logging per-class stats for {len(names)} classes to W&B")
    for i, name in names.items():
        wandb.log(
            {
                f"val/AP50_{name}": float(ap50[i]),
                f"val/AP50-95_{name}": float(ap[i]),
            },
            commit=False,  # Don't create separate step
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "sugarbeet-weed-detection")
    train_cfg = cfg.get("train", {})
    run_name = cfg["experiment"].get("id")

    # Resolve to absolute so Ultralytics skips its
    # RUNS_DIR/task/ prefix logic and saves directly to
    # export-dir/run_name.
    project = str(Path(args.export_dir).resolve())
    run_dir = Path(project) / run_name

    if args.resume:
        last_ckpt = run_dir / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.add_callback("on_fit_epoch_end", log_per_class_ap)
        model.add_callback("on_train_batch_end", log_gradient_stats)
        model.train(resume=True)
        return

    # Init W&B before Ultralytics so the correct project
    # is used, while local output stays under export-dir.
    wandb.init(project=wandb_project, name=run_name, config=cfg)

    data_yaml = write_data_yaml(cfg["data"])
    aug_cfg = cfg.get("augmentation", {})
    model = YOLO(cfg["model"])

    # Load pretrained weights from a different architecture
    # (e.g. yolo26x.pt into yolo26x-p2.yaml for transfer learning).
    # Matching layers transfer; mismatched layers stay random.
    pretrained_weights = cfg.get("pretrained_weights")
    if pretrained_weights:
        model.load(pretrained_weights)

    model.add_callback("on_fit_epoch_end", log_per_class_ap)
    model.add_callback("on_train_batch_end", log_gradient_stats)

    model.train(
        data=data_yaml,
        epochs=train_cfg.get("epochs", 100),
        batch=train_cfg.get("batch", 8),
        imgsz=train_cfg.get("imgsz", 1024),
        device=train_cfg.get("device", "0"),
        workers=train_cfg.get("workers", 8),
        project=project,
        name=run_name,
        patience=train_cfg.get("patience", 15),
        optimizer=train_cfg.get("optimizer", "auto"),
        box=train_cfg.get("box", 7.5),
        cls=train_cfg.get("cls", 0.5),
        dfl=train_cfg.get("dfl", 1.5),
        save=True,
        exist_ok=False,
        pretrained=pretrained_weights is None,
        verbose=True,
        **aug_cfg,
    )


if __name__ == "__main__":
    main()
