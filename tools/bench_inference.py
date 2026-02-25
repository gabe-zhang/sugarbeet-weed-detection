"""Benchmark YOLO inference time with and without warmup.

Measures per-image latency to show the effect of GPU warmup
(CUDA kernel compilation, memory allocation) on inference speed.

Usage:
    uv run tools/bench_inference.py \
        --weights runs/yolo26x-v1/weights/best.pt \
        --imgsz 1024 --warmup 10 --runs 50
"""

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference with/without warmup",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/PhenoBench",
        help="Dataset root (default: data/PhenoBench)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Split to benchmark on (default: val)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size (default: 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed iterations (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run on (default: 0)",
    )
    return parser.parse_args()


def collect_images(data_path: str, split: str) -> list[Path]:
    img_dir = Path(data_path) / "images" / split
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in exts
    )
    if not images:
        msg = f"No images found in {img_dir}"
        raise FileNotFoundError(msg)
    return images


def bench_cold(
    model: YOLO,
    images: list[Path],
    n: int,
    imgsz: int,
) -> list[float]:
    """Run inference without warmup (cold start)."""
    times = []
    for i in range(n):
        img = images[i % len(images)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_warm(
    model: YOLO,
    images: list[Path],
    n_warmup: int,
    n_runs: int,
    imgsz: int,
) -> list[float]:
    """Run warmup iterations, then measure inference."""
    # Warmup
    for i in range(n_warmup):
        img = images[i % len(images)]
        model.predict(img, imgsz=imgsz, verbose=False)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for i in range(n_runs):
        img = images[i % len(images)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def print_stats(label: str, times: list[float]) -> None:
    times_sorted = sorted(times)
    n = len(times_sorted)
    mean = sum(times_sorted) / n
    median = times_sorted[n // 2]
    p95 = times_sorted[int(n * 0.95)]
    p99 = times_sorted[int(n * 0.99)]
    first = times_sorted[0] if times_sorted else 0
    last = times[0]  # first chronological measurement

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  First call:  {last:8.2f} ms")
    print(f"  Mean:        {mean:8.2f} ms")
    print(f"  Median:      {median:8.2f} ms")
    print(f"  Min:         {first:8.2f} ms")
    print(f"  P95:         {p95:8.2f} ms")
    print(f"  P99:         {p99:8.2f} ms")
    print(f"  Total:       {sum(times):8.1f} ms ({n} runs)")


def main() -> None:
    args = parse_args()
    images = collect_images(args.data_path, args.split)
    n_total = min(args.runs, len(images))

    print(f"Model:   {args.weights}")
    print(f"Device:  cuda:{args.device}")
    print(f"ImgSz:   {args.imgsz}")
    print(f"Images:  {len(images)} ({args.split})")
    print(f"Warmup:  {args.warmup} iters")
    print(f"Runs:    {n_total} iters")

    # --- Cold start (fresh model, no warmup) ---
    model_cold = YOLO(args.weights)
    model_cold.to(f"cuda:{args.device}")
    cold_times = bench_cold(model_cold, images, n_total, args.imgsz)
    del model_cold
    torch.cuda.empty_cache()

    # --- Warm start (with warmup) ---
    model_warm = YOLO(args.weights)
    model_warm.to(f"cuda:{args.device}")
    warm_times = bench_warm(
        model_warm, images, args.warmup, n_total, args.imgsz
    )
    del model_warm
    torch.cuda.empty_cache()

    print_stats("Cold Start (no warmup)", cold_times)
    print_stats(
        f"Warm Start ({args.warmup} warmup iters)", warm_times
    )

    # Speedup
    cold_mean = sum(cold_times) / len(cold_times)
    warm_mean = sum(warm_times) / len(warm_times)
    print(f"\n{'=' * 50}")
    print(f"  Speedup: {cold_mean / warm_mean:.2f}x")
    print(f"  Cold mean: {cold_mean:.2f} ms  |"
          f"  Warm mean: {warm_mean:.2f} ms")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
