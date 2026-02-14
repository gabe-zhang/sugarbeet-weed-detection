# Sugarbeet Weed Detection

[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)]()
[![Lightning](https://img.shields.io/badge/Lightning-2.6.1-blueviolet.svg)]()

A small object-detection project for detecting sugarbeet (crop) vs. weed using YOLO26.

## Quick start

- Convert PhenoBench semantic annotations to YOLO format:

```bash
python tools/convert_phenobench.py --dst data/PhenoBench
```

- Start training (uses `uv`):

```bash
yolo settings wandb=True
./scripts/train.sh
# or run directly
uv run src/train.py --config config/yolo26n.yaml --export-dir runs
```

## Files of interest

- `tools/convert_phenobench.py` — converts segmentation dataset from the sibling repo into YOLO-format labels in `data/PhenoBench`.
- `src/train.py` — training entry point using the `ultralytics` YOLO API.
- `config/yolo26n.yaml` — training hyperparameters and dataset config.
- `models/yolo26n.pt` — default model checkpoint used for training.

## Dependencies

See `pyproject.toml` for project dependencies (Lightning, Ultralytics YOLO, PyTorch).

## Notes

This repo is intended to be used alongside `../sugarbeet-weed-segmentation` (PhenoBench source dataset and segmentation training tools).

---

## License & Data Attribution

- This project is licensed under the **MIT License**. See `LICENSE` for details.

- The dataset used to produce detection labels is the **PhenoBench** dataset, which is available under the **CC BY-SA 4.0** license. Please cite PhenoBench if you use the dataset in publications or products:
  - Website: https://www.phenobench.org/
  - License: https://creativecommons.org/licenses/by-sa/4.0/

- For full citation and dataset details, see [CITATION.md](https://github.com/gabe-zhang/sugarbeet-weed-segmentation/blob/main/CITATION.md) and the segmentation repo README.

---

## Third-party notice

Third-party: Ultralytics YOLO (YOLO26) — AGPL‑3.0.

---
