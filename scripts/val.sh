#!/bin/bash

uv run src/val.py \
    --config config/yolo26n.yaml \
    --weights runs/detect/runs/yolo26n-v1/weights/best.pt
