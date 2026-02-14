#!/bin/bash

uv run src/train.py \
    --config config/yolo26n.yaml \
    --export-dir runs
