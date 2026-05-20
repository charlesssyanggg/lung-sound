#!/bin/bash
# Full experiment pipeline: Baseline → Focal Loss → Cross-dataset → Grad-CAM
set -e

echo "=== Step 1: Train Baseline ==="
python train_baseline.py

echo "=== Step 2: Train with Focal Loss (Innovation ①) ==="
python train_focal.py

echo "=== Step 3: Cross-Dataset Evaluation (Innovation ②) ==="
python src/cross_dataset.py \
    --baseline outputs/baseline_best.pth \
    --focal    outputs/focal_best.pth

echo "=== Step 4: Grad-CAM Visualization (Innovation ③) ==="
python src/gradcam.py --model_path outputs/focal_best.pth

echo "=== All experiments completed. Figures saved in figures/ ==="
