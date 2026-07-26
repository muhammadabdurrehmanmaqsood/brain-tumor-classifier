#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================"
echo "🚀 Starting Brain Tumor Model Training"
echo "========================================"

# Run the training script with explicit hyperparameters
python src/train.py \
    --data_dir "/content/dataset/Training" \
    --epochs 15 \
    --batch_size 32 \
    --lr 0.0005 \
    --save_path "resnet50_mri_best.pth"

echo "========================================"
echo "✅ Training Execution Completed."
echo "========================================"
