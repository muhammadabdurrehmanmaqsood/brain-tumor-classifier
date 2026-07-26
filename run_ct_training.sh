%%writefile run_ct_training.sh
#!/bin/bash
set -e

echo "========================================"
echo "🚀 Starting CT Scan Model Training"
echo "========================================"

python src/train.py \
    --data_dir "/content/dataset/ct" \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.0005 \
    --save_path "resnet50_ct_best.pth"

echo "========================================"
echo "✅ CT Training Execution Completed."
echo "========================================"
