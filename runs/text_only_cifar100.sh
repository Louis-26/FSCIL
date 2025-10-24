#!/bin/bash

# Text-Only FSCIL Training Script for CIFAR-100
# 禁用扩散模型，仅使用文本原型的Few-Shot Class-Incremental Learning

set -e  # Exit on error

PROJECT_NAME="text_only_fscil_cifar100"
DATASET="cifar100"
GPU_ID="0"

echo "=================================================="
echo "  Text-Only FSCIL - CIFAR-100 (扩散已禁用)"
echo "=================================================="
echo "Project: $PROJECT_NAME"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "=================================================="

# Run training (已移除扩散相关参数)
python train.py \
    -project $PROJECT_NAME \
    -dataset $DATASET \
    -gpu $GPU_ID \
    -epochs_base 200 \
    -batch_size_base 128 \
    -test_batch_size 100 \
    -num_workers 8 \
    -seed 1

echo "=================================================="
echo "  Training completed!"
echo "=================================================="

