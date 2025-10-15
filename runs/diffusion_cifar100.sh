#!/bin/bash

# DiffusionFSCIL Training Script for CIFAR-100
# Training-Free Few-Shot Class-Incremental Learning via Generative Feature Diffusion

set -e  # Exit on error

PROJECT_NAME="diffusion_fscil_cifar100"
DATASET="cifar100"
GPU_ID="0"

echo "=================================================="
echo "  DiffusionFSCIL - CIFAR-100"
echo "=================================================="
echo "Project: $PROJECT_NAME"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "=================================================="

# Run training
python train.py \
    -project $PROJECT_NAME \
    -dataset $DATASET \
    -gpu $GPU_ID \
    -epochs_base 200 \
    -num_diffusion_steps 1000 \
    -ddim_steps 50 \
    -lr_diffusion 1e-4 \
    -batch_size_diffusion 256 \
    -batch_size_base 128 \
    -test_batch_size 100 \
    -num_workers 8 \
    -seed 1

echo "=================================================="
echo "  Training completed!"
echo "=================================================="

