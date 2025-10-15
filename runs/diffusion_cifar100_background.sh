#!/bin/bash

# DiffusionFSCIL 后台训练脚本 - CIFAR-100
# 使用nohup在后台运行，日志保存到文件

set -e

PROJECT_NAME="diffusion_fscil_cifar100"
DATASET="cifar100"
GPU_ID="0"
LOG_FILE="training_diffusion_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================="
echo "  DiffusionFSCIL - CIFAR-100 (后台运行)"
echo "=================================================="
echo "Project: $PROJECT_NAME"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Log file: $LOG_FILE"
echo "=================================================="
echo ""
echo "启动后台训练..."

# 使用nohup在后台运行
nohup python -u train.py \
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
    -seed 1 \
    > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!

echo "✅ 训练已在后台启动！"
echo ""
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控训练："
echo "  实时查看: tail -f $LOG_FILE"
echo "  查看进度: grep 'Session' $LOG_FILE"
echo "  查看Loss: grep 'Loss:' $LOG_FILE"
echo ""
echo "管理进程："
echo "  查看状态: ps -p $PID"
echo "  终止训练: kill $PID"
echo ""
echo "=================================================="

# 保存PID到文件
echo $PID > training.pid
echo "PID已保存到 training.pid"

