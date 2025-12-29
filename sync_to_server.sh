#!/bin/bash
# 同步代码到服务器，排除输出和权重文件

# 配置
LOCAL_DIR="/path/to/your/local/pu"  # 本地路径
REMOTE_USER="junyu"
REMOTE_HOST="your-server-ip"
REMOTE_DIR="/home/junyu/project/pu"

# rsync 选项说明：
# -a: 归档模式（保留权限、时间等）
# -v: 详细输出
# -z: 压缩传输
# -h: 人类可读的格式
# --progress: 显示进度
# --delete: 删除服务器上多余的文件（慎用）
# --exclude: 排除匹配的文件/目录

rsync -avzh --progress \
    --exclude='outputs/' \
    --exclude='experiments/outputs/' \
    --exclude='experiments/lightning_logs/' \
    --exclude='experiments/ckpt/' \
    --exclude='experiments/wandb/' \
    --exclude='output/weight/' \
    --exclude='esmfold_eval/' \
    --exclude='esmfold_evaluation/' \
    --exclude='evaluations/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='*.npz' \
    --exclude='*.npy' \
    --exclude='*.pkl' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='*.log' \
    --exclude='examples/*/preprocessed/' \
    --exclude='.git/' \
    "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "✓ Sync completed!"
