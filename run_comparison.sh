#!/bin/bash
# =============================================================================
# MiniMind vs MiniKAN 对比实验脚本
# 
# 运行 1000 steps 的对比实验并生成 loss 曲线
# =============================================================================

echo "================================================================================"
echo " MiniMind vs MiniKAN 对比实验 "
echo "================================================================================"
echo ""

# 创建结果目录
mkdir -p results

# 配置参数
STEPS=1000
BATCH_SIZE=4
DIM=512
NUM_LAYERS=8
NUM_HEADS=8
SEQ_LEN=512

echo "实验配置:"
echo "  Steps: $STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Dim: $DIM"
echo "  Layers: $NUM_LAYERS"
echo "  Heads: $NUM_HEADS"
echo "  Seq Len: $SEQ_LEN"
echo ""

# 运行 MiniMind 训练
echo "================================================================================"
echo "[1/2] 训练 MiniMind..."
echo "================================================================================"
python train.py \
    --model minimind \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --dim $DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --seq_len $SEQ_LEN \
    --record_tps \
    --output_dir results

if [ $? -ne 0 ]; then
    echo "错误: MiniMind 训练失败"
    exit 1
fi

echo ""
echo "================================================================================"
echo "[2/2] 训练 MiniKAN..."
echo "================================================================================"
python train.py \
    --model minikan \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --dim $DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --seq_len $SEQ_LEN \
    --num_grids 5 \
    --init_scale 0.1 \
    --record_tps \
    --output_dir results

if [ $? -ne 0 ]; then
    echo "错误: MiniKAN 训练失败"
    exit 1
fi

echo ""
echo "================================================================================"
echo "训练完成! 生成对比图表..."
echo "================================================================================"

# 运行可视化脚本
python visualize_comparison.py --results_dir results

echo ""
echo "================================================================================"
echo "对比实验完成!"
echo "结果保存在: results/"
echo "================================================================================"
