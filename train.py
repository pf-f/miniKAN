#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的训练脚本 - 支持 MiniMind 和 MiniKAN

支持参数:
  --model: 选择模型类型 (minimind 或 minikan)
  --steps: 训练步数
  --record_tps: 是否记录 Tokens Per Second

用法:
  python train.py --model minikan --steps 1000 --record_tps
  python train.py --model minimind --steps 1000 --record_tps
"""

import os
import sys
import json
import time
import argparse
import math
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入 MiniKAN
from model import MiniKANConfig, MiniKANForCausalLM

# 尝试导入 MiniMind
try:
    sys.path.insert(0, '/d/project_2026/miniKAN')
    from minimind.model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    HAS_MINIMIND = True
except ImportError:
    HAS_MINIMIND = False
    print("警告：无法导入 MiniMind，仅支持 MiniKAN")


def create_dummy_dataloader(batch_size=4, seq_len=512, vocab_size=6400, num_batches=1000):
    """
    创建虚拟数据集用于测试
    
    实际使用时应该替换为真实数据加载
    """
    data = []
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((input_ids, labels))
    return data


def get_model(model_type, dim=512, num_layers=8, num_heads=8):
    """根据类型创建模型"""
    vocab_size = 6400
    
    if model_type == 'minikan':
        config = MiniKANConfig(
            hidden_size=dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_grids=5,
            vocab_size=vocab_size,
            spline_weight_init_scale=0.1,
        )
        model = MiniKANForCausalLM(config)
        model_name = "MiniKAN"
        
    elif model_type == 'minimind':
        if not HAS_MINIMIND:
            raise ValueError("MiniMind 不可用")
        
        intermediate_size = int(dim * 8 / 3)
        intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        config = MiniMindConfig(
            hidden_size=dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )
        model = MiniMindForCausalLM(config)
        model_name = "MiniMind"
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model, config, model_name


def train_model(model, dataloader, steps, learning_rate=5e-4, record_tps=False, device='cuda'):
    """
    训练模型
    
    Returns:
        history: 包含 loss 和 tps 的字典列表
    """
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = GradScaler()
    
    history = []
    global_step = 0
    
    pbar = tqdm(total=steps, desc="Training")
    
    while global_step < steps:
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if global_step >= steps:
                break
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # 记录开始时间
            if record_tps:
                torch.cuda.synchronize() if device == 'cuda' else None
                start_time = time.time()
            
            # 前向传播
            with autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                if hasattr(outputs, 'aux_loss'):
                    loss = loss + outputs.aux_loss
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # 计算 TPS
            if record_tps:
                torch.cuda.synchronize() if device == 'cuda' else None
                end_time = time.time()
                elapsed = end_time - start_time
                batch_size, seq_len = input_ids.shape
                tokens_processed = batch_size * seq_len
                tps = tokens_processed / elapsed if elapsed > 0 else 0
            else:
                tps = 0
            
            # 记录历史
            history.append({
                'step': global_step,
                'loss': loss.item(),
                'tps': tps,
            })
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'tps': f'{tps:.0f}' if record_tps else 'N/A'
            })
            
            global_step += 1
    
    pbar.close()
    return history


def save_results(history, model_name, output_dir='results'):
    """保存训练结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump({
            'model': model_name,
            'history': history,
            'final_loss': history[-1]['loss'] if history else None,
            'avg_tps': sum(h['tps'] for h in history) / len(history) if history else 0,
        }, f, indent=2)
    
    print(f"\n结果已保存到: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='训练 MiniMind 或 MiniKAN')
    
    # 模型选择
    parser.add_argument('--model', type=str, required=True, 
                       choices=['minimind', 'minikan'],
                       help='选择模型类型: minimind 或 minikan')
    
    # 模型参数
    parser.add_argument('--dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=8, help='层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_grids', type=int, default=5, help='MiniKAN 网格数')
    parser.add_argument('--init_scale', type=float, default=0.1, help='KAN 初始化权重比例')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=1000, help='训练步数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=512, help='序列长度')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 监控参数
    parser.add_argument('--record_tps', action='store_true', 
                       help='是否记录 Tokens Per Second')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='训练设备')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    print("\n" + "="*80)
    print(f" 训练配置 ".center(80, "="))
    print("="*80)
    print(f"模型类型: {args.model}")
    print(f"模型参数: dim={args.dim}, layers={args.num_layers}, heads={args.num_heads}")
    if args.model == 'minikan':
        print(f"KAN 参数: num_grids={args.num_grids}, init_scale={args.init_scale}")
    print(f"训练参数: steps={args.steps}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"监控参数: record_tps={args.record_tps}, device={args.device}")
    print("="*80 + "\n")
    
    # 创建模型
    print(f"创建 {args.model} 模型...")
    model, config, model_name = get_model(
        args.model, 
        dim=args.dim, 
        num_layers=args.num_layers, 
        num_heads=args.num_heads
    )
    
    # 对于 MiniKAN，设置初始化比例
    if args.model == 'minikan':
        config.num_grids = args.num_grids
        config.spline_weight_init_scale = args.init_scale
        print(f"\n⚠️  KAN 初始化权重比例设置为: {args.init_scale}")
        print("    提示: 如果出现梯度爆炸，请减小 init_scale (建议 0.05-0.1)")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)\n")
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_dummy_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
        num_batches=(args.steps // 10) + 1  # 循环使用
    )
    
    # 训练
    print(f"\n开始训练 {args.model}...")
    history = train_model(
        model, 
        dataloader, 
        steps=args.steps,
        learning_rate=args.learning_rate,
        record_tps=args.record_tps,
        device=args.device
    )
    
    # 保存结果
    result_file = save_results(history, model_name, args.output_dir)
    
    # 打印总结
    print("\n" + "="*80)
    print(f" 训练完成 - {model_name} ".center(80, "="))
    print("="*80)
    print(f"初始 Loss: {history[0]['loss']:.4f}")
    print(f"最终 Loss: {history[-1]['loss']:.4f}")
    if args.record_tps:
        avg_tps = sum(h['tps'] for h in history) / len(history)
        print(f"平均 TPS: {avg_tps:.0f} tokens/sec")
    print(f"结果文件: {result_file}")
    print("="*80)


if __name__ == '__main__':
    main()
