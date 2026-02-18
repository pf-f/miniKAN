#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniKAN 预训练脚本

独立实现，基于 FastKAN 架构的 LLM 预训练
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from datetime import datetime

from model import MiniKANConfig, MiniKANForCausalLM


def get_lr(current_step, total_steps, lr_max, warmup_steps=2000):
    """带 warmup 的余弦退火学习率调度"""
    if current_step < warmup_steps:
        return lr_max * current_step / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return lr_max * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, total_steps, args):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度训练
        with autocast(enabled=args.use_amp):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if hasattr(outputs, 'aux_loss'):
                loss = loss + outputs.aux_loss
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新学习率
            current_step = epoch * len(dataloader) + step
            lr = get_lr(current_step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='MiniKAN Pretraining')
    
    # 模型参数
    parser.add_argument('--dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_kv_heads', type=int, default=2, help='Number of key/value heads')
    parser.add_argument('--vocab_size', type=int, default=6400, help='Vocabulary size')
    parser.add_argument('--num_grids', type=int, default=5, help='FastKAN grid number')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/pretrain.jsonl', help='Training data path')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建配置
    config = MiniKANConfig(
        hidden_size=args.dim,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        vocab_size=args.vocab_size,
        num_grids=args.num_grids,
        max_position_embeddings=args.max_seq_len,
    )
    
    # 创建模型
    print("Creating model...")
    model = MiniKANForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # 混合精度训练
    scaler = GradScaler(enabled=args.use_amp)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模拟数据集（实际使用时需要替换为真实数据加载）
    print("\nNote: This is a template training script.")
    print("You need to implement the actual data loading logic.")
    print("\nExample data format:")
    print("  - JSONL file with 'text' field")
    print("  - Tokenized and padded to max_seq_len")
    
    # 保存模型配置
    config.save_pretrained(args.output_dir)
    print(f"\nConfig saved to {args.output_dir}")
    
    print("\nTo use this script:")
    print("1. Implement your data loading in trainer/dataset.py")
    print("2. Run: python trainer/train_pretrain.py --data_path your_data.jsonl")


if __name__ == '__main__':
    main()
