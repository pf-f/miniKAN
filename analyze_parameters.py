#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的参数量对比分析脚本

对比 MiniMind 和 MiniKAN 的参数量和每层分布
"""

import sys
import os

# 添加 miniKAN2 到路径
sys.path.insert(0, os.path.dirname(__file__))

import torch
from model import MiniKANConfig, MiniKANForCausalLM

# 尝试导入 minimind
try:
    sys.path.insert(0, '/d/project_2026/miniKAN')
    from minimind.model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    HAS_MINIMIND = True
except ImportError:
    HAS_MINIMIND = False
    print("警告：无法导入 MiniMind，仅显示 MiniKAN 信息")


def analyze_model_layers(model, model_name):
    """
    详细分析模型每层的参数量
    
    Returns:
        layer_info: 每层的信息列表
        total_params: 总参数量
    """
    layer_info = []
    total_params = 0
    
    print(f"\n{'='*80}")
    print(f" {model_name} 详细参数分析 ")
    print(f"{'='*80}")
    print(f"{'层名':<50} {'参数量':>15} {'占比':>10}")
    print(f"{'-'*80}")
    
    # 按模块分组统计
    current_block = None
    block_params = 0
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        
        # 识别层类型
        if 'embed_tokens' in name or 'lm_head' in name:
            layer_type = 'Embedding'
        elif 'self_attn' in name:
            layer_type = 'Attention'
        elif 'mlp' in name or 'feed_forward' in name:
            layer_type = 'FFN/KAN'
        elif 'norm' in name:
            layer_type = 'Norm'
        else:
            layer_type = 'Other'
        
        layer_info.append({
            'name': name,
            'params': params,
            'type': layer_type,
            'shape': list(param.shape)
        })
        
        print(f"{name:<50} {params:>15,} {params/sum(p.numel() for p in model.parameters())*100:>9.2f}%")
    
    print(f"{'-'*80}")
    print(f"{'总计':<50} {total_params:>15,} {'100.00%':>10}")
    print(f"{'='*80}")
    
    return layer_info, total_params


def compare_by_component():
    """按组件对比参数量"""
    print("\n" + "="*80)
    print(" 组件级别参数量对比 ".center(80, "="))
    print("="*80)
    
    dim = 512
    num_layers = 8
    num_heads = 8
    
    # MiniKAN
    config_kan = MiniKANConfig(
        hidden_size=dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_grids=5,
        vocab_size=6400,
    )
    model_kan = MiniKANForCausalLM(config_kan)
    
    # 统计 MiniKAN 各组件
    kan_components = {
        'Embedding': 0,
        'Attention': 0,
        'KAN_FFN': 0,
        'Norm': 0,
        'Other': 0
    }
    
    for name, param in model_kan.named_parameters():
        params = param.numel()
        if 'embed' in name or 'lm_head' in name:
            kan_components['Embedding'] += params
        elif 'self_attn' in name:
            kan_components['Attention'] += params
        elif 'mlp' in name:
            kan_components['KAN_FFN'] += params
        elif 'norm' in name:
            kan_components['Norm'] += params
        else:
            kan_components['Other'] += params
    
    print(f"\nMiniKAN (dim={dim}, layers={num_layers}, grids=5):")
    for comp, params in kan_components.items():
        print(f"  {comp:<15}: {params:>12,} ({params/sum(kan_components.values())*100:>5.2f}%)")
    
    if HAS_MINIMIND:
        # MiniMind
        intermediate_size = int(dim * 8 / 3)
        intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        config_mind = MiniMindConfig(
            hidden_size=dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            vocab_size=6400,
        )
        model_mind = MiniMindForCausalLM(config_mind)
        
        # 统计 MiniMind 各组件
        mind_components = {
            'Embedding': 0,
            'Attention': 0,
            'MLP_FFN': 0,
            'Norm': 0,
            'Other': 0
        }
        
        for name, param in model_mind.named_parameters():
            params = param.numel()
            if 'embed' in name or 'lm_head' in name:
                mind_components['Embedding'] += params
            elif 'self_attn' in name:
                mind_components['Attention'] += params
            elif 'mlp' in name:
                mind_components['MLP_FFN'] += params
            elif 'norm' in name:
                mind_components['Norm'] += params
            else:
                mind_components['Other'] += params
        
        print(f"\nMiniMind (dim={dim}, layers={num_layers}, intermediate={intermediate_size}):")
        for comp, params in mind_components.items():
            print(f"  {comp:<15}: {params:>12,} ({params/sum(mind_components.values())*100:>5.2f}%)")
        
        # 对比
        print(f"\n对比分析:")
        print(f"  {'组件':<15} {'MiniMind':>12} {'MiniKAN':>12} {'差异':>12} {'比例':>10}")
        print(f"  {'-'*65}")
        
        all_components = set(kan_components.keys()) | set(mind_components.keys())
        for comp in sorted(all_components):
            mind_p = mind_components.get(comp, 0)
            kan_p = kan_components.get(comp, 0)
            diff = kan_p - mind_p
            ratio = kan_p / mind_p * 100 if mind_p > 0 else 0
            print(f"  {comp:<15} {mind_p:>12,} {kan_p:>12,} {diff:>12,} {ratio:>9.1f}%")
        
        total_mind = sum(mind_components.values())
        total_kan = sum(kan_components.values())
        total_diff = total_kan - total_mind
        total_ratio = total_kan / total_mind * 100
        
        print(f"  {'-'*65}")
        print(f"  {'总计':<15} {total_mind:>12,} {total_kan:>12,} {total_diff:>12,} {total_ratio:>9.1f}%")


def compare_init_scales():
    """对比不同初始化比例的影响"""
    print("\n" + "="*80)
    print(" 初始化权重敏感性分析 ".center(80, "="))
    print("="*80)
    print("\nKAN 网络对初始化非常敏感。测试不同 init_scale 的权重分布：")
    
    config = MiniKANConfig(hidden_size=512, num_hidden_layers=2, num_grids=5)
    
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        config.spline_weight_init_scale = scale
        model = MiniKANForCausalLM(config)
        
        # 获取 KAN 层的权重统计
        kan_weights = []
        for name, param in model.named_parameters():
            if 'spline_linear' in name:
                kan_weights.append(param.data)
        
        if kan_weights:
            all_weights = torch.cat([w.flatten() for w in kan_weights])
            mean = all_weights.mean().item()
            std = all_weights.std().item()
            max_val = all_weights.max().item()
            min_val = all_weights.min().item()
            
            print(f"\n  init_scale = {scale}:")
            print(f"    权重分布: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")
            
            # 检查梯度爆炸风险
            if max_val > 10 or min_val < -10:
                print(f"    ⚠️ 警告: 权重范围过大，可能导致梯度爆炸！")
            elif std > 1:
                print(f"    ⚠️ 警告: 标准差过大，建议减小 init_scale")
            else:
                print(f"    ✅ 权重分布正常")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" MiniMind vs MiniKAN 详细参数量对比 ".center(80, "="))
    print("="*80)
    
    # 1. 详细的层级别对比
    if HAS_MINIMIND:
        config_mind = MiniMindConfig(
            hidden_size=512,
            num_hidden_layers=8,
            intermediate_size=1408,
            vocab_size=6400,
        )
        model_mind = MiniMindForCausalLM(config_mind)
        analyze_model_layers(model_mind, "MiniMind")
    
    config_kan = MiniKANConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_grids=5,
        vocab_size=6400,
    )
    model_kan = MiniKANForCausalLM(config_kan)
    analyze_model_layers(model_kan, "MiniKAN")
    
    # 2. 组件级别对比
    compare_by_component()
    
    # 3. 初始化权重分析
    compare_init_scales()
    
    print("\n" + "="*80)
    print("分析完成！".center(80))
    print("="*80)
    print("\n建议:")
    print("  1. KAN 的 init_scale 建议使用 0.05-0.1 之间")
    print("  2. 如果训练时出现梯度爆炸，请减小 spline_weight_init_scale")
    print("  3. MiniKAN 的 FFN 参数量约为 MiniMind 的 70-75%")


if __name__ == '__main__':
    main()
