#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化对比脚本 - 绘制 MiniMind 和 MiniKAN 的 loss 和 TPS 对比图

用法:
  python visualize_comparison.py --results_dir results
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("错误：需要安装 matplotlib")
    print("请运行: pip install matplotlib")
    sys.exit(1)


def load_results(results_dir):
    """加载训练结果文件"""
    results = {}
    
    # 查找所有结果文件
    pattern = os.path.join(results_dir, '*.json')
    files = glob.glob(pattern)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
            model_name = data.get('model', 'Unknown')
            results[model_name] = data
    
    return results


def plot_loss_comparison(results, output_path):
    """绘制 Loss 对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'MiniMind': '#2E86AB',
        'MiniKAN': '#A23B72'
    }
    
    for model_name, data in results.items():
        history = data.get('history', [])
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        
        color = colors.get(model_name, '#333333')
        ax.plot(steps, losses, label=model_name, color=color, linewidth=2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison: MiniMind vs MiniKAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss 对比图已保存: {output_path}")
    plt.close()


def plot_tps_comparison(results, output_path):
    """绘制 TPS (Tokens Per Second) 对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'MiniMind': '#2E86AB',
        'MiniKAN': '#A23B72'
    }
    
    for model_name, data in results.items():
        history = data.get('history', [])
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        tps_values = [h.get('tps', 0) for h in history]
        
        # 过滤掉 0 值
        valid_data = [(s, t) for s, t in zip(steps, tps_values) if t > 0]
        if not valid_data:
            continue
        
        steps_clean, tps_clean = zip(*valid_data)
        
        color = colors.get(model_name, '#333333')
        ax.plot(steps_clean, tps_clean, label=model_name, color=color, linewidth=2, alpha=0.7)
        
        # 计算平均 TPS
        avg_tps = sum(tps_clean) / len(tps_clean)
        ax.axhline(y=avg_tps, color=color, linestyle='--', alpha=0.5, 
                  label=f'{model_name} Avg: {avg_tps:.0f}')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Tokens Per Second (TPS)', fontsize=12)
    ax.set_title('Training Speed Comparison: MiniMind vs MiniKAN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"TPS 对比图已保存: {output_path}")
    plt.close()


def plot_combined_comparison(results, output_path):
    """绘制组合对比图 (Loss + TPS)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'MiniMind': '#2E86AB',
        'MiniKAN': '#A23B72'
    }
    
    # 左图: Loss
    for model_name, data in results.items():
        history = data.get('history', [])
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        
        color = colors.get(model_name, '#333333')
        ax1.plot(steps, losses, label=model_name, color=color, linewidth=2)
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图: TPS
    for model_name, data in results.items():
        history = data.get('history', [])
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        tps_values = [h.get('tps', 0) for h in history]
        
        valid_data = [(s, t) for s, t in zip(steps, tps_values) if t > 0]
        if not valid_data:
            continue
        
        steps_clean, tps_clean = zip(*valid_data)
        
        color = colors.get(model_name, '#333333')
        ax2.plot(steps_clean, tps_clean, label=model_name, color=color, linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Tokens Per Second', fontsize=12)
    ax2.set_title('Training Speed (TPS)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('MiniMind vs MiniKAN: Training Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"组合对比图已保存: {output_path}")
    plt.close()


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print(" 实验结果摘要 ".center(80, "="))
    print("="*80)
    
    for model_name, data in results.items():
        history = data.get('history', [])
        if not history:
            continue
        
        print(f"\n{model_name}:")
        print(f"  初始 Loss: {history[0]['loss']:.4f}")
        print(f"  最终 Loss: {history[-1]['loss']:.4f}")
        print(f"  Loss 下降: {history[0]['loss'] - history[-1]['loss']:.4f}")
        
        tps_values = [h.get('tps', 0) for h in history if h.get('tps', 0) > 0]
        if tps_values:
            avg_tps = sum(tps_values) / len(tps_values)
            print(f"  平均 TPS: {avg_tps:.0f} tokens/sec")
    
    # 对比分析
    if len(results) == 2:
        print("\n" + "-"*80)
        print("对比分析:")
        
        model_names = list(results.keys())
        data1 = results[model_names[0]]
        data2 = results[model_names[1]]
        
        history1 = data1.get('history', [])
        history2 = data2.get('history', [])
        
        if history1 and history2:
            loss_diff = history2[-1]['loss'] - history1[-1]['loss']
            print(f"  Loss 差异: {model_names[1]} 比 {model_names[0]} {'高' if loss_diff > 0 else '低'} {abs(loss_diff):.4f}")
        
        tps1 = [h.get('tps', 0) for h in history1 if h.get('tps', 0) > 0]
        tps2 = [h.get('tps', 0) for h in history2 if h.get('tps', 0) > 0]
        
        if tps1 and tps2:
            avg_tps1 = sum(tps1) / len(tps1)
            avg_tps2 = sum(tps2) / len(tps2)
            tps_diff = avg_tps2 - avg_tps1
            speed_ratio = avg_tps2 / avg_tps1 if avg_tps1 > 0 else 0
            print(f"  速度差异: {model_names[1]} 比 {model_names[0]} {'快' if tps_diff > 0 else '慢'} {abs(tps_diff):.0f} TPS ({speed_ratio:.2f}x)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='可视化 MiniMind vs MiniKAN 对比结果')
    parser.add_argument('--results_dir', type=str, default='results', help='结果文件目录')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" 加载训练结果... ".center(80, "="))
    print("="*80)
    
    # 加载结果
    results = load_results(args.results_dir)
    
    if not results:
        print(f"错误: 在 {args.results_dir} 目录下未找到结果文件")
        return
    
    print(f"找到 {len(results)} 个结果文件:")
    for model_name in results.keys():
        print(f"  - {model_name}")
    
    # 创建输出目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 生成图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plot_loss_comparison(results, os.path.join(args.results_dir, f'loss_comparison_{timestamp}.png'))
    plot_tps_comparison(results, os.path.join(args.results_dir, f'tps_comparison_{timestamp}.png'))
    plot_combined_comparison(results, os.path.join(args.results_dir, f'combined_comparison_{timestamp}.png'))
    
    # 打印摘要
    print_summary(results)
    
    print(f"\n所有图表已保存到: {args.results_dir}/")


if __name__ == '__main__':
    main()
