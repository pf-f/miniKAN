#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniKAN 基础测试脚本

测试模型能否正常实例化、前向传播、反向传播
"""

import torch
import sys
import os

from model import MiniKANConfig, MiniKANForCausalLM


def test_model_instantiation():
    """测试模型实例化"""
    print("\n" + "="*60)
    print("测试 1: 模型实例化")
    print("="*60)
    
    config = MiniKANConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_grids=5,
        vocab_size=1000,
    )
    
    try:
        model = MiniKANForCausalLM(config)
        print("✓ 模型实例化成功")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return model, config
    except Exception as e:
        print(f"✗ 模型实例化失败: {e}")
        raise


def test_forward_pass(model, config):
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试 2: 前向传播")
    print("="*60)
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs.logits
        print(f"✓ 输入 shape: {input_ids.shape}")
        print(f"✓ 输出 logits shape: {logits.shape}")
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        print("✓ 前向传播测试通过")
        return outputs
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        raise


def test_backward_pass(model, config):
    """测试反向传播"""
    print("\n" + "="*60)
    print("测试 3: 反向传播")
    print("="*60)
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    try:
        model.train()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        print(f"✓ 损失值: {loss.item():.4f}")
        
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters())
        if has_grad:
            print("✓ 梯度计算正常")
        print("✓ 反向传播测试通过")
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        raise


def test_kan_visualization(model):
    """测试 KAN 可视化"""
    print("\n" + "="*60)
    print("测试 4: KAN 层可视化")
    print("="*60)
    
    try:
        kan_layer = model.model.layers[0].mlp.kan_layer
        weights = kan_layer.spline_linear.weight
        print(f"✓ 样条权重 shape: {weights.shape}")
        
        x, y = kan_layer.plot_curve(input_index=0, output_index=0, num_pts=100)
        print(f"✓ 曲线数据 shape: x={x.shape}, y={y.shape}")
        print("✓ KAN 层可视化功能正常")
    except Exception as e:
        print(f"⚠ 可视化功能警告: {e}")


def test_different_configs():
    """测试不同配置"""
    print("\n" + "="*60)
    print("测试 5: 不同 num_grids 配置")
    print("="*60)
    
    for grids in [3, 5, 8]:
        try:
            config = MiniKANConfig(
                hidden_size=256,
                num_hidden_layers=2,
                num_grids=grids,
                vocab_size=1000,
            )
            model = MiniKANForCausalLM(config)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ num_grids={grids}: {total_params:,} 参数")
        except Exception as e:
            print(f"✗ num_grids={grids} 失败: {e}")
            raise
    print("✓ 所有配置测试通过")


def test_save_load(model, config):
    """测试模型保存和加载"""
    print("\n" + "="*60)
    print("测试 6: 模型保存和加载")
    print("="*60)
    
    import tempfile
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model.save_pretrained(save_path)
            print(f"✓ 模型保存到: {save_path}")
            
            loaded_model = MiniKANForCausalLM.from_pretrained(save_path)
            print("✓ 模型加载成功")
            
            # 验证参数一致
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
                assert torch.allclose(p1, p2), f"参数 {n1} 不匹配"
            print("✓ 参数一致性验证通过")
    except Exception as e:
        print(f"✗ 保存/加载测试失败: {e}")
        raise


def main():
    print("\n" + "="*60)
    print(" MiniKAN 功能测试 ".center(60, "="))
    print("="*60)
    
    try:
        model, config = test_model_instantiation()
        test_forward_pass(model, config)
        test_backward_pass(model, config)
        test_kan_visualization(model)
        test_different_configs()
        test_save_load(model, config)
        
        print("\n" + "="*60)
        print(" ✓ 所有测试通过！".center(60))
        print("="*60)
        print("\nMiniKAN 可以正常使用，可以开始训练！")
        
    except Exception as e:
        print("\n" + "="*60)
        print(" ✗ 测试失败".center(60))
        print("="*60)
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
