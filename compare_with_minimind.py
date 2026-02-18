"""
MiniKAN2 对比脚本示例

展示如何与原始 MiniMind 进行对比
"""

import sys
import os

# 假设 minimind 项目在同一目录下
sys.path.insert(0, '/d/project_2026/miniKAN')  # 原始 minimind

from model import MiniKANConfig, MiniKANForCausalLM

# 这里需要导入 minimind 的模型
try:
    from minimind.model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    CAN_COMPARE = True
except ImportError:
    print("Warning: MiniMind not found, comparison disabled")
    CAN_COMPARE = False


def compare_parameters():
    """对比参数量"""
    if not CAN_COMPARE:
        print("MiniMind not available, showing MiniKAN only")
    
    dim = 512
    num_layers = 8
    
    # MiniKAN 配置
    config_kan = MiniKANConfig(
        hidden_size=dim,
        num_hidden_layers=num_layers,
        num_grids=5,
    )
    model_kan = MiniKANForCausalLM(config_kan)
    params_kan = sum(p.numel() for p in model_kan.parameters())
    
    print(f"\nMiniKAN ({dim}d, {num_layers}layers, grids=5):")
    print(f"  Parameters: {params_kan:,} ({params_kan/1e6:.2f}M)")
    
    if CAN_COMPARE:
        # MiniMind 配置
        intermediate = int(dim * 8 / 3)
        intermediate = 64 * ((intermediate + 64 - 1) // 64)
        
        config_mind = MiniMindConfig(
            hidden_size=dim,
            num_hidden_layers=num_layers,
            intermediate_size=intermediate,
        )
        model_mind = MiniMindForCausalLM(config_mind)
        params_mind = sum(p.numel() for p in model_mind.parameters())
        
        print(f"\nMiniMind ({dim}d, {num_layers}layers, intermediate={intermediate}):")
        print(f"  Parameters: {params_mind:,} ({params_mind/1e6:.2f}M)")
        
        print(f"\n对比:")
        print(f"  MiniKAN / MiniMind: {params_kan/params_mind*100:.1f}%")
        print(f"  Saved: {(1-params_kan/params_mind)*100:.1f}%")


if __name__ == '__main__':
    compare_parameters()
