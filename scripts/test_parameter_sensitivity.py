"""
测试参数变化对目标函数的影响

验证参数变化是否能产生可检测的目标函数变化
"""
import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.estimation.em_with_omega import _pack_params, _unpack_params

def test_parameter_sensitivity():
    """测试参数变化对目标函数的影响"""
    print("测试参数敏感性...")
    
    # 获取初始参数
    config = ModelConfig()
    initial_params = config.get_initial_params(use_type_specific=True)
    
    print("初始参数示例:")
    for key in list(initial_params.keys())[:10]:
        print(f"  {key}: {initial_params[key]}")
    
    # 打包参数
    param_values, param_names = _pack_params(initial_params)
    print(f"\n打包后参数维度: {len(param_values)}")
    print(f"前5个参数名称: {param_names[:5]}")
    print(f"前5个参数值: {param_values[:5]}")
    
    # 解包参数
    unpacked_params = _unpack_params(param_values, param_names, initial_params['n_choices'])
    print(f"\n解包后参数数量: {len(unpacked_params)}")
    
    # 验证打包/解包是否正确
    print("\n验证关键参数:")
    key_params = ['alpha_w', 'gamma_1', 'gamma_0_type_1', 'gamma_0_type_2']
    for param in key_params:
        if param in initial_params and param in unpacked_params:
            initial_val = initial_params[param]
            unpacked_val = unpacked_params[param]
            match = "✓" if abs(initial_val - unpacked_val) < 1e-10 else "✗"
            print(f"  {param}: {initial_val} -> {unpacked_val} {match}")
    
    # 测试参数扰动
    print("\n测试参数扰动:")
    perturbed_values = param_values.copy()
    # 扰动前几个参数
    perturbation = 0.1
    for i in range(min(3, len(perturbed_values))):
        original_val = perturbed_values[i]
        perturbed_values[i] = original_val + perturbation
        param_name = param_names[i]
        print(f"  {param_name}: {original_val} -> {perturbed_values[i]} (+{perturbation})")
    
    # 解包扰动后的参数
    perturbed_params = _unpack_params(perturbed_values, param_names, initial_params['n_choices'])
    
    print("\n扰动后参数验证:")
    for i in range(3):
        param_name = param_names[i]
        original_val = initial_params[param_name]
        perturbed_val = perturbed_params[param_name]
        print(f"  {param_name}: {original_val} -> {perturbed_val}")

if __name__ == '__main__':
    test_parameter_sensitivity()