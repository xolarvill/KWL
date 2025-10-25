"""
测试参数打包和解包过程
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.config.model_config import ModelConfig
from src.estimation.em_with_omega import _pack_params, _unpack_params

def test_pack_unpack():
    """测试参数打包和解包"""
    print("测试参数打包和解包过程...")
    
    # 获取初始参数
    config = ModelConfig()
    initial_params = config.get_initial_params(use_type_specific=True)
    
    print("初始参数示例:")
    key_params = ['alpha_w', 'gamma_1', 'gamma_0_type_1', 'gamma_0_type_2']
    for param in key_params:
        if param in initial_params:
            print(f"  {param}: {initial_params[param]}")
    
    # 打包参数
    param_values, param_names = _pack_params(initial_params)
    print(f"\n打包后参数数量: {len(param_values)}")
    print("前10个参数名称:", param_names[:10])
    print("前10个参数值:", param_values[:10])
    
    # 解包参数
    unpacked_params = _unpack_params(param_values, param_names, initial_params['n_choices'])
    
    # 验证关键参数
    print("\n验证解包后的参数:")
    for param in key_params:
        if param in initial_params and param in unpacked_params:
            original = initial_params[param]
            unpacked = unpacked_params[param]
            match = abs(original - unpacked) < 1e-10
            status = "✓" if match else "✗"
            print(f"  {param}: {original} -> {unpacked} {status}")
            
    # 检查是否有参数值为0.1的
    print(f"\n检查是否有异常的0.1值:")
    for i, (name, value) in enumerate(zip(param_names, param_values)):
        if abs(value - 0.1) < 1e-10:
            print(f"  参数 {name} 的值为 {value}")

if __name__ == '__main__':
    test_pack_unpack()