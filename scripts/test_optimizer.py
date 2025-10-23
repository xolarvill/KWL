"""
验证EM算法正确性的简单测试

测试EM算法是否能正确更新参数
"""
import sys
import os
import numpy as np
from scipy.optimize import minimize

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 简单的测试函数
def simple_objective(params):
    """
    简单的测试目标函数
    这个函数应该有一个明显的最小值点
    """
    x, y = params
    # 简单的二次函数，最小值在(1, 2)
    return (x - 1)**2 + (y - 2)**2

def test_optimization():
    """测试优化器是否正常工作"""
    print("测试优化器...")
    
    # 初始参数
    initial_params = np.array([0.0, 0.0])
    print(f"初始参数: {initial_params}")
    
    # 初始目标函数值
    initial_value = simple_objective(initial_params)
    print(f"初始目标函数值: {initial_value}")
    
    # 优化
    result = minimize(
        simple_objective,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': 10}
    )
    
    print(f"优化结果: {result}")
    print(f"最终参数: {result.x}")
    print(f"最终目标函数值: {result.fun}")
    
    # 检查是否真的优化了
    if result.fun < initial_value:
        print("✓ 优化器正常工作")
    else:
        print("✗ 优化器未正常工作")

if __name__ == '__main__':
    test_optimization()