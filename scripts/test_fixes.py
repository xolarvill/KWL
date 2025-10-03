"""
测试修复效果的诊断脚本
"""
import sys
import os
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.estimation.em_nfxp import run_em_algorithm
from src.config.model_config import ModelConfig

def test_data_loading():
    """测试数据加载和丢失问题"""
    print("="*80)
    print("测试1: 数据加载")
    print("="*80)
    
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    # 加载个体数据
    df_individual = data_loader.load_individual_data()
    print(f"✓ 个体数据加载成功: {len(df_individual)} 条观测")
    
    # 创建估计数据集
    config.regional_data_path = config.regional_data_path.replace('geo.xlsx', 'geo_amenities.csv')
    df_estimation, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    
    print(f"✓ 估计数据集创建成功: {len(df_estimation)} 条观测")
    print(f"✓ 状态空间大小: {len(state_space)}")
    
    # 检查关键列
    print(f"\n关键列检查:")
    print(f"  - state_index范围: {df_estimation['state_index'].min()} - {df_estimation['state_index'].max()}")
    print(f"  - choice_index范围: {df_estimation['choice_index'].min()} - {df_estimation['choice_index'].max()}")
    print(f"  - 唯一个体数: {df_estimation['individual_id'].nunique()}")
    
    # 检查工资数据
    if 'wage_predicted' in df_estimation.columns:
        wage_stats = df_estimation['wage_predicted'].describe()
        print(f"\n工资预测统计:")
        print(f"  - 均值: {wage_stats['mean']:.2f}")
        print(f"  - 最小值: {wage_stats['min']:.2f}")
        print(f"  - 最大值: {wage_stats['max']:.2f}")
        print(f"  - 零值数量: {(df_estimation['wage_predicted'] == 0).sum()}")
    
    return df_estimation, state_space, transition_matrices

def test_estimation():
    """测试估计过程"""
    print("\n" + "="*80)
    print("测试2: 运行估计（2次迭代）")
    print("="*80)
    
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    config.regional_data_path = config.regional_data_path.replace('geo.xlsx', 'geo_amenities.csv')
    df_region = data_loader.load_regional_data()
    df_estimation, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    
    # 使用小样本测试
    sample_size = min(1000, len(df_estimation))
    df_sample = df_estimation.head(sample_size)
    
    print(f"\n使用 {sample_size} 条观测进行测试...")
    
    estimation_params = {
        "observed_data": df_sample,
        "state_space": state_space,
        "transition_matrices": transition_matrices,
        "beta": config.discount_factor,
        "n_types": 2,
        "max_iterations": 2,
        "tolerance": config.tolerance,
        "n_choices": len(transition_matrices),
        "regions_df": df_region,
    }
    
    try:
        results = run_em_algorithm(**estimation_params)
        
        print("\n估计结果:")
        print(f"  ✓ 对数似然值: {results['final_log_likelihood']:.4f}")
        print(f"  ✓ 迭代次数: {results['n_iterations']}")
        print(f"  ✓ 类型概率: {results['type_probabilities']}")
        
        # 检查参数估计
        print(f"\n参数估计值:")
        for param, value in results['structural_params'].items():
            if param != 'n_choices':
                print(f"  - {param}: {value:.4f}")
        
        # 检查是否所有值都异常相似
        param_values = [v for k, v in results['structural_params'].items() if k != 'n_choices']
        if len(set([round(v, 2) for v in param_values])) < 3:
            print("\n⚠️  警告: 参数估计值过于相似，可能存在问题")
        else:
            print("\n✓ 参数估计值有差异，看起来正常")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 估计失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 测试1: 数据加载
    df_estimation, state_space, transition_matrices = test_data_loading()
    
    # 测试2: 估计
    success = test_estimation()
    
    print("\n" + "="*80)
    if success:
        print("所有测试通过！修复生效。")
    else:
        print("测试失败，仍需进一步调试。")
    print("="*80)
