"""
测试基于迁移行为的初始化方法
"""
import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.estimation.migration_behavior_analysis import (
    extract_migration_history_features, 
    identify_migration_behavior_types, 
    create_behavior_based_initial_params,
    classify_individual_types
)
from src.data_handler.data_loader import DataLoader
from src.config.model_config import ModelConfig


def test_migration_behavior_analysis():
    """
    测试迁移行为分析功能
    """
    print("开始测试迁移行为分析功能...")
    
    # 加载数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    try:
        # 加载数据
        df_individual, df_region, state_space, transition_matrices = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
        
        print(f"加载数据成功: {len(df_individual)} 个观测")
        print(f"个体数量: {df_individual['individual_id'].nunique()}")
        
        # 测试特征提取
        print("\n1. 测试特征提取...")
        features = extract_migration_history_features(df_individual)
        print(f"提取了 {len(features)} 个个体的特征")
        print("特征列:", list(features.columns))
        
        # 测试类型识别
        print("\n2. 测试类型识别...")
        n_types = 3
        type_assignments, type_probs_matrix = identify_migration_behavior_types(
            df_individual, n_types
        )
        print(f"类型分配形状: {type_probs_matrix.shape}")
        print(f"类型概率范围: [{type_probs_matrix.min():.4f}, {type_probs_matrix.max():.4f}]")
        
        # 检查每种类型的概率
        avg_type_probs = type_probs_matrix.mean(axis=0)
        print(f"各类型平均概率: {avg_type_probs}")
        
        # 测试分类函数
        print("\n3. 测试分类函数...")
        classification_results = classify_individual_types(df_individual, type_probs_matrix)
        print(f"分配的类型数量: {len(classification_results['assigned_types'])}")
        print(f"类型概率分布: {classification_results['type_probabilities']}")
        print(f"平均置信度: {classification_results['avg_confidence']:.4f}")
        
        # 测试参数创建
        print("\n4. 测试参数创建...")
        initial_params = create_behavior_based_initial_params(n_types)
        print("初始参数包含的键:", [k for k in initial_params.keys() if 'type' in k])
        
        print("\n✓ 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_migration_behavior_analysis()
    if success:
        print("\n迁移行为分析模块测试成功！")
    else:
        print("\n迁移行为分析模块测试失败！")
        sys.exit(1)