"""
检查数据结构
"""
import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.config.model_config import ModelConfig


def check_data_structure():
    """
    检查数据结构
    """
    print("检查数据结构...")
    
    # 加载数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    try:
        # 加载数据
        df_individual, df_region, state_space, transition_matrices = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
        
        print(f"个体数据形状: {df_individual.shape}")
        print(f"个体数据列: {list(df_individual.columns)}")
        print("\n前5行数据:")
        print(df_individual.head())
        
        print(f"\n区域数据形状: {df_region.shape}")
        print(f"区域数据列: {list(df_region.columns)}")
        
        return df_individual, df_region
        
    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    df_ind, df_reg = check_data_structure()