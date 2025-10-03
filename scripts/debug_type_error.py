"""
精确定位类型转换错误
"""
import sys
import os
import traceback
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.config.model_config import ModelConfig
from src.model.likelihood import calculate_log_likelihood

def test_single_likelihood():
    """测试单个似然计算"""
    print("精确定位类型转换错误...")
    
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    config.regional_data_path = config.regional_data_path.replace('geo.xlsx', 'geo_amenities.csv')
    df_region = data_loader.load_regional_data()
    df_estimation, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    
    # 使用极小样本
    df_sample = df_estimation.head(10)
    
    # 测试参数
    params = {
        "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0,
        "rho_base_tier_1": 1.0, "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1,
        "gamma_0_type_0": 1.0, "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2,
        "gamma_3": -0.4, "gamma_4": 0.01, "gamma_5": -0.05,
        "alpha_climate": 0.1, "alpha_health": 0.1, "alpha_education": 0.1, 
        "alpha_public_services": 0.1,
        "n_choices": 31
    }
    
    print(f"\n检查transition_matrices的键类型:")
    for key in list(transition_matrices.keys())[:5]:
        print(f"  键: {key}, 类型: {type(key)}")
    
    print(f"\n尝试计算似然值...")
    try:
        # 测试agent_type=0
        log_lik = calculate_log_likelihood(
            params=params,
            observed_data=df_sample,
            state_space=state_space,
            agent_type=0,  # 使用Python int
            beta=0.95,
            transition_matrices=transition_matrices,
            regions_df=df_region
        )
        print(f"✓ 成功计算似然值: {log_lik}")
        
    except TypeError as e:
        print(f"\n✗ TypeError: {e}")
        traceback.print_exc()
        
        # 打印更多调试信息
        print(f"\n调试信息:")
        print(f"  params类型: {type(params)}")
        print(f"  agent_type类型: {type(0)}")
        print(f"  beta类型: {type(0.95)}")
        
    except Exception as e:
        print(f"\n✗ 其他错误: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    test_single_likelihood()
