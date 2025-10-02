"""
测试脚本：验证数值稳定性修复
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.estimation.em_nfxp import run_em_algorithm
from src.config.model_config import ModelConfig

# 忽略除零警告以便我们可以捕获它们
warnings.filterwarnings('error', category=RuntimeWarning)

def test_numerical_stability():
    """测试数值稳定性修复"""
    print("="*80)
    print("测试数值稳定性修复")
    print("="*80)
    
    try:
        # 1. 配置
        config = ModelConfig()
        print("\n[1/5] 配置加载成功")
        
        # 2. 数据加载
        print("\n[2/5] 开始加载数据...")
        data_loader = DataLoader(config)
        
        # 加载预处理后的数据
        wage_pred_path = os.path.join(config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
        if os.path.exists(wage_pred_path):
            print("  使用LightGBM预测的工资数据")
            df_individual = pd.read_csv(wage_pred_path)
        else:
            print("  警告：未找到工资预测数据，使用原始数据")
            df_individual = data_loader.load_individual_data()
        
        # 加载地区数据
        config.regional_data_path = config.regional_data_path.replace('geo.xlsx', 'geo_amenities.csv')
        df_region = data_loader.load_regional_data()
        
        # 创建估计数据集（使用小样本进行快速测试）
        print("\n[3/5] 创建状态空间...")
        df_estimation, state_space, transition_matrices = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
        
        # 使用小样本进行测试
        sample_size = min(1000, len(df_estimation))
        df_estimation_sample = df_estimation.head(sample_size)
        
        print(f"  测试样本大小: {sample_size}")
        print(f"  状态空间大小: {len(state_space)}")
        print(f"  选择数量: {len(transition_matrices)}")
        
        # 3. 运行估计（只进行2次迭代作为测试）
        print("\n[4/5] 开始EM算法测试（2次迭代）...")
        
        estimation_params = {
            "observed_data": df_estimation_sample,
            "state_space": state_space,
            "transition_matrices": transition_matrices,
            "beta": config.discount_factor,
            "n_types": 2,  # 减少类型数量以加快测试
            "max_iterations": 2,  # 只测试2次迭代
            "tolerance": config.tolerance,
            "n_choices": len(transition_matrices),
            "regions_df": df_region,
        }
        
        try:
            results = run_em_algorithm(**estimation_params)
            
            print("\n[5/5] 测试成功完成！")
            print("\n估计结果摘要:")
            print(f"  最终对数似然值: {results['final_log_likelihood']:.4f}")
            print(f"  迭代次数: {results['n_iterations']}")
            print(f"  类型概率: {results['type_probabilities']}")
            
            # 检查是否有无效值
            if np.isnan(results['final_log_likelihood']) or np.isinf(results['final_log_likelihood']):
                print("\n⚠️  警告：对数似然值包含无效值")
                return False
            
            print("\n✓ 数值稳定性测试通过")
            print("✓ 未检测到除零错误")
            print("✓ 未检测到无效值（NaN或Inf）")
            return True
            
        except RuntimeWarning as e:
            print(f"\n✗ 检测到运行时警告: {e}")
            return False
        except Exception as e:
            print(f"\n✗ 估计过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_numerical_stability()
    
    if success:
        print("\n" + "="*80)
        print("所有测试通过！可以继续运行完整的估计。")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("测试失败，需要进一步调试。")
        print("="*80)
        sys.exit(1)
