"""
ABM模拟主脚本
"""
import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.abm.environment import run_abm_simulation, run_counterfactual_policy_analysis
from src.config.model_config import ModelConfig
from src.utils.outreg2 import output_estimation_results


def main():
    """
    ABM模拟主函数
    """
    print("开始ABM反事实政策模拟...")
    
    # 加载已估计的参数（这里使用模拟参数，实际应用中应从估计结果加载）
    # 为了演示，我们创建一组模拟的估计参数
    estimated_params = {
        "alpha_w": 0.8, "lambda": 1.5, "alpha_home": 0.5,
        "rho_base_tier_1": 2.0, "rho_base_tier_2": 1.5, "rho_base_tier_3": 1.0,
        "rho_edu": 0.3, "rho_health": 0.2, "rho_house": 0.4,
        "gamma_0_type_0": 0.5, "gamma_0_type_1": 1.2, "gamma_1": -0.15,
        "gamma_2": 0.3, "gamma_3": -0.35, "gamma_4": 0.02, "gamma_5": -0.1,
        "alpha_climate": 0.15, "alpha_health": 0.25, "alpha_education": 0.2,
        "alpha_public_services": 0.3,
        "n_choices": 31  # 这个参数不需要输出
    }
    
    # 类型概率
    type_probabilities = [0.4, 0.35, 0.25]  # 三个类型
    
    # 运行反事实政策分析
    policy_results = run_counterfactual_policy_analysis(
        estimated_params,
        type_probabilities
    )
    
    # 保存结果
    os.makedirs("results/policy", exist_ok=True)
    
    # 保存政策分析结果
    with open("results/policy/policy_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("ABM反事实政策分析结果摘要\n")
        f.write("="*40 + "\n")
        f.write(f"政策情景1 (提高公共设施): 迁移量变化 {policy_results['comparison']['migration_volume_change']['policy_1']:.2f}\n")
        f.write(f"政策情景2 (降低迁移成本): 迁移量变化 {policy_results['comparison']['migration_volume_change']['policy_2']:.2f}\n")
        
        pop_chg_1 = policy_results['comparison']['population_redistribution']['policy_1']
        pop_chg_2 = policy_results['comparison']['population_redistribution']['policy_2']
        f.write(f"政策情景1人口重新分布: 总变化 {pop_chg_1['total_absolute_change']:.2f}\n")
        f.write(f"政策情景2人口重新分布: 总变化 {pop_chg_2['total_absolute_change']:.2f}\n")
    
    print("ABM反事实政策模拟完成！")
    print("结果已保存到 results/policy/ 目录下")


if __name__ == '__main__':
    main()