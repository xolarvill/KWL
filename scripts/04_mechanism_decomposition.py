"""
机制分解分析：量化不同机制对迁移决策的贡献
"""
import sys
import os
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.config.model_config import ModelConfig
from src.model.bellman import solve_bellman_equation
from src.model.utility import calculate_flow_utility
from src.model.likelihood import calculate_choice_probabilities

def run_mechanism_decomposition():
    """运行机制分解分析"""
    print("="*80)
    print("机制分解分析")
    print("="*80)
    
    # 加载配置和数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    # 加载估计结果（假设已经保存）
    results_path = "results/estimation/final_estimates.pkl"
    if not os.path.exists(results_path):
        print("警告：未找到估计结果，使用模拟参数")
        estimated_params = {
            "alpha_w": 0.8, "lambda": 1.5, "alpha_home": 0.5,
            "rho_base_tier_1": 2.0, "rho_edu": 0.3, "rho_health": 0.2, "rho_house": 0.4,
            "gamma_0_type_0": 0.5, "gamma_0_type_1": 1.2, "gamma_1": -0.15,
            "gamma_2": 0.3, "gamma_3": -0.35, "gamma_4": 0.02, "gamma_5": -0.1,
            "alpha_climate": 0.15, "alpha_health": 0.25, "alpha_education": 0.2,
            "alpha_public_services": 0.3, "n_choices": 31
        }
    
    # 加载数据
    config.regional_data_path = config.regional_data_path.replace('geo.xlsx', 'geo_amenities.csv')
    df_region = data_loader.load_regional_data()
    df_estimation, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    
    # 定义反事实场景
    scenarios = {
        "baseline": estimated_params.copy(),
        "no_hukou": {**estimated_params, "rho_base_tier_1": 0, "rho_edu": 0, "rho_health": 0, "rho_house": 0},
        "no_home_premium": {**estimated_params, "alpha_home": 0},
        "no_distance_cost": {**estimated_params, "gamma_1": 0},
        "no_migration_cost": {**estimated_params, "gamma_0_type_0": 0, "gamma_0_type_1": 0, "gamma_1": 0}
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\n计算场景: {scenario_name}")
        
        # 求解价值函数
        v_converged, _ = solve_bellman_equation(
            utility_function=calculate_flow_utility,
            state_space=state_space,
            params=params,
            agent_type=0,
            beta=config.discount_factor,
            transition_matrices=transition_matrices,
            regions_df=df_region,
            max_iterations=100
        )
        
        # 计算选择概率
        ccps = calculate_choice_probabilities(
            v_converged,
            calculate_flow_utility,
            state_space,
            params,
            0,
            config.discount_factor,
            transition_matrices,
            df_region
        )
        
        # 计算迁移率
        migration_prob = 1 - np.diag(ccps).mean()
        results[scenario_name] = {
            "migration_rate": migration_prob,
            "avg_value": v_converged.mean()
        }
        
        print(f"  迁移率: {migration_prob:.4f}")
        print(f"  平均价值: {v_converged.mean():.4f}")
    
    # 计算各机制的贡献
    baseline_migration = results["baseline"]["migration_rate"]
    
    decomposition = {
        "户籍制度": (results["no_hukou"]["migration_rate"] - baseline_migration) / baseline_migration * 100,
        "家乡溢价": (results["no_home_premium"]["migration_rate"] - baseline_migration) / baseline_migration * 100,
        "地理距离": (results["no_distance_cost"]["migration_rate"] - baseline_migration) / baseline_migration * 100,
        "迁移成本": (results["no_migration_cost"]["migration_rate"] - baseline_migration) / baseline_migration * 100,
    }
    
    # 生成LaTeX表格
    output_path = "results/tables/mechanism_decomposition.tex"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{机制分解：各因素对迁移决策的贡献}\n")
        f.write("\\label{tab:mechanism_decomposition}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("机制 & 基准迁移率 & 移除后变化(\\%) \\\\\n")
        f.write("\\midrule\n")
        f.write(f"基准模型 & {baseline_migration:.4f} & - \\\\\n")
        for mechanism, change in decomposition.items():
            f.write(f"{mechanism} & - & {change:+.2f}\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n机制分解结果已保存到: {output_path}")
    return decomposition

if __name__ == '__main__':
    decomposition = run_mechanism_decomposition()
    print("\n机制分解完成！")
