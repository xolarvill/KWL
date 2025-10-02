"""
ABM政策反事实分析：模拟不同政策场景下的人口流动
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.abm.environment import Environment
from src.abm.agents import Agent
from src.config.model_config import ModelConfig

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False

def run_policy_simulation(policy_name, policy_params, n_agents=10000, n_periods=20):
    """
    运行单个政策场景的ABM模拟
    
    Args:
        policy_name: 政策名称
        policy_params: 政策参数字典
        n_agents: 代理人数量
        n_periods: 模拟期数
    
    Returns:
        模拟结果字典
    """
    print(f"\n{'='*60}")
    print(f"政策场景: {policy_name}")
    print(f"{'='*60}")
    
    # 初始化环境
    config = ModelConfig()
    env = Environment(config, policy_params)
    
    # 初始化代理人
    agents = []
    for i in range(n_agents):
        agent = Agent(
            agent_id=i,
            initial_location=np.random.choice(31),  # 随机初始位置
            hukou_location=np.random.choice(31),    # 随机户籍
            age=np.random.randint(18, 60),
            agent_type=np.random.choice([0, 1], p=[0.6, 0.4])
        )
        agents.append(agent)
    
    # 记录结果
    population_by_region = np.zeros((n_periods, 31))
    migration_flows = np.zeros((n_periods, 31, 31))
    
    # 运行模拟
    for t in range(n_periods):
        print(f"  期数 {t+1}/{n_periods}", end='\r')
        
        # 记录当前人口分布
        for agent in agents:
            population_by_region[t, agent.current_location] += 1
        
        # 每个代理人做决策
        for agent in agents:
            old_location = agent.current_location
            new_location = agent.make_decision(env, policy_params)
            
            if new_location != old_location:
                migration_flows[t, old_location, new_location] += 1
            
            agent.current_location = new_location
            agent.age += 1
    
    print(f"\n  模拟完成")
    
    return {
        "population": population_by_region,
        "flows": migration_flows,
        "final_distribution": population_by_region[-1, :] / n_agents
    }

def main():
    """主函数：运行多个政策场景并对比"""
    
    # 基准参数
    baseline_params = {
        "alpha_w": 0.8, "lambda": 1.5, "alpha_home": 0.5,
        "rho_base_tier_1": 2.0, "rho_edu": 0.3, "rho_health": 0.2, "rho_house": 0.4,
        "gamma_0_type_0": 0.5, "gamma_0_type_1": 1.2, "gamma_1": -0.15,
        "gamma_2": 0.3, "gamma_3": -0.35, "gamma_4": 0.02, "gamma_5": -0.1,
        "alpha_climate": 0.15, "alpha_health": 0.25, "alpha_education": 0.2,
        "alpha_public_services": 0.3, "n_choices": 31
    }
    
    # 定义政策场景
    policies = {
        "基准场景": baseline_params,
        "取消户籍限制": {**baseline_params, "rho_base_tier_1": 0, "rho_edu": 0, "rho_health": 0, "rho_house": 0},
        "提升二线城市吸引力": {**baseline_params, "alpha_public_services": 0.5},
        "降低迁移成本": {**baseline_params, "gamma_0_type_0": 0.2, "gamma_0_type_1": 0.5, "gamma_1": -0.05},
    }
    
    # 运行所有场景
    all_results = {}
    for policy_name, policy_params in policies.items():
        all_results[policy_name] = run_policy_simulation(
            policy_name, 
            policy_params,
            n_agents=5000,  # 使用较小样本以加快速度
            n_periods=10
        )
    
    # 生成对比图表
    print("\n生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 人口分布变化
    ax = axes[0, 0]
    for policy_name, results in all_results.items():
        top_5_regions = np.argsort(results['final_distribution'])[-5:]
        ax.bar(range(5), results['final_distribution'][top_5_regions], 
               alpha=0.7, label=policy_name)
    ax.set_title('前5大城市人口分布对比', fontsize=14)
    ax.set_xlabel('城市排名')
    ax.set_ylabel('人口占比')
    ax.legend()
    
    # 2. 迁移率时间序列
    ax = axes[0, 1]
    for policy_name, results in all_results.items():
        migration_rates = []
        for t in range(results['flows'].shape[0]):
            total_moves = results['flows'][t].sum()
            total_pop = results['population'][t].sum()
            migration_rates.append(total_moves / total_pop if total_pop > 0 else 0)
        ax.plot(migration_rates, label=policy_name, marker='o')
    ax.set_title('迁移率时间演化', fontsize=14)
    ax.set_xlabel('时期')
    ax.set_ylabel('迁移率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 人口集中度（HHI指数）
    ax = axes[1, 0]
    hhi_values = {}
    for policy_name, results in all_results.items():
        hhi = np.sum(results['final_distribution']**2)
        hhi_values[policy_name] = hhi
    ax.bar(range(len(hhi_values)), list(hhi_values.values()))
    ax.set_xticks(range(len(hhi_values)))
    ax.set_xticklabels(list(hhi_values.keys()), rotation=45, ha='right')
    ax.set_title('人口集中度（HHI指数）', fontsize=14)
    ax.set_ylabel('HHI')
    
    # 4. 净流入前10城市
    ax = axes[1, 1]
    baseline_pop = all_results['基准场景']['population'][0, :]
    final_pop = all_results['基准场景']['population'][-1, :]
    net_inflow = final_pop - baseline_pop
    top_10 = np.argsort(net_inflow)[-10:]
    ax.barh(range(10), net_inflow[top_10])
    ax.set_title('净流入人口前10城市（基准场景）', fontsize=14)
    ax.set_xlabel('净流入人数')
    ax.set_ylabel('城市')
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/policy_counterfactual.png', dpi=300, bbox_inches='tight')
    print("图表已保存到: results/figures/policy_counterfactual.png")
    
    # 生成政策分析摘要
    summary_path = "results/policy/policy_analysis_summary.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ABM政策反事实分析摘要\n")
        f.write("="*80 + "\n\n")
        
        for policy_name, results in all_results.items():
            f.write(f"\n{policy_name}:\n")
            f.write(f"  最终人口集中度(HHI): {hhi_values[policy_name]:.4f}\n")
            f.write(f"  平均迁移率: {np.mean([results['flows'][t].sum()/results['population'][t].sum() for t in range(results['flows'].shape[0])]):.4f}\n")
            f.write(f"  人口最多的5个地区: {np.argsort(results['final_distribution'])[-5:]}\n")
    
    print(f"政策分析摘要已保存到: {summary_path}")
    
    return all_results

if __name__ == '__main__':
    results = main()
    print("\n政策反事实分析完成！")
