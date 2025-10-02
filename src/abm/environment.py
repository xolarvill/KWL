"""
ABM代理基模型模拟
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.abm.agents import Agent
from src.config.model_config import ModelConfig


class Environment:
    """
    ABM环境类，代表中国的省份和地区
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.n_regions = config.n_regions  # 31个省份/地区
        self.n_periods = config.n_period  # 模拟期数
        
        # 初始化地区特征（简化处理，实际应用中从数据加载）
        self.region_characteristics = self._initialize_region_characteristics()
        
        # 初始化距离矩阵、邻接矩阵等（实际应用中从数据加载）
        self.distance_matrix = np.random.rand(self.n_regions, self.n_regions) * 1000  # 随机距离
        self.adjacency_matrix = (np.random.rand(self.n_regions, self.n_regions) > 0.8).astype(int)  # 随机邻接
        
        # 确保对角线为0（自己到自己的距离为0，自己不与自己邻接）
        np.fill_diagonal(self.distance_matrix, 0)
        np.fill_diagonal(self.adjacency_matrix, 0)
        
    def _initialize_region_characteristics(self):
        """
        初始化地区特征
        """
        characteristics = {}
        for region_id in range(self.n_regions):
            characteristics[region_id] = {
                'avg_wage': np.random.normal(60000, 15000),  # 平均工资
                'amenity_climate': np.random.normal(0, 1),   # 气候舒适度
                'amenity_health': np.random.normal(0, 1),    # 医疗舒适度
                'amenity_education': np.random.normal(0, 1), # 教育舒适度
                'amenity_public_services': np.random.normal(0, 1), # 公共服务舒适度
                'housing_price': np.random.normal(15000, 5000), # 房价
                'population': np.random.randint(1000000, 100000000), # 人口
                'tier': np.random.choice([1, 2, 3])  # 城市等级
            }
        return characteristics
    
    def get_region_characteristics(self, region_id: int, period: int = 0):
        """
        获取指定地区的特征
        """
        return self.region_characteristics.get(region_id, {})
    
    def get_average_wage(self, region_id: int) -> float:
        """
        获取指定地区的平均工资
        """
        return self.region_characteristics[region_id].get('avg_wage', 50000)
    
    def get_distance(self, region1: int, region2: int) -> float:
        """
        获取两个地区之间的距离
        """
        if region1 < self.n_regions and region2 < self.n_regions:
            return self.distance_matrix[region1, region2]
        else:
            return float('inf')  # 如果地区ID超出范围，返回无穷大距离
    
    def is_adjacent(self, region1: int, region2: int) -> bool:
        """
        检查两个地区是否相邻
        """
        if region1 < self.n_regions and region2 < self.n_regions:
            return bool(self.adjacency_matrix[region1, region2])
        else:
            return False
    
    def get_population(self, region_id: int) -> int:
        """
        获取指定地区的人口
        """
        return self.region_characteristics[region_id].get('population', 10000000)


def run_abm_simulation(
    estimated_params: Dict[str, float],
    type_probabilities: List[float],
    n_agents: int = 10000,
    n_periods: int = 20,
    policy_scenario: str = "baseline"
) -> Dict[str, Any]:
    """
    运行ABM模拟
    
    Args:
        estimated_params: 从结构估计得到的参数
        type_probabilities: 类型概率
        n_agents: 代理人数量
        n_periods: 模拟期数
        policy_scenario: 政策情景
    
    Returns:
        Dict[str, Any]: 模拟结果
    """
    print(f"开始ABM模拟，政策情景: {policy_scenario}")
    print(f"代理人数量: {n_agents}, 模拟期数: {n_periods}")
    
    # 初始化环境
    config = ModelConfig()
    environment = Environment(config)
    
    # 初始化代理人
    agents = []
    for i in range(n_agents):
        # 根据类型概率分配代理人类型
        agent_type = np.random.choice(len(type_probabilities), p=type_probabilities)
        
        # 创建代理人
        agent = Agent(
            agent_id=i,
            initial_location=np.random.choice(environment.n_regions),
            agent_type=agent_type,
            params=estimated_params
        )
        agents.append(agent)
    
    # 模拟过程
    simulation_results = {
        'migration_flows': [],
        'population_distribution': [],
        'average_wages': [],
        'policy_effects': []
    }
    
    for period in range(n_periods):
        print(f"模拟第 {period+1} 期...")
        
        # 记录期初状态
        period_population = [0] * environment.n_regions
        for agent in agents:
            period_population[agent.current_location] += 1
            
        simulation_results['population_distribution'].append(period_population.copy())
        
        # 代理人决策和移动
        n_migrations = 0
        for agent in agents:
            # 代理人根据效用最大化选择下一个位置
            new_location = agent.make_location_choice(environment, period)
            if new_location != agent.current_location:
                # 更新位置
                agent.move_to(new_location)
                n_migrations += 1
        
        # 记录迁移流量
        simulation_results['migration_flows'].append(n_migrations)
        
        # 计算平均工资
        avg_wages = [environment.get_average_wage(region) for region in range(environment.n_regions)]
        simulation_results['average_wages'].append(avg_wages)
    
    print("ABM模拟完成。")
    return simulation_results


def run_counterfactual_policy_analysis(
    baseline_params: Dict[str, float],
    baseline_type_probs: List[float]
) -> Dict[str, Any]:
    """
    运行反事实政策分析
    
    Args:
        baseline_params: 基线参数
        baseline_type_probs: 基线类型概率
    
    Returns:
        Dict[str, Any]: 政策分析结果
    """
    print("开始反事实政策分析...")
    
    # 基线情景模拟
    print("运行基线情景...")
    baseline_results = run_abm_simulation(
        baseline_params, baseline_type_probs, 
        policy_scenario="baseline"
    )
    
    # 政策情景1: 提高某地区的公共设施水平
    print("运行政策情景1: 提高北京的公共设施水平...")
    policy_params_1 = baseline_params.copy()
    # 修改北京相关的公共设施参数（这里简化处理）
    policy_params_1['alpha_public_services'] += 0.2  # 提高公共设施的效用权重
    
    policy_results_1 = run_abm_simulation(
        policy_params_1, baseline_type_probs,
        policy_scenario="policy_1_enhanced_public_services"
    )
    
    # 政策情景2: 降低迁移成本
    print("运行政策情景2: 降低迁移成本...")
    policy_params_2 = baseline_params.copy()
    # 调整迁移成本参数
    policy_params_2['gamma_0_type_0'] *= 0.8  # 降低固定迁移成本
    policy_params_2['gamma_0_type_1'] *= 0.8
    
    policy_results_2 = run_abm_simulation(
        policy_params_2, baseline_type_probs,
        policy_scenario="policy_2_reduced_migration_cost"
    )
    
    # 比较结果
    policy_analysis_results = {
        'baseline': baseline_results,
        'policy_1_enhanced_public_services': policy_results_1,
        'policy_2_reduced_migration_cost': policy_results_2,
        'comparison': compare_policy_scenarios(baseline_results, policy_results_1, policy_results_2)
    }
    
    print("反事实政策分析完成。")
    return policy_analysis_results


def compare_policy_scenarios(
    baseline_results: Dict[str, Any],
    policy1_results: Dict[str, Any],
    policy2_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    比较不同政策情景的结果
    """
    comparison = {
        'migration_volume_change': {
            'policy_1': np.mean(policy1_results['migration_flows']) - np.mean(baseline_results['migration_flows']),
            'policy_2': np.mean(policy2_results['migration_flows']) - np.mean(baseline_results['migration_flows'])
        },
        'population_redistribution': {
            'policy_1': calculate_population_change(baseline_results['population_distribution'], 
                                                   policy1_results['population_distribution']),
            'policy_2': calculate_population_change(baseline_results['population_distribution'], 
                                                   policy2_results['population_distribution'])
        }
    }
    
    return comparison


def calculate_population_change(baseline_pop: List[List[int]], policy_pop: List[List[int]]) -> Dict[str, float]:
    """
    计算人口分布变化
    """
    if not baseline_pop or not policy_pop:
        return {}
    
    n_periods = min(len(baseline_pop), len(policy_pop))
    n_regions = len(baseline_pop[0]) if baseline_pop else 0
    
    # 计算平均人口分布变化
    baseline_avg = np.mean(baseline_pop[:n_periods], axis=0) if n_periods > 0 else np.zeros(n_regions)
    policy_avg = np.mean(policy_pop[:n_periods], axis=0) if n_periods > 0 else np.zeros(n_regions)
    
    changes = policy_avg - baseline_avg
    total_change = np.sum(np.abs(changes))
    
    return {
        'total_absolute_change': total_change,
        'avg_per_region': np.mean(np.abs(changes)),
        'max_single_region_change': np.max(np.abs(changes))
    }


def save_abm_results(results: Dict[str, Any], output_path: str = "results/policy/abm_results.npz"):
    """
    保存ABM模拟结果
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 由于结果可能是复杂的嵌套结构，这里简化保存
    # 在实际应用中，可能需要更复杂的序列化方法
    np.savez_compressed(output_path, **{
        'migration_flows': np.array(results.get('migration_flows', [])),
        'population_distribution': np.array(results.get('population_distribution', [])),
        'average_wages': np.array(results.get('average_wages', []))
    })
    
    print(f"ABM结果已保存到: {output_path}")


if __name__ == '__main__':
    print("ABM模拟脚本 - 该脚本需要结构估计的结果作为输入")
    print("请先运行结构估计脚本 (02_run_estimation.py)，然后使用估计结果运行此脚本")