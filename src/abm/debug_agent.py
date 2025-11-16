"""
调试Agent决策
"""
import numpy as np
import pandas as pd

# 导入模块
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.abm.pure_python_utility import PurePythonUtility
from src.abm.agent_decision import ABMAgent

# 设置
n_regions = 10

# 创建Utility
utility = PurePythonUtility(
    distance_matrix=np.random.uniform(100, 2000, (n_regions, n_regions)),
    adjacency_matrix=(np.random.rand(n_regions, n_regions) > 0.8).astype(int),
    region_data=None,
    n_regions=n_regions
)
utility._create_random_data()

# Agent数据
agent_data = {
    'age': 28,
    'current_location': 0,
    'hukou_location': 0,
    'agent_type': 0,
    'education': 16
}

# 参数
params = {
    'alpha_w': 1.0, 'alpha_home': 1.0, 'alpha_climate': 0.1, 'alpha_health': 0.1,
    'alpha_education': 0.1, 'alpha_public_services': 0.1, 'alpha_hazard': 0.1,
    'rho_base_tier_1': 2.0, 'rho_base_tier_2': 1.0, 'rho_base_tier_3': 0.5,
    'rho_edu': 0.3, 'rho_health': 0.2, 'rho_house': 0.4,
    'gamma_0_type_0': 0.5, 'gamma_0_type_1': 1.5, 'gamma_0_type_2': 2.0,
    'gamma_1': -0.15, 'gamma_2': 0.3, 'gamma_3': -0.35, 'gamma_4': 0.02, 'gamma_5': -0.1
}

# 计算效用
utilities = utility.calculate_utility(
    agent_data=agent_data,
    params=params,
    eta_i=0.5,
    nu_ij=None,
    xi_ij=None
)

print(f"=== 调试信息 ===")
print(f"n_regions: {n_regions}")
print(f"utilities shape: {utilities.shape}")
print(f"utilities: {utilities}")
print(f"utilities length: {len(utilities)}")

# Softmax
probs = np.exp(utilities - utilities.max()) / np.sum(np.exp(utilities - utilities.max()))
print(f"probs shape: {probs.shape}")
print(f"probs length: {len(probs)}")
print(f"probs sum: {probs.sum()}")

# 创建Agent
agent = ABMAgent(
    agent_id=0,
    agent_type=0,
    age=28,
    education=16,
    hukou_location=0,
    current_location=0,
    eta_i=0.5,
    n_regions=n_regions
)

# 在Agent内部添加调试
print(f"\nAgent内部状态:")
print(f"agent.state.n_regions: {agent.state.n_regions}")

# 尝试复制Agent.make_decision的逻辑
agent_data2 = agent._prepare_agent_data()
print(f"agent_data: {agent_data2}")

utilities2 = utility.calculate_utility(
    agent_data=agent_data2,
    params=params,
    eta_i=agent.state.eta_i,
    nu_ij=None,
    xi_ij=None
)

print(f"\nutilities2 shape: {utilities2.shape}")
print(f"utilities2: {utilities2}")

probs2 = np.exp(utilities2 - utilities2.max()) / np.sum(np.exp(utilities2 - utilities2.max()))
print(f"probs2 shape: {probs2.shape}")
print(f"probs2 length: {len(probs2)}")

# 测试选择
try:
    chosen = np.random.choice(agent.state.n_regions, p=probs2)
    print(f"✓ 选择成功: {chosen}")
except ValueError as e:
    print(f"✗ 选择失败: {e}")
    print(f"  probs2 length: {len(probs2)}, n_regions: {agent.state.n_regions}")
    print(f"  两者相等吗: {len(probs2) == agent.state.n_regions}")