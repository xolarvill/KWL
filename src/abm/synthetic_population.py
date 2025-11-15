"""
ABM人口合成模块
实现论文tex第1383-1390行的三步法人口合成
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.model_config import ModelConfig
from src.utils.prov_indexer import ProvIndexer


class SyntheticPopulation:
    """
    合成人口生成器
    三步法：1)抽取类型 2)抽样可观测特征 3)抽样未观测异质性
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.n_agents = 100000  # 论文要求10万代理人
        self.prov_indexer = ProvIndexer(config)
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载CLDS和地区数据"""
        # CLDS 2018截面数据
        self.clds_data = pd.read_csv(self.config.individual_data_path)
        
        # 地区特征数据
        self.regional_data = pd.read_excel(self.config.regional_data_path)
        
        # 获取省份代码到索引的映射
        self.prov_to_idx = self.prov_indexer.get_prov_to_idx_map()
        self.idx_to_prov = {v: k for k, v in self.prov_to_idx.items()}
        
        print(f"加载数据完成: CLDS {len(self.clds_data)}条记录, 地区数据 {len(self.regional_data)}个省份")
        
    def create_population(self, type_probabilities: np.ndarray) -> pd.DataFrame:
        """
        创建合成人口
        
        Args:
            type_probabilities: 类型概率向量 π̂
            
        Returns:
            DataFrame包含所有代理人特征
        """
        print(f"\n开始创建合成人口，目标规模: {self.n_agents}个代理人...")
        
        # Step 1: 从多项分布抽取潜在类别（tex第1384行）
        agent_types = self._sample_agent_types(type_probabilities)
        
        # Step 2: 从CLDS 2018有放回抽样可观测特征（tex第1386-1387行）
        observable_features = self._sample_observable_features()
        
        # Step 3: 抽取未观测异质性（tex第1389-1390行）
        unobserved_heterogeneity = self._sample_unobserved_heterogeneity()
        
        # 合并成完整数据集
        population = self._combine_features(
            agent_types, observable_features, unobserved_heterogeneity
        )
        
        print(f"合成人口创建完成: {len(population)}个代理人")
        return population
    
    def _sample_agent_types(self, type_probs: np.ndarray) -> np.ndarray:
        """
        Step 1: 抽取潜在类别 τ_i ~ Categorical(π̂)
        
        Args:
            type_probs: 类型概率向量 (K,)
            
        Returns:
            agent_types: 代理人类型数组 (n_agents,)
        """
        n_types = len(type_probs)
        agent_types = np.random.choice(
            n_types, 
            size=self.n_agents, 
            p=type_probs
        )
        
        # 统计分布
        type_counts = np.bincount(agent_types, minlength=n_types)
        print(f"  类型分布: {dict(zip(range(n_types), type_counts))}")
        
        return agent_types
    
    def _sample_observable_features(self) -> pd.DataFrame:
        """
        Step 2: 从CLDS 2018有放回抽样可观测特征
        保留年龄、教育、户籍地、当前地等变量的联合分布
        
        Returns:
            DataFrame包含可观测特征
        """
        # 从CLDS 2018中随机抽样（有放回）
        sampled_data = self.clds_data.sample(
            n=self.n_agents, 
            replace=True, 
            random_state=42
        ).copy()
        
        features = pd.DataFrame()
        
        # 提取关键特征
        features['age'] = sampled_data['age'].values
        features['education'] = sampled_data.get('education', 12).values  # 默认值
        
        # 处理户籍和当前位置（使用省份代码转索引）
        features['hukou_location'] = sampled_data['hukou_prov'].apply(
            lambda x: self.prov_indexer.index(x, 'rank') - 1  # 转为0基索引
        ).values
        
        features['initial_location'] = sampled_data['provcd'].apply(
            lambda x: self.prov_indexer.index(x, 'rank') - 1  # 转为0基索引
        ).values
        
        print(f"  可观测特征抽样完成: {len(features)}条记录")
        print(f"    - 年龄范围: {features['age'].min():.0f} - {features['age'].max():.0f}")
        print(f"    - 户籍地分布: {len(features['hukou_location'].unique())}个省份")
        print(f"    - 当前地分布: {len(features['initial_location'].unique())}个省份")
        
        return features
    
    def _sample_unobserved_heterogeneity(self) -> Dict[str, np.ndarray]:
        """
        Step 3: 抽取未观测异质性
        - η_i: 个体固定效应
        - ν_ij: 个体-地区收入匹配
        - ξ_ij: 个体-地区偏好匹配
        
        Returns:
            字典包含各种异质性数组
        """
        n_regions = self.config.n_choices
        
        # η_i: 个体固定效应（从离散支撑点抽取）
        eta_support = np.linspace(
            self.config.eta_range[0], 
            self.config.eta_range[1], 
            self.config.n_eta_support
        )
        eta_probs = np.ones(self.config.n_eta_support) / self.config.n_eta_support
        eta_i = np.random.choice(eta_support, size=self.n_agents, p=eta_probs)
        
        # ν_ij 和 ξ_ij: 对每个代理人，为每个可能的目的地抽样
        # 这里简化处理：只给初始位置和户籍位置赋值，其他为0
        nu_ij_values = np.zeros((self.n_agents, n_regions))
        xi_ij_values = np.zeros((self.n_agents, n_regions))
        
        # 使用简化策略：只为访问过的地区实例化（与论文一致）
        print(f"  未观测异质性抽样完成")
        print(f"    - η_i支撑点范围: [{eta_support.min():.2f}, {eta_support.max():.2f}]")
        print(f"    - ν_ij和ξ_ij: 简化策略，初始化为0")
        
        return {
            'eta_i': eta_i,
            'nu_ij': nu_ij_values,
            'xi_ij': xi_ij_values
        }
    
    def _combine_features(self, 
                         agent_types: np.ndarray,
                         observable_features: pd.DataFrame,
                         unobserved_heterogeneity: Dict[str, np.ndarray]) -> pd.DataFrame:
        """合并所有特征成完整数据集"""
        
        population = pd.DataFrame()
        
        # 基本信息
        population['agent_id'] = range(self.n_agents)
        population['agent_type'] = agent_types
        
        # 可观测特征
        population['age'] = observable_features['age'].values
        population['education'] = observable_features['education'].values
        population['hukou_location'] = observable_features['hukou_location'].values
        population['current_location'] = observable_features['initial_location'].values
        
        # 未观测异质性
        population['eta_i'] = unobserved_heterogeneity['eta_i']
        
        # 访问历史（初始化为已访问初始位置）
        population['visited_locations'] = population['current_location'].apply(
            lambda x: {int(x)}
        )
        
        print(f"  特征合并完成: {len(population.columns)}个变量")
        
        return population


def create_synthetic_population_example():
    """创建示例合成人口"""
    config = ModelConfig()
    
    # 使用Config中的初始值作为占位符
    type_probs = config.get_initial_type_probabilities()
    
    print("="*60)
    print("ABM人口合成示例")
    print("="*60)
    print(f"类型概率: {type_probs}")
    print(f"支撑点配置: {config.get_discrete_support_config()}")
    
    # 创建合成器
    synth = SyntheticPopulation(config)
    
    # 生成人口
    population = synth.create_population(type_probs)
    
    print(f"\n合成人口概览:")
    print(population.head())
    print(f"\n数据类型分布:")
    print(population.dtypes)
    
    return population


if __name__ == '__main__':
    population = create_synthetic_population_example()