import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from model_config import ModelConfig
from migration_parameters import MigrationParameters
from dynamic_programming import DynamicProgramming
from individual_likelihood import IndividualLikelihood
from likelihood_aggregator import LikelihoodAggregator

class MDPModel:
    """基于MDP的动态离散选择模型，整合所有组件"""
    def __init__(self, 
                 config: ModelConfig = None,
                 data: pd.DataFrame = None,
                 geo_data: pd.DataFrame = None,
                 dismatrix: np.ndarray = None,
                 adjmatrix: np.ndarray = None):
        # 如果没有提供配置，则创建默认配置
        if config is None:
            self.config = ModelConfig()
        else:
            self.config = config
            
        # 存储数据
        self.data = data
        self.geo_data = geo_data
        self.dismatrix = dismatrix
        self.adjmatrix = adjmatrix
        
        # 初始化参数
        self.params = MigrationParameters(self.config)
        
        # 如果提供了所有必要数据，则初始化似然聚合器
        if all(x is not None for x in [data, geo_data, dismatrix, adjmatrix]):
            self.likelihood_aggregator = LikelihoodAggregator(
                config=self.config,
                data=self.data,
                geo_data=self.geo_data,
                dismatrix=self.dismatrix,
                adjmatrix=self.adjmatrix,
                params=self.params
            )
        else:
            self.likelihood_aggregator = None
    
    def estimate(self, max_iter: int = None) -> MigrationParameters:
        """估计模型参数"""
        if self.likelihood_aggregator is None:
            raise ValueError("必须先提供所有必要数据才能进行参数估计")
        
        # 使用似然聚合器的估计方法
        return self.likelihood_aggregator.estimate(max_iter=max_iter)
    
    def calculate_log_likelihood(self) -> torch.Tensor:
        """计算当前参数下的对数似然"""
        if self.likelihood_aggregator is None:
            raise ValueError("必须先提供所有必要数据才能计算对数似然")
        
        # 获取当前参数的张量表示
        params_tensor = self.likelihood_aggregator.get_parameters_tensor()
        
        # 计算负对数似然（因为优化器是最小化目标函数）
        neg_log_lik = self.likelihood_aggregator(params_tensor)
        
        # 返回对数似然（取负）
        return -neg_log_lik
    
    def predict_choice_probabilities(self, age: int, csta: int, tau: int = None) -> torch.Tensor:
        """预测在给定状态下选择各地点的概率"""
        if self.likelihood_aggregator is None:
            raise ValueError("必须先提供所有必要数据才能进行预测")
        
        # 获取动态规划对象
        dp = self.likelihood_aggregator.dp
        
        # 如果没有指定类型，则计算所有类型的加权平均
        if tau is None:
            probs = torch.zeros(len(self.likelihood_aggregator.all_provinces))
            for t in range(self.config.n_tau_types):
                tau_probs = torch.stack([dp.get_choice_probability(age, csta, t, j) 
                                       for j in range(len(self.likelihood_aggregator.all_provinces))])
                probs += self.params.pi_tau[t] * tau_probs
            return probs
        else:
            # 返回特定类型的选择概率
            return torch.stack([dp.get_choice_probability(age, csta, tau, j) 
                              for j in range(len(self.likelihood_aggregator.all_provinces))])
    
    def calculate_counterfactual(self, policy_params: Dict[str, float]) -> torch.Tensor:
        """计算反事实政策下的对数似然"""
        if self.likelihood_aggregator is None:
            raise ValueError("必须先提供所有必要数据才能计算反事实")
        
        # 保存原始参数
        original_params = {name: param.clone() for name, param in self.params.named_parameters()}
        
        # 更新参数为政策参数
        with torch.no_grad():
            for name, value in policy_params.items():
                if hasattr(self.params, name):
                    getattr(self.params, name).copy_(torch.tensor(value))
        
        # 计算新参数下的对数似然
        cf_log_lik = self.calculate_log_likelihood()
        
        # 恢复原始参数
        with torch.no_grad():
            for name, param in original_params.items():
                if hasattr(self.params, name):
                    getattr(self.params, name).copy_(param)
        
        return cf_log_lik

# 示例用法
def example_usage():
    # 创建配置
    config = ModelConfig()
    
    # 加载数据（示例，实际使用时需要替换为真实数据）
    data = pd.read_csv('individual_data.csv')
    geo_data = pd.read_csv('geo_data.csv')
    adjmatrix = np.load('adjmatrix.npy')
    dismatrix = np.load('dismatrix.npy')
    
    # 创建模型
    model = MDPModel(
        config=config,
        data=data,
        geo_data=geo_data,
        dismatrix=dismatrix,
        adjmatrix=adjmatrix
    )
    
    # 估计参数
    estimated_params = model.estimate(max_iter=100)
    
    # 计算对数似然
    log_lik = model.calculate_log_likelihood()
    print(f"对数似然: {log_lik.item()}")
    
    # 预测选择概率
    probs = model.predict_choice_probabilities(age=30, csta=1)
    print(f"选择概率: {probs}")
    
    # 计算反事实政策（例如，降低迁移成本）
    cf_log_lik = model.calculate_counterfactual({'gamma0': 0.3, 'gamma1': -0.05})
    print(f"反事实政策下的对数似然: {cf_log_lik.item()}")

if __name__ == "__main__":
    example_usage()