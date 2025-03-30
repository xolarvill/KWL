# dynamic_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from config.model_config import ModelConfig
from model.migration_parameters import MigrationParameters
from model.individual_likelihood import DynamicProgramming, IndividualLikelihood
from model.likelihood_aggregator import LikelihoodAggregator

class DynamicModel(nn.Module):
    """动态离散选择模型的统一接口
    
    该类整合了DynamicProgramming、IndividualLikelihood和LikelihoodAggregator，
    提供了一个统一的接口来进行模型估计。
    
    参数:
        config (ModelConfig): 模型配置
        individual_data (pd.DataFrame): 个体面板数据
        regional_data (pd.DataFrame): 地区特征数据
        adjacency_matrix (np.ndarray): 地区临近矩阵
        distance_matrix (np.ndarray): 地区距离矩阵
    """
    def __init__(self, 
                 config: ModelConfig,
                 individual_data: pd.DataFrame,
                 regional_data: pd.DataFrame,
                 adjacency_matrix: np.ndarray,
                 distance_matrix: np.ndarray):
        super().__init__()
        self.config = config
        self.individual_data = individual_data
        self.regional_data = regional_data
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = distance_matrix
        
        # 初始化参数
        self.params = MigrationParameters(config)
        
        # 创建似然聚合器
        self.likelihood_aggregator = self._create_likelihood_aggregator()
        
    def _create_likelihood_aggregator(self) -> LikelihoodAggregator:
        """创建似然聚合器"""
        return LikelihoodAggregator(
            config=self.config,
            individual_data=self.individual_data,
            regional_data=self.regional_data,
            adjacency_matrix=self.adjacency_matrix,
            distance_matrix=self.distance_matrix,
            params=self.params
        )
    
    def forward(self) -> torch.Tensor:
        """前向传播，计算总体对数似然"""
        return self.likelihood_aggregator.log_likelihood()
    
    def log_likelihood(self) -> torch.Tensor:
        """计算总体对数似然"""
        return self.likelihood_aggregator.log_likelihood()
    
    def calculate_individual_likelihood(self, pid: int) -> torch.Tensor:
        """计算单个个体的似然函数"""
        return self.likelihood_aggregator._compute_single_likelihood(pid)
    
    def parameters(self):
        """返回模型参数"""
        return self.params.parameters()
    
    def named_parameters(self):
        """返回命名参数"""
        return self.params.named_parameters()
    
    def get_dynamic_programming(self) -> DynamicProgramming:
        """获取动态规划求解器"""
        return self.likelihood_aggregator.dp
    
    def get_params(self) -> MigrationParameters:
        """获取模型参数"""
        return self.params
    
    def get_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.config
    
    def get_likelihood_aggregator(self) -> LikelihoodAggregator:
        """获取似然聚合器"""
        return self.likelihood_aggregator
    
    @classmethod
    def from_data_loader(cls, config: ModelConfig, data_loader):
        """从数据加载器创建模型"""
        individual_data = data_loader.load_individual_data()
        regional_data = data_loader.load_regional_data()
        adjacency_matrix = data_loader.load_adjacency_matrix()
        
        # 假设我们有距离矩阵，如果没有，可以从邻接矩阵生成一个简单的距离矩阵
        try:
            distance_matrix = np.load('file/distance.npy')
        except FileNotFoundError:
            print("未找到距离矩阵，使用邻接矩阵生成简单距离矩阵")
            # 简单距离矩阵：邻接为1，非邻接为2，对角线为0
            distance_matrix = np.where(adjacency_matrix > 0, 1, 2)
            np.fill_diagonal(distance_matrix, 0)
        
        return cls(
            config=config,
            individual_data=individual_data,
            regional_data=regional_data,
            adjacency_matrix=adjacency_matrix,
            distance_matrix=distance_matrix
        )
                            
        # 计算该组合的概率权重
        omega_prob = (
            self.config.prob_eta_support_points[eta_idx] *
            self.config.prob_nu_support_points[nu_idx] *
            self.config.prob_xi_support_points[xi_idx] *
            self.config.prob_sigmavarepsilon_support_points[sigma_idx]
        )
        
        # 加权求和
        tau_lik += omega_prob * omega_lik
            
        # 加权求和（使用类型概率）
        total_lik += self.tau_probabilities[tau] * tau_lik
        
        return total_lik
    
    def _calculate_omega_likelihood(self, individual_data, tau, eta_idx, nu_idx, xi_idx, sigma_idx):
        """计算特定类型tau和支撑点组合下的似然"""
        # 实现具体的似然计算逻辑
        pass
        
    def log_likelihood(self, data):
        """计算总体似然函数"""
        # 计算所有个体的对数似然总和
        log_lik = torch.tensor(0.0)
        
        for individual_id in data['pid'].unique():
            individual_data = data[data['pid'] == individual_id]
            individual_lik = self.individual_likelihood(individual_data)
            log_lik += torch.log(individual_lik + 1e-12)  # 防止log(0)
            
        return log_lik