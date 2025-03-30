import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
from model_config import ModelConfig
from migration_parameters import MigrationParameters
from individual_likelihood import IndividualLikelihood, DynamicProgramming
import torch.nn as nn

class LikelihoodAggregator(nn.Module):
    """聚合所有个体的似然函数，考虑所有随机效应组合"""
    def __init__(self, 
                 config: ModelConfig, 
                 individual_data: pd.DataFrame, 
                 regional_data: pd.DataFrame,
                 adjacency_matrix: np.ndarray,
                 distance_matrix: np.ndarray,
                 params: Optional[MigrationParameters] = None):
        super().__init__()
        self.config = config
        self.individual_data = individual_data
        self.regional_data = regional_data
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = distance_matrix
        self.all_pids = individual_data['pid'].unique().tolist()
        self.all_provinces = regional_data.index.tolist()
        
        # 如果没有提供参数，则创建一个新的参数对象
        if params is None:
            self.params = MigrationParameters(config)
        else:
            self.params = params
            
        # 创建动态规划求解器
        self.dp = DynamicProgramming(
            config=self.config,
            params=self.params,
            geo_data=self.regional_data,
            dismatrx=self.distance_matrix,
            adjmatrix=self.adjacency_matrix
        )
        
        # 预计算期望价值函数
        self.dp.calculate_ev(self.all_provinces)
        
    def forward(self) -> Tensor:
        """计算总体对数似然函数"""
        return self.log_likelihood()
    
    def log_likelihood(self) -> Tensor:
        """计算总体对数似然函数"""
        # 使用joblib并行计算个体似然
        individual_likelihoods = self._compute_individual_likelihoods()
        
        # 计算对数似然总和
        log_lik = torch.sum(torch.log(torch.stack(individual_likelihoods) + 1e-10))
        
        return log_lik
    
    def _compute_individual_likelihoods(self, n_jobs: int = -1) -> List[Tensor]:
        """并行计算所有个体的似然函数"""
        # 使用joblib并行计算
        if n_jobs != 1:
            likelihoods = Parallel(n_jobs=n_jobs)(delayed(self._compute_single_likelihood)(pid) for pid in self.all_pids)
        else:
            # 串行计算（用于调试）
            likelihoods = [self._compute_single_likelihood(pid) for pid in self.all_pids]
            
        return likelihoods
    
    def _compute_single_likelihood(self, pid: int) -> Tensor:
        """计算单个个体的似然函数"""
        try:
            individual_likelihood = IndividualLikelihood(
                pid=pid,
                data=self.individual_data,
                dp=self.dp,
                params=self.params,
                config=self.config
            )
            return individual_likelihood.calculate()
        except Exception as e:
            print(f"计算个体 {pid} 的似然函数时出错: {str(e)}")
            return torch.tensor(1e-10, requires_grad=True)  # 返回一个很小的值，避免log(0)
    
    def parameters(self):
        """返回模型参数"""
        return self.params.parameters()
    
    def named_parameters(self):
        """返回命名参数"""
        return self.params.named_parameters()
    
