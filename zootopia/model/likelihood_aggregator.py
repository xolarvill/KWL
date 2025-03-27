import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from model_config import ModelConfig
from migration_parameters import MigrationParameters
from dynamic_programming import DynamicProgramming
from individual_likelihood import IndividualLikelihood

# 聚合所有个体的似然函数，考虑所有随机效应组合
class LikelihoodAggregator:
    """聚合所有个体的似然函数，考虑所有随机效应组合"""
    def __init__(self, 
                 config: ModelConfig, 
                 individual_data: pd.DataFrame, 
                 regional_data: pd.DataFrame,
                 adjacency_matrix: np.ndarray,
                 dp: DynamicProgramming, 
                 params: MigrationParameters):
        self.config = config
        self.individual_data = individual_data
        self.regional_data = regional_data
        self.dp = dp

        if params is None:
            self.params = MigrationParameters(config)
        else:
            self.params = params
        
        # 获取所有个体ID和省份ID
        self.all_pids = data['pid'].unique().tolist()
        self.all_provinces = geo_data['provcd'].unique().tolist()
        
        # 初始化动态规划对象
        self.dp = DynamicProgramming(
            config=self.config,
            params=self.params,
            geo_data=self.geo_data,
            dismatrix=self.dismatrix,
            adjmatrix=self.adjmatrix
        )
        
        # 预计算期望价值函数
        self.dp.calculate_ev(self.all_provinces)
    
    def __call__(self, params_tensor: torch.Tensor) -> torch.Tensor:
        """目标函数：输入参数张量，返回负对数似然（标量）"""
        # 更新参数值
        self._update_parameters(params_tensor)
        
        # 重新计算期望价值函数（因为参数已更新）
        self.dp.calculate_ev(self.all_provinces)
        
        # 并行计算所有个体的对数似然
        log_liks = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._compute_individual_log_lik)(pid) 
            for pid in self.all_pids
        )
        
        # 聚合为总体对数似然
        total_log_lik = torch.sum(torch.stack(log_liks))
        return -total_log_lik  # 最小化负对数似然

    def _update_parameters(self, params_tensor: torch.Tensor) -> None:
        """将优化器的参数张量更新到MigrationParameters"""
        with torch.no_grad():
            for i, (name, param) in enumerate(self.params.named_parameters()):
                param.copy_(params_tensor[i])

    def _compute_individual_log_lik(self, pid: int) -> torch.Tensor:
        """计算单个个体的对数似然（数值稳定版本）"""
        individual = IndividualLikelihood(
            pid=pid, 
            data=self.data, 
            dp=self.dp, 
            params=self.params,
            config=self.config
        )
        lik = individual.calculate()
        log_lik = torch.log(lik + 1e-12)  # 防止log(0)
        return log_lik
    
    def parallel_likelihood(self) -> Tensor:
        """并行计算并返回所有个体的似然张量"""
        likelihoods = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._compute_single)(pid) for pid in self.all_pids
        )
        return torch.stack(likelihoods)

    def _compute_single(self, pid: int) -> Tensor:
        """单个个体的似然计算"""
        return IndividualLikelihood(
            pid=pid, 
            data=self.data, 
            dp=self.dp, 
            params=self.params,
            config=self.config
        ).calculate()
    
    def get_parameters_tensor(self) -> torch.Tensor:
        """获取当前参数的张量表示，用于优化器"""
        return torch.cat([param.flatten() for param in self.params.parameters()])
    
    def estimate(self, max_iter: int = None) -> MigrationParameters:
        """使用L-BFGS优化器估计参数"""
        from scipy.optimize import minimize
        from utils.std import compute_hessian, ParameterResults
        
        # 使用配置中的最大迭代次数，如果没有提供
        if max_iter is None:
            max_iter = self.config.max_iter
        
        # 获取初始参数
        initial_params = self.get_parameters_tensor().detach().numpy()
        
        # 使用SciPy的L-BFGS优化器
        result = minimize(
            fun=lambda x: self(torch.tensor(x, requires_grad=True)).item(),
            x0=initial_params,
            method='L-BFGS-B',
            jac=lambda x: torch.autograd.grad(self(torch.tensor(x, requires_grad=True)), 
                                             torch.tensor(x, requires_grad=True))[0].numpy(),
            options={'maxiter': max_iter}
        )
        
        # 更新最终参数
        final_params = torch.tensor(result.x)
        self._update_parameters(final_params)
        
        # 计算Hessian矩阵（如果需要）
        try:
            hessian = compute_hessian(self, self.params)
            # 生成结果对象
            results = ParameterResults(self.params, hessian)
            results.calculate_statistics()
            return results
        except:
            # 如果计算Hessian失败，直接返回参数
            return self.params