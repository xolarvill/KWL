import torch
import pandas as pd
from typing import List
from joblib import Parallel, delayed

class TotalLogLikelihood:
    """计算总体样本的对数似然，整合所有个体结果"""
    def __init__(
        self,
        all_pids: List[int],
        data: pd.DataFrame,
        params: MigrationParameters,
        all_provinces: List[int],
        max_age: int = 60,
        n_jobs: int = -1
    ):
        self.all_pids = all_pids
        self.data = data
        self.params = params
        self.n_jobs = n_jobs
        self.dp = DynamicProgramming(params, max_age)  # 动态规划预计算
        self.dp.calculate_ev(all_provinces)            # 预计算期望价值函数

    def __call__(self, params_tensor: torch.Tensor) -> torch.Tensor:
        """目标函数：输入参数张量，返回负对数似然（标量）"""
        # 更新参数值
        self._update_parameters(params_tensor)
        
        # 并行计算所有个体的对数似然
        log_liks = Parallel(n_jobs=self.n_jobs)(
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
        individual = IndividualLikelihood(pid, self.data, self.dp, self.params)
        log_lik = torch.log(individual.calculate() + 1e-12)  # 防止log(0)
        return log_lik