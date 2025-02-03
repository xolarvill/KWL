import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List
from joblib import Parallel, delayed

# 避免未定义
dismatrix = []
adjmatrix = []
GeoData = []

class MigrationParameters:
    """封装所有待估参数，支持自动微分"""
    def __init__(self):
        # u(x,j)
        self.alpha0 = torch.nn.Parameter(torch.tensor(0.8)) # wage income parameter 
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.8)) # houseprice
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.8)) # weather = hazard + temperature + air quality + water supply
        self.alpha3 = torch.nn.Parameter(torch.tensor(0.8)) # education 
        self.alpha4 = torch.nn.Parameter(torch.tensor(0.8)) # health
        self.alpha5 = torch.nn.Parameter(torch.tensor(0.8)) # traffic = public transportation + road service
        self.alphaH = torch.nn.Parameter(torch.tensor(0.1)) # home premium parameter
        self.xi = torch.nn.Parameter(torch.tensor(0.1)) # random permanent component
        self.zeta = torch.nn.Parameter(torch.tensor(0.1)) # exogenous shock
        
        # wage
        self.nu = torch.nn.Parameter(torch.tensor(0.8)) # 
        self.eta = torch.nn.Parameter(torch.tensor(0.8)) # 
        self.beta_amenity = torch.nn.Parameter(torch.tensor(0.3))     # 城市宜居度系数
        self.sigma_eps = torch.nn.Parameter(torch.tensor(0.5))        # 暂态效应标准差
        
        # 迁移成本参数（gamma系列）
        self.gammaF = torch.nn.Parameter(torch.tensor(0.8)) # mirgration friction parameter
        self.gamma0 = torch.nn.Parameter(torch.tensor(0.5)) # heterogenous friction parameter 
        self.gamma1 = torch.nn.Parameter(torch.tensor(-0.1)) # 距离衰减系数
        self.gamma2 = torch.nn.Parameter(torch.tensor(0.5)) # 邻近省份折扣
        self.gamma3 = torch.nn.Parameter(torch.tensor(0.8)) # 先前省份折扣
        self.gamma4 = torch.nn.Parameter(torch.tensor(0.05)) # 年龄对迁移成本的影响
        self.gamma5 = torch.nn.Parameter(torch.tensor(0.8)) # 更大的城市更便宜
        
        # 固定效应支持点（离散化）
        self.eta_support = torch.tensor([-1.0, 0.0, 1.0])             # 个体固定效应（3个支持点）
        self.nu_support = torch.tensor([-0.5, 0.0, 0.5])              # 地区匹配效应（3个支持点）

    def to_dict(self) -> Dict[str, Tensor]:
        """转换为参数字典，用于数值计算"""
        return {name: param.data for name, param in self.named_parameters()}

    def named_parameters(self) -> Dict[str, Tensor]:
        """获取所有可训练参数"""
        return dict(self.named_parameters())
    

class DynamicProgramming:
    """预计算期望价值函数（EV），逆向归纳法"""
    def __init__(self, params: MigrationParameters, max_age: int = 60):
        self.params = params
        self.max_age = max_age
        self.EV = {}  # 期望价值函数，格式：{(age, csta): Tensor}

    def calculate_ev(self, all_provinces: List[int]) -> Dict[tuple, Tensor]:
        """逆向计算每个(age, csta)的期望价值"""
        for age in range(self.max_age, 0, -1):
            for csta in all_provinces:
                # 计算所有可能迁移到j的效用，并取LogSumExp（假设Logit选择）
                utility = self._calculate_utility(age, csta, all_provinces)
                self.EV[(age, csta)] = torch.logsumexp(utility, dim=0)  # LogSumExp
        return self.EV

    def _calculate_utility(self, age: int, csta: int, all_provinces: List[int]) -> Tensor:
        """计算从当前省份csta迁移到j的效用"""
        utility = []
        for j in all_provinces:
            # 迁移成本 = 基础成本 + 距离成本 + 邻近折扣 + 年龄效应
            cost = self.params.gammaF * (
                self.params.gamma0 + 
                self.params.gamma1 * dismatrix[csta][j] +
                self.params.gamma2 * dismatrix[csta][j] +
                self.params.gamma3 * adjmatrix[csta][j] +
                self.params.gamma4 * age +
                self.params.gamma5 * 10 
            )
            # 总效用 = 收入效用 + 宜居度效用 - 迁移成本 + 期望价值（下一期）
            u = (
                self.params.alpha_income * GeoData.loc[j, 'mean_wage'] +
                self.params.beta_amenity * GeoData.loc[j, 'amenity'] -
                cost +
                (self.EV.get((age+1, j), 0.0) if age < self.max_age else 0.0)
            )
            utility.append(u)
        return torch.tensor(utility)
    
class IndividualLikelihood:
    """计算单个个体的似然函数，考虑所有随机效应组合"""
    def __init__(self, pid: int, data: pd.DataFrame, dp: DynamicProgramming, params: MigrationParameters):
        self.pid = pid
        self.data = data[data['pid'] == pid].sort_values('year')
        self.dp = dp
        self.params = params
        self.n_periods = len(self.data)

    def calculate(self) -> Tensor:
        """计算个体的总似然（离散化随机效应求和）"""
        total_lik = torch.tensor(0.0)
        
        # 遍历所有固定效应和地区匹配效应的支持点组合
        for eta in self.params.eta_support:
            for nu in self.params.nu_support:
                # 计算该组合下的似然贡献（并行加速点）
                lik = self._calculate_single_combination(eta, nu)
                total_lik += lik / (len(self.params.eta_support) * len(self.params.nu_support))
        return total_lik

    def _calculate_single_combination(self, eta: Tensor, nu: Tensor) -> Tensor:
        """计算特定eta和nu组合下的似然"""
        log_lik = torch.tensor(0.0)
        for t in range(self.n_periods):
            row = self.data.iloc[t]
            csta = row['provcd']
            j = row['next_provcd']  # 假设数据中包含下一期迁移到的省份
            
            # 迁移选择概率（从EV中获取）
            utility = self.dp.EV.get((row['age'], csta), torch.tensor(0.0))
            rho_j = torch.exp(utility[j] - torch.logsumexp(utility, dim=0))
            
            # 工资观测概率（正态分布）
            expected_wage = (
                GeoData.loc[csta, 'mean_wage'] +
                self.params.alpha_income * row['age'] +  # 假设G(X,a)是线性年龄效应
                eta + nu
            )
            wage_prob = torch.exp(-0.5 * ((row['income'] - expected_wage) / self.params.sigma_eps) ** 2) / (self.params.sigma_eps * torch.sqrt(2 * torch.tensor(np.pi)))
            
            log_lik += torch.log(rho_j) + torch.log(wage_prob)
        return torch.exp(log_lik)  # 返回概率值（非对数）
    

class LikelihoodAggregator:
    """并行计算所有个体的似然"""
    def __init__(self, all_pids: List[int], data: pd.DataFrame, dp: DynamicProgramming, params: MigrationParameters):
        self.all_pids = all_pids
        self.data = data
        self.dp = dp
        self.params = params

    def parallel_likelihood(self, n_jobs: int = -1) -> Tensor:
        """并行计算并返回所有个体的似然张量"""
        likelihoods = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_single)(pid) for pid in self.all_pids
        )
        return torch.stack(likelihoods)

    def _compute_single(self, pid: int) -> Tensor:
        """单个个体的似然计算"""
        return IndividualLikelihood(pid, self.data, self.dp, self.params).calculate()