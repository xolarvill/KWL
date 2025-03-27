import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Tuple
from model_config import ModelConfig
from migration_parameters import MigrationParameters
from dynamic_programming import DynamicProgramming
from visited import calK_all_time

class IndividualLikelihood:
    """计算单个个体的似然函数，考虑所有随机效应组合"""
    def __init__(self, 
                 pid: int, 
                 data: pd.DataFrame, 
                 dp: DynamicProgramming, 
                 params: MigrationParameters,
                 config: ModelConfig):
        self.pid = pid
        self.data = data[data['pid'] == pid].sort_values('year')
        self.dp = dp
        self.params = params
        self.config = config
        self.n_periods = len(self.data)
        
        # 获取个体的位置历史
        self.location_history = calK_all_time(self.data, pid)

    def calculate(self) -> Tensor:
        """计算个体的总似然（考虑所有类型的概率加权和）"""
        total_lik = torch.tensor(0.0)
        
        # 遍历所有可能的类型tau
        for tau in range(self.config.n_tau_types):
            # 计算该类型下的似然贡献
            lik_tau = self._calculate_type_likelihood(tau)
            
            # 加权求和（使用类型概率）
            total_lik += self.params.pi_tau[tau] * lik_tau
        
        return total_lik

    def _calculate_type_likelihood(self, tau: int) -> Tensor:
        """计算特定类型tau下的似然"""
        log_lik = torch.tensor(0.0)
        
        # 获取该类型的固定效应和匹配效应支撑点
        eta = self.params.eta_support[tau]  # 个体固定效应
        nu = self.params.nu_support[tau]    # 个体-地区匹配效应
        xi = self.params.xi_support[tau]    # 地区偏好效应
        sigma_eps = self.params.sigmavarepsilon_support[tau]  # 暂态效应标准差
        
        # 遍历个体的所有观测期
        for t in range(1, self.n_periods):  # 从第二期开始，因为需要前一期的位置
            current_row = self.data.iloc[t]
            previous_row = self.data.iloc[t-1]
            
            # 当前位置和前一期位置
            csta = previous_row['provcd']  # 前一期位置
            j = current_row['provcd']      # 当前位置
            age = current_row['age']       # 当前年龄
            
            # 计算迁移选择概率
            choice_prob = self._calculate_choice_probability(age, csta, tau, j)
            
            # 如果有工资观测值，计算工资观测概率
            if 'income' in current_row and not pd.isna(current_row['income']):
                wage_prob = self._calculate_wage_probability(current_row, tau, eta, nu, sigma_eps)
                log_lik += torch.log(choice_prob) + torch.log(wage_prob)
            else:
                log_lik += torch.log(choice_prob)
        
        return torch.exp(log_lik)  # 返回概率值（非对数）
    
    def _calculate_choice_probability(self, age: int, csta: int, tau: int, j: int) -> Tensor:
        """计算在特定状态下选择地点j的概率"""
        # 从动态规划对象中获取选择概率
        return self.dp.get_choice_probability(age, csta, tau, j)
    
    def _calculate_wage_probability(self, row: pd.Series, tau: int, eta: Tensor, nu: Tensor, sigma_eps: Tensor) -> Tensor:
        """计算工资观测概率（正态分布）"""
        j = row['provcd']  # 当前位置
        age = row['age']   # 当前年龄
        year = row['year'] # 当前年份
        observed_wage = row['income']  # 观测到的工资
        
        # 计算预期工资
        # 地区平均经济水平
        region_mean = self.dp.geo_data.loc[j, 'mean_wage']
        
        # 年龄效应（一次项和二次项）
        age_effect = self.params.r1 * age + self.params.r2 * (age ** 2)
        
        # 时间效应
        time_effect = self.params.rt * (year - self.config.base_year)
        
        # 预期工资 = 地区平均 + 年龄效应 + 时间效应 + 个体固定效应 + 个体-地区匹配效应
        expected_wage = region_mean + age_effect + time_effect + eta + nu
        
        # 计算工资观测概率（正态分布）
        wage_prob = torch.exp(-0.5 * ((observed_wage - expected_wage) / sigma_eps) ** 2) / \
                    (sigma_eps * torch.sqrt(torch.tensor(2 * np.pi)))
        
        return wage_prob