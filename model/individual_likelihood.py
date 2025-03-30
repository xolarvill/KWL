import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Tuple
from config import ModelConfig
from migration_parameters import MigrationParameters
from utils.indicator import isnot, inverse_isnot, isin, inverse_isin
from utils.visited import visited_sequence

class DynamicProgramming:
    """动态规划求解"""
    def __init__(self, 
                 config: ModelConfig,
                 params: MigrationParameters,
                 geo_data: pd.DataFrame,
                 distance_matrix: np.ndarray,
                 adjmatrix: np.ndarray):
        self.config = config
        self.params = params
        self.geo_data = geo_data
        self.dismatrix = distance_matrix
        self.adjmatrix = adjmatrix
        self.max_age = self.config.max_age
        self.discount_factor = self.config.discount_factor
        self.EV = {}  # 期望价值函数，格式：{(age, current_location, tau, eta_idx, nu_idx, xi_idx): Tensor}
        self.choice_probs = {}  # 选择概率，格式：{(age, current_location, tau, eta_idx, nu_idx, xi_idx): Tensor}
        
        # 支撑点索引范围
        self.n_eta = len(self.params.eta_support)
        self.n_nu = len(self.params.nu_support)
        self.n_xi = len(self.params.xi_support)
        self.n_sigma = len(self.params.sigmavarepsilon_support)

    def calculate_ev(self, all_provinces: List[int]) -> Dict[Tuple, Tensor]:
        """逆向计算每个(age, current_location, tau, eta_idx, nu_idx, xi_idx)的期望价值"""
        # 初始化终端期的价值函数
        for current_location in all_provinces:
            for tau in range(self.config.n_tau_types):
                for eta_idx in range(self.n_eta):
                    for nu_idx in range(self.n_nu):
                        for xi_idx in range(self.n_xi):
                            self.EV[(self.config.terminal_period, current_location, tau, eta_idx, nu_idx, xi_idx)] = torch.zeros(len(all_provinces))
        
        # 逆向归纳法计算期望价值函数
        for age in range(self.config.terminal_period - 1, 0, -1):
            for current_location in all_provinces:
                for tau in range(self.config.n_tau_types):
                    for eta_idx in range(self.n_eta):
                        for nu_idx in range(self.n_nu):
                            for xi_idx in range(self.n_xi):
                                # 计算当前状态下选择每个可能位置的即时效用和期望未来价值
                                utility = self._calculate_utility(age, current_location, tau, eta_idx, nu_idx, xi_idx, all_provinces)
                                
                                # 计算选择概率
                                logits = utility
                                probs = torch.nn.functional.softmax(logits, dim=0)
                                self.choice_probs[(age, current_location, tau, eta_idx, nu_idx, xi_idx)] = probs
                                
                                # 计算期望价值函数（LogSumExp形式）
                                self.EV[(age, current_location, tau, eta_idx, nu_idx, xi_idx)] = torch.logsumexp(utility, dim=0)
        
        return self.EV
    
    def _calculate_utility(self, 
                           age: int, 
                           current_location: int, 
                           tau: int,
                           eta_idx: int,
                           nu_idx: int,
                           xi_idx: int,
                           all_provinces: List[int]) -> Tensor:
        """计算从当前省份current_location迁移到j的效用"""
        utility = []
        
        # 获取对应支撑点的值
        eta = self.params.eta_support[eta_idx]
        nu = self.params.nu_support[nu_idx]
        xi = self.params.xi_support[xi_idx]
        
        for j in all_provinces:
            # 计算迁移成本
            migration_cost = self._calculate_migration_cost(age, current_location, j, tau)
            
            # 计算经济效益
            economic_utility = self._calculate_economic_utility(age, j, 0, eta, nu)  # 这里年份设为0，实际使用时需要传入正确的年份
            
            # 计算非经济效益（宜居度）
            amenity_utility = self._calculate_amenity_utility(j)
            
            # 计算恋家溢价和户籍制度障碍
            home_premium = self.params.alphaH * isnot(j, current_location)  # 恋家溢价
            hukou_penalty = self.params.alphaP * inverse_isnot(j, current_location)  # 户籍制度障碍
            
            # 计算地区偏好
            region_preference = xi  # 使用xi支撑点作为地区偏好
            
            # 计算总效用 = 经济效用 + 宜居度效用 + 恋家溢价 + 户籍制度障碍 + 地区偏好 - 迁移成本
            total_utility = (
                economic_utility +
                amenity_utility +
                home_premium +
                hukou_penalty +
                region_preference -
                migration_cost
            )
            
            # 加上贴现的期望未来价值
            if age < self.config.terminal_period:
                future_value = self.discount_factor * self.EV.get((age+1, j, tau, eta_idx, nu_idx, xi_idx), torch.tensor(0.0))
                total_utility += future_value
            
            utility.append(total_utility)
        
        return torch.tensor(utility)
    
    def _calculate_migration_cost(self, 
                                  age: int, 
                                  current_location: int, 
                                  j: int, 
                                  tau: int) -> Tensor:
        """计算迁移成本"""
        # 如果不迁移，成本为0
        if current_location == j:
            return torch.tensor(0.0)
        
        # 根据tau类型选择不同的gamma0值
        if tau == 0:
            gamma0 = self.params.gamma0_tau1
        elif tau == 1:
            gamma0 = self.params.gamma0_tau2
        else:
            gamma0 = self.params.gamma0_tau3
        
        # 计算迁移成本 = 基础成本 + 距离成本 + 邻近折扣 + 先前省份折扣 + 年龄效应 + 城市规模效应
        cost = self.params.gammaF * (
            gamma0 +  # 基础迁移成本（根据tau类型不同）
            self.params.gamma1 * self.dismatrix[current_location][j] +  # 距离衰减
            self.params.gamma2 * isin(j, self.adjmatrix[current_location]) +  # 邻近省份折扣
            self.params.gamma3 * isin(j, [current_location]) +  # 先前省份折扣
            self.params.gamma4 * age +  # 年龄对迁移成本的影响
            self.params.gamma5 * self.geo_data.loc[j, 'population']  # 城市规模对迁移成本的影响
        )
        
        return cost
    
    def _calculate_economic_utility(self, 
                                    age: int, 
                                    j: int, 
                                    year: int,
                                    eta: Tensor,
                                    nu: Tensor) -> Tensor:
        """计算经济效益"""
        # 地区平均经济水平
        region_mean = self.geo_data.loc[j, 'mean_wage']
        
        # 年龄效应（一次项和二次项）
        age_effect = self.params.r1 * age + self.params.r2 * (age ** 2)
        
        # 时间效应（假设当前年份为基准年份+age）
        time_effect = self.params.rt * year
        
        # 总经济效益
        economic_benefit = region_mean + age_effect + time_effect + eta + nu
        
        # 经济效益的边际效用
        economic_utility = self.params.alpha0 * economic_benefit
        
        return economic_utility
    
    def _calculate_amenity_utility(self, j: int) -> Tensor:
        """计算非经济效益（宜居度）"""
        # 房价效用
        house_price_utility = self.params.alpha1 * self.geo_data.loc[j, 'house_price']
        
        # 天气效用（包括自然灾害、温度、空气质量、水资源）
        weather_utility = self.params.alpha2 * self.geo_data.loc[j, 'weather']
        
        # 教育效用
        education_utility = self.params.alpha3 * self.geo_data.loc[j, 'education']
        
        # 健康效用
        health_utility = self.params.alpha4 * self.geo_data.loc[j, 'health']
        
        # 交通效用（包括公共交通和道路服务）
        traffic_utility = self.params.alpha5 * self.geo_data.loc[j, 'traffic']
        
        # 总宜居度效用
        amenity_utility = house_price_utility + weather_utility + education_utility + health_utility + traffic_utility
        
        return amenity_utility
    
    def get_choice_probability(self, 
                               age: int, 
                               current_location: int, 
                               tau: int,
                               eta_idx: int,
                               nu_idx: int,
                               xi_idx: int,
                               j: int) -> Tensor:
        """获取在特定状态下选择地点j的概率"""
        probs = self.choice_probs.get((age, current_location, tau, eta_idx, nu_idx, xi_idx), None)
        if probs is None:
            # 如果概率尚未计算，则计算它
            all_provinces = list(range(self.config.n_regions))
            utility = self._calculate_utility(age, current_location, tau, eta_idx, nu_idx, xi_idx, all_provinces)
            probs = torch.nn.functional.softmax(utility, dim=0)
            self.choice_probs[(age, current_location, tau, eta_idx, nu_idx, xi_idx)] = probs
        
        return probs[j]


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
        
        # 检查数据是否为空
        if self.n_periods == 0:
            raise ValueError(f"个体 {pid} 没有数据记录")
        
        # 获取个体的位置历史
        self.location_history = visited_sequence(data = self.data, pid = pid)

        # 支撑点索引范围
        self.n_eta = len(self.params.eta_support)
        self.n_nu = len(self.params.nu_support)
        self.n_xi = len(self.params.xi_support)
        self.n_sigma = len(self.params.sigmavarepsilon_support)

    def calculate(self) -> Tensor:
        """计算个体的总似然（考虑所有类型和支撑点组合的概率加权和）"""
        total_lik = torch.tensor(0.0, requires_grad=True)
        
        # 遍历所有可能的类型tau
        for tau in range(self.config.n_tau_types):
            tau_lik = torch.tensor(0.0, requires_grad=True)
            
            # 遍历所有可能的支撑点组合
            for eta_idx in range(self.n_eta):
                for nu_idx in range(self.n_nu):
                    for xi_idx in range(self.n_xi):
                        for sigma_idx in range(self.n_sigma):
                            # 计算该组合下的似然贡献
                            omega_lik = self._calculate_omega_likelihood(tau, eta_idx, nu_idx, xi_idx, sigma_idx)
                            
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
            total_lik += self.params.pi_tau[tau] * tau_lik
        
        return total_lik

    def _calculate_omega_likelihood(self, tau: int, eta_idx: int, nu_idx: int, xi_idx: int, sigma_idx: int) -> Tensor:
        """计算特定类型tau和支撑点组合下的似然"""
        log_lik = torch.tensor(0.0, requires_grad=True)  # 启用梯度计算
        
        # 获取该组合的支撑点值
        eta = self.params.eta_support[eta_idx]
        nu = self.params.nu_support[nu_idx]
        xi = self.params.xi_support[xi_idx]
        sigma_eps = self.params.sigmavarepsilon_support[sigma_idx]
        
        # 遍历个体的所有观测期
        for t in range(1, self.n_periods):  # 从第二期开始，因为需要前一期的位置
            current_row = self.data.iloc[t]
            previous_row = self.data.iloc[t-1]
            
            # 当前位置和前一期位置
            current_location = previous_row['provcd']  # 前一期位置
            j = current_row['provcd']      # 当前位置
            age = current_row['age']       # 当前年龄
            
            # 计算迁移选择概率
            choice_prob = self.dp.get_choice_probability(age, current_location, tau, eta_idx, nu_idx, xi_idx, j)
            
            # 如果有工资观测值，计算工资观测概率
            if 'income' in current_row and not pd.isna(current_row['income']):
                wage_prob = self._calculate_wage_probability(current_row, eta, nu, sigma_eps)
                log_lik += torch.log(choice_prob + 1e-10) + torch.log(wage_prob + 1e-10)  # 添加小常数避免log(0)
            else:
                log_lik += torch.log(choice_prob + 1e-10)  # 添加小常数避免log(0)
        
        return torch.exp(log_lik)  # 返回概率值（非对数）
    
    def _calculate_wage_probability(self, row: pd.Series, eta: Tensor, nu: Tensor, sigma_eps: Tensor) -> Tensor:
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
        base_year = getattr(self.config, 'base_year', 2010)  # 默认基准年为2010
        time_effect = self.params.rt * (year - base_year)
        
        # 预期工资 = 地区平均 + 年龄效应 + 时间效应 + 个体固定效应 + 个体-地区匹配效应
        expected_wage = region_mean + age_effect + time_effect + eta + nu
        
        # 计算工资观测概率（正态分布）
        wage_prob = torch.exp(-0.5 * ((observed_wage - expected_wage) / sigma_eps) ** 2) / \
                    (sigma_eps * torch.sqrt(torch.tensor(2 * np.pi)))
        
        return wage_prob