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
                 adjacency_matrix: np.ndarray,
                 linguistic_matrix: np.ndarray):
        # 接受输入
        self.config = config
        self.params = params
        self.geo_data = geo_data
        self.dismatrix = distance_matrix
        self.adjmatrix = adjacency_matrix
        self.linguistic_matrix = linguistic_matrix
        
        # 从config中获取必要参数
        self.age_max = self.config.age_max
        self.discount_factor = self.config.discount_factor
        
        # 支撑点索引范围
        self.n_eta = self.config.n_eta_support_points
        self.n_nu = self.config.n_nu_support_points
        self.n_xi = self.config.n_xi_support_points
        self.n_sigma = self.config.n_sigmavarepsilon_support_points
        
        # 暂存container
        self.EV = {}  # 期望价值函数，格式：{(age, current_location, tau, eta_idx, nu_idx, xi_idx): Tensor}
        self.choice_probs = {}  # 选择概率，格式：{(age, current_location, tau, eta_idx, nu_idx, xi_idx): Tensor}

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
        if tau == 1:
            gamma0 = self.params.gamma0_tau1
        elif tau == 2:
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

    def _calculate_amenity_utility(self, year: int, j: int) -> Tensor:
        """计算非经济效益（宜居度）"""
        # 获取特定年份的地区数据
        year_data = self.geo_data[self.geo_data['year'] == year]
        location_data = year_data[year_data['provcd'] == j]
        
        # 房价效用
        house_price_utility = self.params.alpha1 * location_data['house_price'].values[0]
        
        # 天气效用（包括自然灾害、温度、空气质量、水资源）
        environment_utility = self.params.alpha2 * location_data['weather'].values[0]
        
        # 教育
        education_utility = self.params.alpha3 * location_data['education'].values[0]
        
        # 医疗
        health_utility = self.params.alpha4 * location_data['health'].values[0]
        
        # 商业
        business_utility = self.params.alpha5 * location_data['business'].values[0]
        
        # 方言
        cultural_utility = self.params.alpha6 * self.linguistic_matrix[j,0]
        
        # 公共设施
        public_utility = self.params.alpha7 * location_data['public'].values[0]
        
        # 计算线形总宜居度效用
        amenity_utility = house_price_utility + environment_utility + education_utility + health_utility + business_utility + cultural_utility + public_utility
        
        return amenity_utility
    
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
    
    def get_rho_probability(self, 
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
    """个体似然函数计算"""
    def __init__(self, 
                 individual_data: pd.DataFrame,  # 已经按pid分割好的个体数据
                 dp: DynamicProgramming, 
                 params: MigrationParameters,
                 config: ModelConfig):
        """
        初始化个体似然函数计算器
        
        参数:
            individual_data (pd.DataFrame): 单个个体的历史轨迹数据
            dp (DynamicProgramming): 动态规划求解器
            params (MigrationParameters): 模型参数
            config (ModelConfig): 模型配置
        """
        self.data = individual_data
        self.dp = dp
        self.params = params
        self.config = config
        
        # 获取个体的hukou信息
        self.hukou = individual_data['hukou'].iloc[0]  # 假设同一个体的hukou不变
        
        # 获取个体的pid
        self.pid = individual_data['pid'].iloc[0]
        
        # 获取个体的观测序列
        self.observed_sequence = visited_sequence(individual_data)
        
        # 获取个体的财富序列
        self.wealth_sequence = individual_data['wealth'].values
        
        # 获取个体的年龄序列
        self.age_sequence = individual_data['age'].values
        
        # 获取个体的年份序列
        self.year_sequence = individual_data['year'].values
        
        # 获取个体的地点序列
        self.location_sequence = individual_data['location'].values

    def _calculate_utility(self, 
                           age: int, 
                           current_location: int, 
                           j: int,
                           year: int,
                           tau: int,
                           eta: Tensor,
                           nu: Tensor,
                           xi: Tensor) -> Tensor:
        """计算效用函数"""
        # 计算迁移成本
        migration_cost = self.dp._calculate_migration_cost(age, current_location, j, tau)
        
        # 计算经济效益
        economic_utility = self.dp._calculate_economic_utility(age, j, year, eta, nu)
        
        # 计算非经济效益
        amenity_utility = self.dp._calculate_amenity_utility(year, j)
        
        # 计算恋家溢价（如果当前地点不是户籍所在地）
        # home_premium = self.params.alphaH * isnot(j, self.hukou)  # 注释掉恋家溢价计算
        
        # 总效用
        total_utility = economic_utility + amenity_utility - migration_cost  # 移除 home_premium
        
        return total_utility

    def _calculate_wealth_probability(self, row: pd.Series, eta: Tensor, nu: Tensor, sigma_eps: Tensor) -> Tensor:
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
    
    def _calculate_omega_likelihood(self, tau: int, eta_idx: int, nu_idx: int, xi_idx: int, sigma_idx: int) -> Tensor:
        """计算特定类型tau和支撑点组合下的似然"""
        log_lik = torch.tensor(0.0, requires_grad=True)  # 启用梯度计算
        
        # 获取该组合的支撑点值
        eta = self.params.eta_support[eta_idx]
        nu = self.params.nu_support[nu_idx]
        xi = self.params.xi_support[xi_idx]
        sigma_eps = self.params.sigmavarepsilon_support[sigma_idx]
        
        # 遍历个体的所有观测期
        for t in range(1, len(self.data)):  # 从第二期开始，因为需要前一期的位置
            current_row = self.data.iloc[t]
            previous_row = self.data.iloc[t-1]
            
            # 当前位置和前一期位置
            current_location = previous_row['provcd']  # 前一期位置
            j = current_row['provcd']      # 当前位置
            age = current_row['age']       # 当前年龄
            
            # 计算迁移选择概率
            choice_prob = self.dp.get_rho_probability(age, current_location, tau, eta_idx, nu_idx, xi_idx, j)
            
            # 如果有工资观测值，计算工资观测概率
            if 'income' in current_row and not pd.isna(current_row['income']):
                wage_prob = self._calculate_wealth_probability(current_row, eta, nu, sigma_eps)
                log_lik += torch.log(choice_prob + 1e-10) + torch.log(wage_prob + 1e-10)  # 添加小常数避免log(0)
            else:
                log_lik += torch.log(choice_prob + 1e-10)  # 添加小常数避免log(0)
        
        return torch.exp(log_lik)  # 返回概率值（非对数）

    def calculate(self) -> Tensor:
        """计算个体的总似然（考虑所有类型和支撑点组合的概率加权和）"""
        total_lik = torch.tensor(0.0, requires_grad=True)
        
        # 遍历所有可能的类型tau
        for tau in range(self.config.n_tau_types):
            tau_lik = torch.tensor(0.0, requires_grad=True)
            
            # 遍历所有可能的支撑点组合
            for eta_idx in range(len(self.params.eta_support)):
                for nu_idx in range(len(self.params.nu_support)):
                    for xi_idx in range(len(self.params.xi_support)):
                        for sigma_idx in range(len(self.params.sigmavarepsilon_support)):
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
 