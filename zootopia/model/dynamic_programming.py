import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List
from joblib import Parallel, delayed
from model import ModelConfig, MigrationParameters
from utils.indicator import isnot, inverse_isnot, isin, inverse_isin

# 直接计算效用函数并没有意义，应该在DP中结合实例计算
class DynamicProgramming:
    """动态规划求解"""
    def __init__(self, 
                 config: ModelConfig,
                 params: MigrationParameters,
                 geo_data: pd.DataFrame,
                 dismatrx: np.ndarray,
                 adjmatrix: np.ndarray):
        self.config = config
        self.params = params
        self.geo_data = geo_data
        self.dismatrix = dismatrix
        self.adjmatrix = adjmatrix
        self.max_age = self.config.max_age
        self.discount_factor = self.config.discount_factor
        self.EV = {}  # 期望价值函数，格式：{(age, csta, tau): Tensor}
        self.choice_probs = {}  # 选择概率，格式：{(age, csta, tau): Tensor}

    def calculate_ev(self, all_provinces: List[int]) -> Dict[Tuple, Tensor]:
        """逆向计算每个(age, csta, tau)的期望价值"""
        # 初始化终端期的价值函数
        for csta in all_provinces:
            for tau in range(self.config.n_tau_types):
                self.EV[(self.config.terminal_period, csta, tau)] = torch.zeros(len(all_provinces))
        
        # 逆向归纳法计算期望价值函数
        for age in range(self.config.terminal_period-1, 0, -1):
            for csta in all_provinces:
                for tau in range(self.config.n_tau_types):
                    # 计算当前状态下选择每个可能位置的即时效用和期望未来价值
                    utility = self._calculate_utility(age, csta, tau, all_provinces)
                    
                    # 计算选择概率
                    logits = utility
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    self.choice_probs[(age, csta, tau)] = probs
                    
                    # 计算期望价值函数（LogSumExp形式）
                    self.EV[(age, csta, tau)] = torch.logsumexp(utility, dim=0)
        
        return self.EV
    
    def _calculate_utility(self, age: int, csta: int, tau: int, all_provinces: List[int]) -> Tensor:
        """计算从当前省份csta迁移到j的效用"""
        utility = []
        
        for j in all_provinces:
            # 计算迁移成本
            migration_cost = self._calculate_migration_cost(age, csta, j, tau)
            
            # 计算经济效益
            economic_utility = self._calculate_economic_utility(age, j, tau)
            
            # 计算非经济效益（宜居度）
            amenity_utility = self._calculate_amenity_utility(j, tau)
            
            # 计算恋家溢价和户籍制度障碍
            home_premium = self.params.alphaH * chi_isnot(j, csta)  # 恋家溢价
            hukou_penalty = self.params.alphaP * chi_invisnot(j, csta)  # 户籍制度障碍
            
            # 计算地区偏好
            region_preference = self.params.xi_support[tau]  # 使用tau类型的地区偏好支撑点
            
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
                future_value = self.discount_factor * self.EV.get((age+1, j, tau), torch.tensor(0.0))
                total_utility += future_value
            
            utility.append(total_utility)
        
        return torch.tensor(utility)
    
    def _calculate_migration_cost(self, age: int, csta: int, j: int, tau: int) -> Tensor:
        """计算迁移成本"""
        # 如果不迁移，成本为0
        if csta == j:
            return torch.tensor(0.0)
        
        # 计算迁移成本 = 基础成本 + 距离成本 + 邻近折扣 + 先前省份折扣 + 年龄效应 + 城市规模效应
        cost = self.params.gammaF * (
            self.params.gamma0 +  # 基础迁移成本
            self.params.gamma1 * self.dismatrix[csta][j] +  # 距离衰减
            self.params.gamma2 * chi_inout(j, self.adjmatrix[csta]) +  # 邻近省份折扣
            self.params.gamma3 * chi_inout(j, [csta]) +  # 先前省份折扣
            self.params.gamma4 * age +  # 年龄对迁移成本的影响
            self.params.gamma5 * self.geo_data.loc[j, 'population']  # 城市规模对迁移成本的影响
        )
        
        return cost
    
    def _calculate_economic_utility(self, age: int, j: int, tau: int) -> Tensor:
        """计算经济效益"""
        # 地区平均经济水平
        region_mean = self.geo_data.loc[j, 'mean_wage']
        
        # 年龄效应（一次项和二次项）
        age_effect = self.params.r1 * age + self.params.r2 * (age ** 2)
        
        # 时间效应（假设当前年份为基准年份+age）
        time_effect = self.params.rt * age
        
        # 个体固定效应和个体-地区匹配效应（使用tau类型的支撑点）
        eta = self.params.eta_support[tau]  # 个体固定效应
        nu = self.params.nu_support[tau]  # 个体-地区匹配效应
        
        # 总经济效益
        economic_benefit = region_mean + age_effect + time_effect + eta + nu
        
        # 经济效益的边际效用
        economic_utility = self.params.alpha0 * economic_benefit
        
        return economic_utility
    
    def _calculate_amenity_utility(self, j: int, tau: int) -> Tensor:
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
    
    def get_choice_probability(self, age: int, csta: int, tau: int, j: int) -> Tensor:
        """获取在特定状态下选择地点j的概率"""
        probs = self.choice_probs.get((age, csta, tau), None)
        if probs is None:
            # 如果概率尚未计算，则计算它
            utility = self._calculate_utility(age, csta, tau, list(range(len(probs))))
            probs = torch.nn.functional.softmax(utility, dim=0)
            self.choice_probs[(age, csta, tau)] = probs
        
        return probs[j]