import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from config import ModelConfig
from .migration_parameters import MigrationParameters
from ..utils.indicator import isnot
from ..utils.visited import visited_sequence

class DynamicProgramming:
    """
    动态规划求解器。
    负责计算模型的期望价值函数(EV)和选择概率。
    这部分计算是通用的，不针对任何特定个体。
    """
    def __init__(self, 
                 config: ModelConfig,
                 params: MigrationParameters,
                 geo_data: pd.DataFrame,
                 distance_matrix: np.ndarray,
                 adjacency_matrix: np.ndarray,
                 linguistic_matrix: np.ndarray):
        self.config = config
        self.params = params
        self.geo_data = geo_data
        self.dismatrix = distance_matrix
        self.adjmatrix = adjacency_matrix
        self.linguistic_matrix = linguistic_matrix
        self.age_max = config.age_max
        self.discount_factor = config.discount_factor
        self.n_eta = config.n_eta_support_points
        self.n_nu = config.n_nu_support_points
        self.n_xi = config.n_xi_support_points
        
        self.EV = {}
        self.choice_probs = {}

    def _calculate_migration_cost(self, age: int, current_location: int, j: int, tau: int) -> Tensor:
        if current_location == j:
            return torch.tensor(0.0)
        gamma0 = getattr(self.params, f'gamma0_tau{tau+1}', self.params.gamma0)
        cost = self.params.gammaF * (
            gamma0 +
            self.params.gamma1 * self.dismatrix[current_location][j] +
            self.params.gamma2 * (1.0 if j in self.adjmatrix[current_location] else 0.0) +
            self.params.gamma3 * (1.0 if j == current_location else 0.0) +
            self.params.gamma4 * age +
            self.params.gamma5 * self.geo_data.loc[j, 'population']
        )
        return cost

    def _calculate_economic_utility(self, age: int, j: int, year: int, eta: Tensor, nu: Tensor) -> Tensor:
        region_mean = self.geo_data.loc[j, 'mean_wage']
        age_effect = self.params.r1 * age + self.params.r2 * (age ** 2)
        time_effect = self.params.rt * year
        economic_benefit = region_mean + age_effect + time_effect + eta + nu
        return self.params.alpha0 * economic_benefit

    def _calculate_amenity_utility(self, year: int, j: int) -> Tensor:
        location_data = self.geo_data[(self.geo_data['year'] == year) & (self.geo_data['provcd'] == j)]
        if location_data.empty: return torch.tensor(0.0)
        
        amenity_utility = (
            self.params.alpha1 * location_data['house_price'].values[0] +
            self.params.alpha2 * location_data['weather'].values[0] +
            self.params.alpha3 * location_data['education'].values[0] +
            self.params.alpha4 * location_data['health'].values[0] +
            self.params.alpha5 * location_data['business'].values[0] +
            self.params.alpha6 * self.linguistic_matrix[j, 0] +
            self.params.alpha7 * location_data['public'].values[0]
        )
        return amenity_utility

    def _calculate_utility(self, 
                           age: int, 
                           current_location: int, 
                           tau: int,
                           eta_idx: int,
                           nu_idx: int,
                           xi_idx: int,
                           all_provinces: List[int],
                           hukou: Optional[int] = None) -> Tensor:
        """
        计算效用。
        如果提供了hukou，则计算户籍惩罚。在计算通用EV时，hukou为None。
        """
        utility = []
        eta = self.params.eta_support[eta_idx]
        nu = self.params.nu_support[nu_idx]
        xi = self.params.xi_support[xi_idx]

        for j in all_provinces:
            migration_cost = self._calculate_migration_cost(age, current_location, j, tau)
            economic_utility = self._calculate_economic_utility(age, j, 0, eta, nu)
            amenity_utility = self._calculate_amenity_utility(0, j)
            home_premium = self.params.alphaH * isnot(j, current_location)
            
            # 户籍惩罚：仅当hukou被提供时才计算
            hukou_penalty = self.params.alphaP * isnot(j, hukou) if hukou is not None else torch.tensor(0.0)

            total_utility = (
                economic_utility +
                amenity_utility +
                home_premium +
                xi -
                migration_cost - 
                hukou_penalty
            )
            
            if age < self.config.terminal_period:
                future_value = self.discount_factor * self.EV.get((age + 1, j, tau, eta_idx, nu_idx, xi_idx), torch.tensor(0.0))
                total_utility += future_value
            
            utility.append(total_utility)
        
        return torch.stack(utility)

    def calculate_ev(self, all_provinces: List[int]):
        """逆向归纳计算期望价值函数EV。"""
        for age in range(self.config.terminal_period, 0, -1):
            for loc in all_provinces:
                for tau in range(self.config.n_tau_types):
                    for eta_idx in range(self.n_eta):
                        for nu_idx in range(self.n_nu):
                            for xi_idx in range(self.n_xi):
                                state = (age, loc, tau, eta_idx, nu_idx, xi_idx)
                                if age == self.config.terminal_period:
                                    self.EV[state] = torch.tensor(0.0)
                                    continue
                                
                                # 计算EV时不考虑特定hukou
                                utility = self._calculate_utility(age, loc, tau, eta_idx, nu_idx, xi_idx, all_provinces, hukou=None)
                                self.EV[state] = torch.logsumexp(utility, dim=0)
                                # 缓存通用的选择概率
                                self.choice_probs[state] = torch.nn.functional.softmax(utility, dim=0)

    def get_choice_probability(self, age: int, current_location: int, tau: int, eta_idx: int, nu_idx: int, xi_idx: int, j: int, hukou: int) -> Tensor:
        """
        获取特定个体（有hukou信息）的选择概率。
        这部分不能使用缓存，因为它依赖于特定个体的hukou。
        """
        all_provinces = list(range(self.config.n_regions))
        utility = self._calculate_utility(age, current_location, tau, eta_idx, nu_idx, xi_idx, all_provinces, hukou=hukou)
        probs = torch.nn.functional.softmax(utility, dim=0)
        return probs[j]

class IndividualLikelihood:
    """计算单个个体的似然函数。"""
    def __init__(self, 
                 individual_data: pd.DataFrame,
                 dp: DynamicProgramming, 
                 params: MigrationParameters,
                 config: ModelConfig):
        self.data = individual_data
        self.dp = dp
        self.params = params
        self.config = config
        self.hukou = individual_data['hukou'].iloc[0]
        self.pid = individual_data['pid'].iloc[0]

    def _calculate_wage_probability(self, row: pd.Series, eta: Tensor, nu: Tensor, sigma_eps: Tensor) -> Tensor:
        """计算工资观测概率。"""
        j, age, year, observed_wage = row['provcd'], row['age'], row['year'], row['income']
        
        region_mean = self.dp.geo_data.loc[j, 'mean_wage']
        age_effect = self.params.r1 * age + self.params.r2 * (age ** 2)
        time_effect = self.params.rt * (year - self.config.base_year)
        expected_wage = region_mean + age_effect + time_effect + eta + nu
        
        return torch.exp(-0.5 * ((observed_wage - expected_wage) / sigma_eps) ** 2) / (sigma_eps * torch.sqrt(torch.tensor(2 * np.pi)))

    def _calculate_omega_likelihood(self, tau: int, eta_idx: int, nu_idx: int, xi_idx: int, sigma_idx: int) -> Tensor:
        """计算在特定异质性组合(omega)下的似然。"""
        log_lik = torch.tensor(0.0, requires_grad=True)
        eta = self.params.eta_support[eta_idx]
        nu = self.params.nu_support[nu_idx]
        sigma_eps = self.params.sigmavarepsilon_support[sigma_idx]

        for t in range(1, len(self.data)):
            prev_row = self.data.iloc[t-1]
            curr_row = self.data.iloc[t]
            
            loc, j, age = prev_row['provcd'], curr_row['provcd'], curr_row['age']
            
            # 获取选择概率时，传入该个体的hukou
            choice_prob = self.dp.get_choice_probability(age, loc, tau, eta_idx, nu_idx, xi_idx, j, self.hukou)
            
            log_lik_t = torch.log(choice_prob + 1e-20)
            if 'income' in curr_row and pd.notna(curr_row['income']):
                wage_prob = self._calculate_wage_probability(curr_row, eta, nu, sigma_eps)
                log_lik_t += torch.log(wage_prob + 1e-20)
            
            log_lik = log_lik + log_lik_t
        
        return torch.exp(log_lik)

    def calculate(self) -> Tensor:
        """计算个体的总似然（对所有异质性进行积分）。"""
        total_lik = torch.tensor(0.0, requires_grad=True)
        
        for tau in range(self.config.n_tau_types):
            tau_lik = torch.tensor(0.0, requires_grad=True)
            
            for eta_idx in range(self.config.n_eta_support_points):
                for nu_idx in range(self.config.n_nu_support_points):
                    for xi_idx in range(self.config.n_xi_support_points):
                        for sigma_idx in range(self.config.n_sigmavarepsilon_support_points):
                            
                            omega_lik = self._calculate_omega_likelihood(tau, eta_idx, nu_idx, xi_idx, sigma_idx)
                            
                            omega_prob = (
                                self.config.prob_eta_support_points[eta_idx] *
                                self.config.prob_nu_support_points[nu_idx] *
                                self.config.prob_xi_support_points[xi_idx] *
                                self.config.prob_sigmavarepsilon_support_points[sigma_idx]
                            )
                            
                            tau_lik = tau_lik + omega_prob * omega_lik
            
            total_lik = total_lik + self.params.pi_tau[tau] * tau_lik
        
        return total_lik