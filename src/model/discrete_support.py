"""
离散支撑点模块

实现论文中的离散因子近似方法(984-1016行):
- 为未观测异质性 ω_i = {η_i, {ν_ij}, {ξ_ij}, σ_ε,i} 生成离散支撑点
- 枚举所有可能的支撑点组合
- 计算每个组合的先验概率
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
import logging


class DiscreteSupportGenerator:
    """
    离散支撑点生成器

    用于生成和管理未观测异质性变量的离散支撑点
    """

    def __init__(
        self,
        n_eta_support: int = 7,
        n_nu_support: int = 5,
        n_xi_support: int = 5,
        n_sigma_support: int = 4,
        eta_range: Tuple[float, float] = (-2.0, 2.0),
        nu_range: Tuple[float, float] = (-1.5, 1.5),
        xi_range: Tuple[float, float] = (-1.0, 1.0),
        sigma_range: Tuple[float, float] = (0.3, 1.5)
    ):
        """
        初始化支撑点生成器

        参数:
        ----
        n_eta_support : int
            个体固定效应η_i的支撑点数量
        n_nu_support : int
            个体-地区收入匹配ν_ij的支撑点数量（每个地区）
        n_xi_support : int
            个体-地区偏好匹配ξ_ij的支撑点数量（每个地区）
        n_sigma_support : int
            工资波动性σ_ε的支撑点数量
        eta_range, nu_range, xi_range, sigma_range : Tuple[float, float]
            各变量的取值范围
        """
        self.n_eta = n_eta_support
        self.n_nu = n_nu_support
        self.n_xi = n_xi_support
        self.n_sigma = n_sigma_support

        # 生成对称的支撑点（包含0）
        self.eta_support = self._generate_symmetric_support(eta_range, n_eta_support)
        self.nu_support = self._generate_symmetric_support(nu_range, n_nu_support)
        self.xi_support = self._generate_symmetric_support(xi_range, n_xi_support)

        # σ_ε必须为正，使用对数均匀分布
        self.sigma_support = np.exp(np.linspace(
            np.log(sigma_range[0]),
            np.log(sigma_range[1]),
            n_sigma_support
        ))

        # 均匀先验概率
        self.eta_prob = np.ones(n_eta_support) / n_eta_support
        self.nu_prob = np.ones(n_nu_support) / n_nu_support
        self.xi_prob = np.ones(n_xi_support) / n_xi_support
        self.sigma_prob = np.ones(n_sigma_support) / n_sigma_support

    def _generate_symmetric_support(
        self,
        value_range: Tuple[float, float],
        n_points: int
    ) -> np.ndarray:
        """
        生成对称的支撑点（包含0）

        例如: n=7 → [-2, -1.33, -0.67, 0, 0.67, 1.33, 2]
        """
        if n_points % 2 == 1:
            # 奇数：包含0作为中心点
            half_n = n_points // 2
            positive_points = np.linspace(0, value_range[1], half_n + 1)
            negative_points = -positive_points[1:][::-1]
            support = np.concatenate([negative_points, positive_points])
        else:
            # 偶数：不包含0
            support = np.linspace(value_range[0], value_range[1], n_points)

        return support

    def get_support_info(self) -> Dict[str, Any]:
        """
        获取支撑点信息摘要
        """
        return {
            'eta': {
                'n_points': self.n_eta,
                'values': self.eta_support,
                'probs': self.eta_prob
            },
            'nu': {
                'n_points': self.n_nu,
                'values': self.nu_support,
                'probs': self.nu_prob
            },
            'xi': {
                'n_points': self.n_xi,
                'values': self.xi_support,
                'probs': self.xi_prob
            },
            'sigma_epsilon': {
                'n_points': self.n_sigma,
                'values': self.sigma_support,
                'probs': self.sigma_prob
            }
        }

    def get_region_specific_nu_support(
        self,
        region_data: pd.DataFrame,
        delta_0: float,
        delta_1: float,
        internet_column: str = '移动电话普及率'
    ) -> Dict[str, np.ndarray]:
        """
        生成地区特定的ν支撑点（带互联网调节）

        根据论文公式(736-742行):
            σ²_ν,jt = exp(δ_0 - δ_1 · Internet_jt)
            ν_ij ~ N(0, σ²_ν,jt)

        参数:
        ----
        region_data : pd.DataFrame
            地区数据，必须包含列: provcd, {internet_column}
        delta_0 : float
            方差基准参数
        delta_1 : float
            互联网效应参数 (δ_1 > 0 表示互联网降低不确定性)
        internet_column : str
            互联网普及率的列名

        返回:
        ----
        Dict[str, np.ndarray]
            键为地区代码(provcd)，值为该地区的ν支撑点数组
        """
        logger = logging.getLogger()

        # 检查互联网列是否存在
        if internet_column not in region_data.columns:
            logger.warning(f"地区数据中未找到'{internet_column}'列，使用默认nu_support")
            # 返回所有地区使用相同的支撑点
            unique_provcds = region_data['provcd'].unique()
            return {provcd: self.nu_support for provcd in unique_provcds}

        # 计算每个地区的方差
        region_variance = {}
        for provcd in region_data['provcd'].unique():
            region_subset = region_data[region_data['provcd'] == provcd]

            # 使用该地区的平均互联网普及率（如果有多年数据）
            internet_rate = region_subset[internet_column].mean()

            # 处理缺失值
            if np.isnan(internet_rate):
                logger.warning(f"地区{provcd}的互联网数据缺失，使用默认方差")
                variance = np.exp(delta_0)  # 默认方差（互联网=0时）
            else:
                # 计算该地区的σ²_ν,jt
                variance = np.exp(delta_0 - delta_1 * internet_rate)

            region_variance[provcd] = variance

        # 为每个地区生成调节后的支撑点
        region_specific_supports = {}
        for provcd, variance in region_variance.items():
            # 缩放支撑点：原始支撑点假设σ²=1，现在根据实际方差缩放
            sigma_jt = np.sqrt(variance)
            scaled_support = self.nu_support * sigma_jt
            region_specific_supports[provcd] = scaled_support

        logger.info(f"生成了{len(region_specific_supports)}个地区的特定ν支撑点")
        logger.debug(f"方差范围: [{min(region_variance.values()):.4f}, {max(region_variance.values()):.4f}]")

        return region_specific_supports


class SimplifiedOmegaEnumerator:
    """
    简化的ω枚举器

    关键简化策略（解决维度灾难）：
    1. ν_ij 和 ξ_ij 只在**个体访问过的地区**实例化
    2. 未访问地区在首次访问时从分布中抽取
    3. 使用个体的历史轨迹信息来减少枚举空间
    """

    def __init__(self, support_generator: DiscreteSupportGenerator):
        """
        初始化枚举器

        参数:
        ----
        support_generator : DiscreteSupportGenerator
            支撑点生成器实例
        """
        self.gen = support_generator
        self.logger = logging.getLogger()

    def enumerate_omega_for_individual(
        self,
        individual_data: pd.DataFrame,
        max_combinations: int = 10000
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        为单个个体枚举ω组合

        参数:
        ----
        individual_data : pd.DataFrame
            单个个体的完整历史数据（多行，每行一个时期）
            必须包含列: provcd, year
        max_combinations : int
            最大组合数（超过则使用蒙特卡洛采样）

        返回:
        ----
        omega_combinations : List[Dict]
            ω组合列表，每个dict包含: {'eta', 'nu_dict', 'xi_dict', 'sigma'}
        omega_probs : np.ndarray
            每个组合的先验概率
        """
        # 1. 识别该个体访问过的地区
        visited_regions = individual_data['provcd_t'].unique()
        n_visited = len(visited_regions)

        # 2. 计算总组合数
        total_combinations = (
            self.gen.n_eta *           # η_i
            (self.gen.n_nu ** n_visited) *  # ν_ij for each visited region
            (self.gen.n_xi ** n_visited) *  # ξ_ij for each visited region
            self.gen.n_sigma           # σ_ε
        )

        self.logger.debug(f"Individual visited {n_visited} regions, "
                         f"total combinations: {total_combinations}")

        # 3. 决定使用完全枚举还是蒙特卡洛采样
        if total_combinations <= max_combinations:
            return self._enumerate_exact(visited_regions)
        else:
            # 【优化】减少刷屏：只在首次或每100次个体时输出警告
            if not hasattr(self, '_mc_warning_count'):
                self._mc_warning_count = 0
            self._mc_warning_count += 1
            
            if self._mc_warning_count <= 5 or self._mc_warning_count % 100 == 0:
                self.logger.warning(f"Too many combinations ({total_combinations}), "
                                   f"using Monte Carlo sampling with {max_combinations} draws "
                                   f"(个体 #{self._mc_warning_count})")
            else:
                # 其余情况只记录debug日志
                self.logger.debug(f"Monte Carlo sampling for individual #{self._mc_warning_count}")
            
            return self._enumerate_monte_carlo(visited_regions, max_combinations)

    def _enumerate_exact(
        self,
        visited_regions: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        完全枚举所有ω组合
        """
        omega_list = []
        prob_list = []

        # 为每个访问过的地区生成nu和xi的支撑点
        nu_combinations = list(product(self.gen.nu_support, repeat=len(visited_regions)))
        xi_combinations = list(product(self.gen.xi_support, repeat=len(visited_regions)))
        
        # 预计算概率值以提高效率
        eta_probs = self.gen.eta_prob
        nu_prob = (self.gen.nu_prob[0] ** len(visited_regions))  # 简化：均匀概率
        xi_prob = (self.gen.xi_prob[0] ** len(visited_regions))
        sigma_probs = self.gen.sigma_prob
        
        # 枚举所有组合
        for eta_idx, (eta_val, eta_prob) in enumerate(zip(self.gen.eta_support, eta_probs)):
            for nu_tuple in nu_combinations:
                for xi_tuple in xi_combinations:
                    for sigma_idx, (sigma_val, sigma_prob) in enumerate(zip(self.gen.sigma_support, sigma_probs)):
                        # 构建ω字典
                        omega = {
                            'eta': eta_val,
                            'nu_dict': dict(zip(visited_regions, nu_tuple)),
                            'xi_dict': dict(zip(visited_regions, xi_tuple)),
                            'sigma': sigma_val
                        }
                        omega_list.append(omega)

                        # 计算联合先验概率（假设独立）
                        joint_prob = eta_prob * nu_prob * xi_prob * sigma_prob
                        prob_list.append(joint_prob)

        # 归一化概率
        prob_array = np.array(prob_list)
        prob_array = prob_array / np.sum(prob_array)

        return omega_list, prob_array

    def _enumerate_monte_carlo(
        self,
        visited_regions: np.ndarray,
        n_samples: int
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        使用蒙特卡洛采样近似枚举
        """
        omega_list = []

        for _ in range(n_samples):
            # 从各分布中抽样
            eta_val = np.random.choice(self.gen.eta_support, p=self.gen.eta_prob)
            nu_vals = np.random.choice(self.gen.nu_support, size=len(visited_regions), p=self.gen.nu_prob)
            xi_vals = np.random.choice(self.gen.xi_support, size=len(visited_regions), p=self.gen.xi_prob)
            sigma_val = np.random.choice(self.gen.sigma_support, p=self.gen.sigma_prob)

            omega = {
                'eta': eta_val,
                'nu_dict': dict(zip(visited_regions, nu_vals)),
                'xi_dict': dict(zip(visited_regions, xi_vals)),
                'sigma': sigma_val
            }
            omega_list.append(omega)

        # 蒙特卡洛：等权重
        prob_array = np.ones(n_samples) / n_samples

        return omega_list, prob_array


def extract_omega_values_for_state(
    omega: Dict[str, Any],
    state_data: pd.DataFrame,
    region_col: str = 'provcd'
) -> Dict[str, np.ndarray]:
    """
    从ω字典中提取状态空间对应的值

    参数:
    ----
    omega : Dict
        单个ω组合，包含 {'eta', 'nu_dict', 'xi_dict', 'sigma'}
    state_data : pd.DataFrame
        状态空间数据，包含多个状态
    region_col : str
        地区列名

    返回:
    ----
    Dict with keys:
        'eta_i': np.ndarray (n_states,) - 广播到所有状态
        'nu_ij': np.ndarray (n_states, n_choices) - 根据访问历史填充
        'xi_ij': np.ndarray (n_states, n_choices) - 根据访问历史填充
        'sigma_epsilon': float - 标量
    """
    n_states = len(state_data)

    # η_i: 对所有状态相同
    eta_array = np.full(n_states, omega['eta'])

    # σ_ε: 标量
    sigma_epsilon = omega['sigma']

    # TODO: nu_ij 和 xi_ij 需要根据状态空间和访问历史构建
    # 这里先返回None，实际使用时需要在上层逻辑中处理

    return {
        'eta_i': eta_array,
        'sigma_epsilon': sigma_epsilon,
        'nu_dict': omega['nu_dict'],
        'xi_dict': omega['xi_dict']
    }


# =====================================
# 辅助函数：用于EM算法集成
# =====================================

def enumerate_omega_for_all_individuals(
    observed_data: pd.DataFrame,
    support_generator: DiscreteSupportGenerator,
    individual_id_col: str = 'IID',
    max_combinations_per_individual: int = 5000
) -> Dict[Any, Tuple[List[Dict], np.ndarray]]:
    """
    为所有个体生成ω枚举

    参数:
    ----
    observed_data : pd.DataFrame
        完整的观测数据
    support_generator : DiscreteSupportGenerator
        支撑点生成器
    individual_id_col : str
        个体ID列名
    max_combinations_per_individual : int
        每个个体的最大组合数

    返回:
    ----
    Dict[individual_id, (omega_list, omega_probs)]
        键为个体ID，值为该个体的ω组合列表和概率
    """
    logger = logging.getLogger()
    enumerator = SimplifiedOmegaEnumerator(support_generator)

    omega_dict = {}
    unique_individuals = observed_data[individual_id_col].unique()

    logger.info(f"Enumerating omega for {len(unique_individuals)} individuals...")

    for i, individual_id in enumerate(unique_individuals):
        if i % 100 == 0:
            logger.info(f"  Processing individual {i+1}/{len(unique_individuals)}")

        # 提取该个体的所有观测
        individual_data = observed_data[observed_data[individual_id_col] == individual_id]

        # 枚举ω
        omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=max_combinations_per_individual
        )

        omega_dict[individual_id] = (omega_list, omega_probs)

    logger.info(f"Omega enumeration completed for all individuals.")

    return omega_dict
