"""
工资方程模块

实现论文中的简化工资方程(删除前景理论):
    ln w_itj = μ_jt + G(X_i, a_it) + η_i + ν_ij + ε_itj
    
收入效用: U_income = α_w · ln w_itj (简单对数效用)

其中:
- μ_j(K_t): 地区基础工资（从预估计文件读取）
- G(·): ML插件预测的人力资本积累部分
- η_i: 个体固定效应（离散支撑点）
- ν_ij: 个体-地区匹配效应（离散支撑点）
- ε_itj: 暂态随机冲击（用于工资似然计算）
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy.stats import norm
import logging


def calculate_predicted_wage(
    individual_data: pd.DataFrame,
    region_data: pd.DataFrame,
    params: Dict[str, Any],
    eta_i: np.ndarray,
    nu_ij: np.ndarray,
    ml_plugin_predictions: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    计算工资预测值 w_predicted

    根据论文公式(718-720行):
        w_itj = μ_j(K_t) + G(X_i, a_it, t) + η_i + ν_ij

    参数:
    ----
    individual_data : pd.DataFrame
        个体数据，必须包含列: provcd, year
    region_data : pd.DataFrame
        地区数据，必须包含列: provcd, year, 地区基本经济面(μ_jt)
    params : Dict[str, Any]
        模型参数字典
    eta_i : np.ndarray
        个体固定效应值 (shape: n_obs,)
    nu_ij : np.ndarray
        个体-地区匹配效应值 (shape: n_obs,)
    ml_plugin_predictions : np.ndarray, optional
        ML插件预测的G(·)部分 (shape: n_obs,)
        如果为None，则使用简化的年龄二次函数

    返回:
    ----
    np.ndarray
        预测工资 (shape: n_obs,)
    """
    logger = logging.getLogger()

    # 1. 获取地区基础工资 μ_j(K_t)
    # 合并individual_data和region_data以获取每个观测对应的μ_jt
    merged = individual_data.merge(
        region_data[['provcd', 'year', '地区基本经济面']],
        on=['provcd', 'year'],
        how='left'
    )

    mu_jt = merged['地区基本经济面'].values

    # 处理缺失值（如果有）
    if np.any(np.isnan(mu_jt)):
        logger.warning(f"发现{np.sum(np.isnan(mu_jt))}个缺失的μ_jt值，用0填充")
        mu_jt = np.nan_to_num(mu_jt, nan=0.0)

    # 2. 获取人力资本积累部分 G(X_i, a_it, t)
    if ml_plugin_predictions is not None:
        G_xit = ml_plugin_predictions
    else:
        # 简化版本：使用年龄的二次函数作为fallback
        age = individual_data['age'].values
        G_xit = (
            params.get('r1', 0.05) * age
            + params.get('r2', -0.001) * (age ** 2)
        )

    # 3. 组合所有组件
    w_predicted = mu_jt + G_xit + eta_i + nu_ij

    return w_predicted


def calculate_wage_likelihood(
    w_observed: np.ndarray,
    w_predicted: np.ndarray,
    sigma_epsilon: float
) -> np.ndarray:
    """
    计算工资观测值的对数似然密度

    根据论文公式(746-750行):
        Ψ_itj = (1/σ_ε) φ((w_obs - w_pred) / σ_ε)

    参数:
    ----
    w_observed : np.ndarray
        观测到的工资值 (shape: n_obs,)
    w_predicted : np.ndarray
        预测的工资值 (shape: n_obs,)
    sigma_epsilon : float
        暂态冲击的标准差

    返回:
    ----
    np.ndarray
        每个观测的对数似然密度 (shape: n_obs,)
    """
    # 计算残差
    residuals = w_observed - w_predicted

    # 计算标准化残差
    standardized_residuals = residuals / np.maximum(sigma_epsilon, 1e-6)

    # 计算对数似然: log(pdf) = log(1/σ) + log(φ(z))
    # 其中 log(φ(z)) = -0.5 * z^2 - 0.5 * log(2π)
    log_likelihood = (
        -np.log(np.maximum(sigma_epsilon, 1e-6))  # -log(σ)
        - 0.5 * (standardized_residuals ** 2)      # -0.5 * z^2
        - 0.5 * np.log(2 * np.pi)                  # -0.5 * log(2π)
    )

    # 处理异常值
    log_likelihood = np.nan_to_num(log_likelihood, nan=-1e10, posinf=-1e10, neginf=-1e10)

    return log_likelihood


# 前景理论相关函数已删除 - 改用简单对数收入效用
# 保留空函数以保持API兼容性，但实际不再使用前景理论
def calculate_reference_wage(
    individual_data: pd.DataFrame,
    current_wages: np.ndarray,
    reference_type: str = 'lagged'
) -> np.ndarray:
    """
    参照工资计算函数 (已废弃 - 前景理论已删除)
    
    为保持API兼容性而保留，返回当前工资
    """
    return current_wages


def calculate_prospect_theory_utility(
    w_current: np.ndarray,
    w_reference: np.ndarray,
    alpha_w: float,
    lambda_loss_aversion: float,
    use_log_difference: bool = True
) -> np.ndarray:
    """
    前景理论效用函数 (已废弃 - 改用简单对数效用)
    
    为保持API兼容性而保留，实际返回简单对数效用
    """
    # 简单对数效用：α_w · ln(w_current)
    w_current_safe = np.maximum(w_current, 1e-6)
    return alpha_w * np.log(w_current_safe)


def calculate_nu_variance_with_internet(
    region_data: pd.DataFrame,
    delta_0: float,
    delta_1: float,
    internet_column: str = '移动电话普及率'
) -> pd.DataFrame:
    """
    计算带互联网调节的个体-地区匹配效应方差

    根据论文公式(736-742行):
        σ²_ν,jt = exp(δ_0 - δ_1 · Internet_jt)

    参数:
    ----
    region_data : pd.DataFrame
        地区数据，必须包含列: provcd, year, {internet_column}
    delta_0 : float
        方差基准参数
    delta_1 : float
        互联网效应参数 (预期 δ_1 > 0，即互联网降低不确定性)
    internet_column : str
        互联网普及率的列名

    返回:
    ----
    pd.DataFrame
        包含新列 'nu_variance' 的地区数据
    """
    result = region_data.copy()

    # 提取互联网普及率
    internet_rate = result[internet_column].values

    # 处理缺失值
    internet_rate = np.nan_to_num(internet_rate, nan=0.0)

    # 计算方差: σ²_ν,jt = exp(δ_0 - δ_1 · Internet_jt)
    nu_variance = np.exp(delta_0 - delta_1 * internet_rate)

    result['nu_variance'] = nu_variance
    result['nu_std'] = np.sqrt(nu_variance)

    return result
