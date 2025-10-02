"""
该模块实现参数估计的标准误计算和统计推断
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Any, Tuple
import warnings

def compute_hessian_numerical(
    log_likelihood_func,
    params: Dict[str, float],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    agent_type: int,
    beta: float,
    h: float = 1e-4
) -> np.ndarray:
    """
    使用数值方法计算对数似然函数的海塞矩阵
    
    Args:
        log_likelihood_func: 对数似然函数
        params: 参数字典
        observed_data: 观测数据
        state_space: 状态空间
        transition_matrices: 转移矩阵
        agent_type: 代理人类型
        beta: 折现因子
        h: 数值微分步长
    
    Returns:
        np.ndarray: 海塞矩阵
    """
    # 提取参数值并记录参数名称
    param_names = sorted(params.keys())
    param_values = np.array([params[name] for name in param_names])
    
    n_params = len(param_values)
    hessian = np.zeros((n_params, n_params))
    
    # 计算梯度函数（一阶导数）
    def gradient(theta):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            # 创建正向和负向扰动的参数字典
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += h
            theta_minus[i] -= h
            
            # 重构参数字典
            params_plus = {name: val for name, val in zip(param_names, theta_plus)}
            params_minus = {name: val for name, val in zip(param_names, theta_minus)}
            
            # 计算数值导数
            ll_plus = log_likelihood_func(
                params=params_plus,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=agent_type,
                beta=beta,
                transition_matrices=transition_matrices
            )
            ll_minus = log_likelihood_func(
                params=params_minus,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=agent_type,
                beta=beta,
                transition_matrices=transition_matrices
            )
            
            grad[i] = (ll_plus - ll_minus) / (2 * h)
        return grad
    
    # 计算海塞矩阵（二阶导数）
    base_grad = gradient(param_values)
    for i in range(n_params):
        for j in range(i, n_params):  # 只计算上三角矩阵
            if i == j:
                # 对角元素
                theta_plus = param_values.copy()
                theta_minus = param_values.copy()
                theta_plus[i] += h
                theta_minus[i] -= h
                
                grad_plus = gradient(theta_plus)[i]
                grad_minus = gradient(theta_minus)[i]
                
                hessian[i, j] = (grad_plus - grad_minus) / (2 * h)
            else:
                # 非对角元素
                theta_plus = param_values.copy()
                theta_minus = param_values.copy()
                theta_plus[i] += h
                theta_minus[i] -= h
                
                grad_plus = gradient(theta_plus)[j]
                grad_minus = gradient(theta_minus)[j]
                
                hessian[i, j] = (grad_plus - grad_minus) / (2 * h)
                hessian[j, i] = hessian[i, j]  # 海塞矩阵是对称的
    
    return hessian


def estimate_standard_errors(
    log_likelihood_func,
    params: Dict[str, float],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    agent_type: int,
    beta: float
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    估计参数的标准误、t统计量和p值
    
    Args:
        log_likelihood_func: 对数似然函数
        params: 估计得到的参数
        observed_data: 观测数据
        state_space: 状态空间
        transition_matrices: 转移矩阵
        agent_type: 代理人类型
        beta: 折现因子
    
    Returns:
        Tuple[Dict, Dict, Dict]: (标准误, t统计量, p值)
    """
    warnings.warn("使用数值方法计算海塞矩阵可能计算量较大，建议在参数收敛后再运行")
    
    # 计算海塞矩阵
    hessian = compute_hessian_numerical(
        log_likelihood_func, params, observed_data, state_space, 
        transition_matrices, agent_type, beta
    )
    
    # 信息矩阵是负海塞矩阵（负的二阶导数矩阵）
    info_matrix = -hessian
    
    # 计算协方差矩阵（信息矩阵的逆）
    try:
        cov_matrix = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        cov_matrix = np.linalg.pinv(info_matrix)
        warnings.warn("使用伪逆计算协方差矩阵，可能存在多重共线性")
    
    # 提取参数名称
    param_names = sorted(params.keys())
    
    # 计算标准误（协方差矩阵对角线的平方根）
    std_errors = {}
    for i, name in enumerate(param_names):
        std_errors[name] = np.sqrt(max(0, cov_matrix[i, i])) if i < cov_matrix.shape[0] and i < cov_matrix.shape[1] else 0.0
    
    # 计算t统计量
    t_stats = {}
    for name in param_names:
        if std_errors[name] > 0:
            t_stats[name] = params[name] / std_errors[name]
        else:
            t_stats[name] = np.inf if params[name] > 0 else -np.inf
    
    # 计算p值（双侧检验）
    p_values = {}
    for name in param_names:
        # 使用标准正态分布近似（大样本）
        p_values[name] = 2 * (1 - norm.cdf(abs(t_stats[name])))
    
    return std_errors, t_stats, p_values


def compute_information_criteria(
    log_likelihood: float,
    n_params: int,
    n_observations: int
) -> Dict[str, float]:
    """
    计算信息准则（AIC, BIC）
    
    Args:
        log_likelihood: 对数似然值
        n_params: 参数数量
        n_observations: 观测值数量
    
    Returns:
        Dict[str, float]: 信息准则
    """
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_observations)
    
    return {"AIC": aic, "BIC": bic}