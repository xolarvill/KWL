"""
参数搜索工具模块：用于评估和优化Type特定参数

主要功能：
1. Type均衡度评估
2. BIC计算
3. 参数组合快速评估
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from scipy.stats import entropy


def calculate_type_balance_score(posterior_probs: np.ndarray) -> float:
    """
    计算Type分布的均衡度分数

    Args:
        posterior_probs: 后验概率矩阵 (N x K)

    Returns:
        float: 均衡度分数，0-1之间，越接近1越均衡
    """
    # 计算每个type的平均概率
    type_probs = posterior_probs.mean(axis=0)

    # 方法1: 使用熵度量（归一化）
    # 完全均衡时，熵最大 = log(K)
    K = len(type_probs)
    max_entropy = np.log(K)
    current_entropy = entropy(type_probs, base=np.e)
    entropy_score = current_entropy / max_entropy if max_entropy > 0 else 0

    # 方法2: 使用最小type占比（防止退化）
    min_type_prob = np.min(type_probs)
    min_acceptable = 0.1  # 每个type至少占10%
    min_prob_score = min(min_type_prob / min_acceptable, 1.0)

    # 方法3: 使用方差（低方差表示均衡）
    ideal_prob = 1.0 / K
    variance = np.var(type_probs)
    max_variance = ideal_prob * (1 - ideal_prob)  # 最大可能方差（二项分布）
    variance_score = 1 - (variance / max_variance) if max_variance > 0 else 0

    # 综合分数（加权平均）
    balance_score = 0.4 * entropy_score + 0.4 * min_prob_score + 0.2 * variance_score

    return balance_score


def calculate_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    计算贝叶斯信息准则(BIC)

    Args:
        log_likelihood: 对数似然值
        n_params: 参数数量
        n_obs: 观测数量

    Returns:
        float: BIC值，越小越好
    """
    return -2 * log_likelihood + n_params * np.log(n_obs)


def calculate_aic(log_likelihood: float, n_params: int) -> float:
    """
    计算赤池信息准则(AIC)

    Args:
        log_likelihood: 对数似然值
        n_params: 参数数量

    Returns:
        float: AIC值，越小越好
    """
    return -2 * log_likelihood + 2 * n_params


def evaluate_type_separation(
    posterior_probs: np.ndarray,
    log_likelihood: float,
    n_params: int,
    n_obs: int
) -> Dict[str, float]:
    """
    全面评估Type分离质量

    Args:
        posterior_probs: 后验概率矩阵
        log_likelihood: 对数似然值
        n_params: 参数数量
        n_obs: 观测数量

    Returns:
        评估指标字典
    """
    type_probs = posterior_probs.mean(axis=0)

    metrics = {
        # Type分布指标
        'balance_score': calculate_type_balance_score(posterior_probs),
        'entropy': entropy(type_probs, base=np.e),
        'min_type_prob': np.min(type_probs),
        'max_type_prob': np.max(type_probs),
        'type_prob_std': np.std(type_probs),

        # 信息准则
        'bic': calculate_bic(log_likelihood, n_params, n_obs),
        'aic': calculate_aic(log_likelihood, n_params),
        'log_likelihood': log_likelihood,

        # Type占比
        'type_0_prob': type_probs[0] if len(type_probs) > 0 else 0,
        'type_1_prob': type_probs[1] if len(type_probs) > 1 else 0,
        'type_2_prob': type_probs[2] if len(type_probs) > 2 else 0,
    }

    return metrics


def check_degeneracy(posterior_probs: np.ndarray, threshold: float = 0.95) -> Tuple[bool, str]:
    """
    检查是否存在type退化

    Args:
        posterior_probs: 后验概率矩阵
        threshold: 退化阈值（某个type占比超过此值视为退化）

    Returns:
        (is_degenerate, message)
    """
    type_probs = posterior_probs.mean(axis=0)
    max_prob = np.max(type_probs)
    max_type = np.argmax(type_probs)

    if max_prob > threshold:
        return True, f"Type {max_type} dominates with {max_prob:.1%} probability"

    min_prob = np.min(type_probs)
    min_type = np.argmin(type_probs)

    if min_prob < 0.05:
        return True, f"Type {min_type} nearly vanished with {min_prob:.1%} probability"

    return False, "Type distribution is balanced"


def suggest_gamma_adjustment(
    type_probs: np.ndarray,
    current_gamma_0: List[float]
) -> Dict[str, Any]:
    """
    基于当前type分布，建议gamma_0调整方向

    Args:
        type_probs: 各type的概率
        current_gamma_0: 当前的gamma_0值列表

    Returns:
        调整建议字典
    """
    suggestions = []

    for i, (prob, gamma) in enumerate(zip(type_probs, current_gamma_0)):
        if prob > 0.6:
            # Type占比过高，增加其迁移成本使其less attractive
            suggestions.append({
                'type': i,
                'current_gamma': gamma,
                'suggested_gamma': gamma * 1.5,
                'reason': f'Type {i} too dominant ({prob:.1%}), increase cost'
            })
        elif prob < 0.1:
            # Type占比过低，降低其迁移成本使其more attractive
            suggestions.append({
                'type': i,
                'current_gamma': gamma,
                'suggested_gamma': gamma * 0.5,
                'reason': f'Type {i} too rare ({prob:.1%}), decrease cost'
            })
        else:
            suggestions.append({
                'type': i,
                'current_gamma': gamma,
                'suggested_gamma': gamma,
                'reason': f'Type {i} balanced ({prob:.1%}), keep current'
            })

    return {
        'suggestions': suggestions,
        'overall_balance': calculate_type_balance_score(
            np.array([type_probs])  # Convert to 2D for the function
        )
    }
