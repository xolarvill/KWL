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
    n_obs: int,
    migration_rate: float = None
) -> Dict[str, float]:
    """
    全面评估Type分离质量，适用于低迁移率数据

    Args:
        posterior_probs: 后验概率矩阵
        log_likelihood: 对数似然值
        n_params: 参数数量
        n_obs: 观测数量
        migration_rate: 迁移率，如果提供则使用适应性评估

    Returns:
        评估指标字典
    """
    type_probs = posterior_probs.mean(axis=0)

    # 根据迁移率调整评估策略
    if migration_rate is not None and migration_rate < 0.1:  # 低迁移率场景
        # 使用更灵活的平衡指标
        # 不强制要求完全均衡，而是评估类型的可区分性
        entropy_score = entropy(type_probs, base=np.e) / np.log(len(type_probs)) if len(type_probs) > 1 else 0
        
        # 改进的平衡评分，适应低迁移率
        # 重点看是否有类型完全消失，而不是强制均衡
        min_type_prob = np.min(type_probs)
        max_type_prob = np.max(type_probs)
        
        # 避免任何类型完全消失（>5%阈值）
        survival_score = 1.0 if min_type_prob > 0.05 else min_type_prob / 0.05
        
        # 计算类型区分度
        separation_score = 1 - (max_type_prob - min_type_prob)  # 区分度越高，差异越大
        
        # 综合平衡评分
        balance_score = 0.5 * survival_score + 0.3 * entropy_score + 0.2 * separation_score
    else:  # 标准场景
        balance_score = calculate_type_balance_score(posterior_probs)

    metrics = {
        # Type分布指标
        'balance_score': balance_score,
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


def evaluate_type_separation_for_low_migration_rate(
    posterior_probs: np.ndarray,
    log_likelihood: float,
    n_params: int,
    n_obs: int,
    migration_rate: float = 0.05
) -> Dict[str, float]:
    """
    为低迁移率数据定制的类型分离评估指标
    """
    # 计算每种类型的平均概率
    type_probs = posterior_probs.mean(axis=0)
    n_types = len(type_probs)
    
    # 根据迁移率调整评估策略
    if migration_rate < 0.1:  # 低迁移率场景
        # 使用更灵活的平衡指标
        # 不强制要求完全均衡，而是评估类型的可区分性
        entropy_score = entropy(type_probs, base=np.e) / np.log(n_types) if n_types > 1 else 0
        
        # 改进的平衡评分，适应低迁移率
        # 重点看是否有类型完全消失，而不是强制均衡
        min_type_prob = np.min(type_probs)
        max_type_prob = np.max(type_probs)
        
        # 避免任何类型完全消失（>5%阈值）
        survival_score = 1.0 if min_type_prob > 0.05 else min_type_prob / 0.05
        
        # 计算类型区分度
        separation_score = 1 - (max_type_prob - min_type_prob)  # 区分度越高，差异越大
        
        # 综合平衡评分
        balance_score = 0.5 * survival_score + 0.3 * entropy_score + 0.2 * separation_score
    
    else:  # 高迁移率场景，使用标准指标
        balance_score = calculate_type_balance_score(posterior_probs)
    
    # 信息准则（不变）
    bic = calculate_bic(log_likelihood, n_params, n_obs)
    aic = calculate_aic(log_likelihood, n_params)
    
    metrics = {
        'balance_score': balance_score,
        'entropy': entropy(type_probs, base=np.e),
        'min_type_prob': np.min(type_probs),
        'max_type_prob': np.max(type_probs),
        'type_prob_std': np.std(type_probs),
        'bic': bic,
        'aic': aic,
        'log_likelihood': log_likelihood
    }
    
    # 按类型添加概率
    for i, prob in enumerate(type_probs):
        metrics[f'type_{i}_prob'] = prob
    
    return metrics


def check_degeneracy_for_low_migration_data(
    posterior_probs: np.ndarray,
    migration_rate: float = 0.05,
    dominance_threshold: float = None,
    vanishing_threshold: float = None
) -> Tuple[bool, str]:
    """
    为低迁移率数据定制的退化检查
    """
    type_probs = posterior_probs.mean(axis=0)
    max_prob = np.max(type_probs)
    max_type = np.argmax(type_probs)
    
    if dominance_threshold is None:
        # 对于低迁移率数据，使用更宽松的阈值
        dominance_threshold = 0.9 if migration_rate > 0.1 else 0.95
    
    if vanishing_threshold is None:
        # 对于低迁移率数据，使用更宽松的阈值
        vanishing_threshold = 0.02 if migration_rate > 0.1 else 0.05
    
    if max_prob > dominance_threshold:
        return True, f"Type {max_type} dominates with {max_prob:.1%} probability (low migration adaptation)"
    
    min_prob = np.min(type_probs)
    min_type = np.argmin(type_probs)
    
    if min_prob < vanishing_threshold:
        return True, f"Type {min_type} nearly vanished with {min_prob:.1%} probability (low migration adaptation)"
    
    return False, f"Type distribution is balanced (migration rate: {migration_rate:.2%})"


def adaptive_type_identification_score(posterior_probs, migration_rate):
    """
    适应迁移率的类型识别评分
    """
    type_probs = posterior_probs.mean(axis=0)
    
    if migration_rate < 0.1:  # 低迁移率情况
        # 评估类型是否能够区分不同的迁移模式
        # 即使类型分布不均衡，只要能识别不同行为模式就认为成功
        
        # 基于后验概率的类型确定性
        certainty = np.mean(np.max(posterior_probs, axis=1))
        
        # 类型生存率（没有类型完全消失）
        n_types = len(type_probs)
        survival_rate = np.sum(type_probs > 0.05) / n_types
        
        # 综合评分
        adaptive_score = 0.6 * certainty + 0.4 * survival_rate
        
    else:  # 高迁移率情况
        # 使用标准的均衡评分
        adaptive_score = calculate_type_balance_score(posterior_probs)
    
    return adaptive_score


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
