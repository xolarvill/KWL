"""
E-step个体处理模块 - 支持并行化
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, List, Optional

def process_single_individual_e_step(
    individual_id: Any,
    individual_data: Any,
    omega_list: List[Dict],
    omega_probs: np.ndarray,
    params: Dict[str, Any],
    pi_k: np.ndarray,
    K: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    prov_to_idx: Dict[int, int],
    bellman_cache: Any,
    cache_stats: Dict[str, int]
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    处理单个个体的E-step计算
    
    返回:
        individual_id: 个体ID
        joint_probs: 联合后验概率矩阵 (n_omega, K)
        marginal_likelihood: 边缘化似然 (K,)
    """
    logger = logging.getLogger()
    
    if individual_data.empty:
        logger.warning(f"  Skipping individual {individual_id} as their data is empty.")
        # 返回空结果，将在主函数中处理
        return individual_id, np.array([]), np.full(K, -1e10)
    
    n_omega = len(omega_list)
    
    # 计算个体状态数量（用于热启动验证）
    n_individual_states = len(individual_data['visited_locations'].iloc[0]) * len(individual_data['age_t'].unique())
    
    # 3. 计算每个(τ, ω)组合的似然
    # 结构: posterior_matrix[omega_idx, type_idx]
    log_lik_matrix = np.zeros((n_omega, K))
    
    individual_cache_hits = 0
    individual_cache_misses = 0
    
    for omega_idx, omega in enumerate(omega_list):
        # 提取ω值
        eta_i = omega['eta']
        sigma_epsilon = omega['sigma']
        
        for k in range(K):
            try:
                # 构建type-specific参数
                # 注意：现在只有gamma_0是type-specific，其他参数已经是共享的
                type_params = params.copy()
                if f'gamma_0_type_{k}' in params:
                    type_params['gamma_0'] = params[f'gamma_0_type_{k}']
                
                type_params['sigma_epsilon'] = sigma_epsilon
                
                # 检查缓存 - 使用增强版缓存系统（智能查询）
                converged_v_individual = None
                initial_v = None
                
                if bellman_cache is None:
                    # 缓存为None，直接求解Bellman方程
                    logger.debug(f"E-step缓存为None，直接求解Bellman方程: {individual_id}, ω={omega_idx}, 类型={k}")
                elif hasattr(bellman_cache, 'get'):  # 增强版缓存
                    solution_shape = (n_individual_states, 3)  # 假设3个选择
                    converged_v_individual = bellman_cache.get(individual_id, type_params, int(k), solution_shape)
                    if converged_v_individual is not None:
                        individual_cache_hits += 1
                        logger.debug(f"E-step缓存命中: {individual_id}, ω={omega_idx}, 类型={k}")
                    else:
                        # 增强版缓存会自动处理热启动，这里不需要额外处理
                        logger.debug(f"E-step缓存未命中: {individual_id}, ω={omega_idx}, 类型={k}")
                else:
                    # 向后兼容：旧版缓存逻辑
                    cache_key = (individual_id, tuple(sorted(type_params.items())), int(k))
                    converged_v_individual = bellman_cache.get(cache_key)
                    if converged_v_individual is not None:
                        individual_cache_hits += 1
                    else:
                        # 热启动：尝试使用相似参数的解作为初始值
                        # 保持之前修复的形状匹配逻辑（吸取之前的教训）
                        if hasattr(bellman_cache, 'cache') and len(bellman_cache.cache) > 0:
                            try:
                                for key, cached_solution in bellman_cache.cache.items():
                                    if isinstance(cached_solution, np.ndarray) and cached_solution.shape[0] == n_individual_states:
                                        initial_v = cached_solution
                                        break
                            except Exception:
                                pass
                
                # 调试：检查变量状态
                if converged_v_individual is None and initial_v is None:
                    logger.debug(f"E-step 无解可用: {individual_id}, ω={omega_idx}, 类型={k} - 将求解Bellman方程")
                    
                    # 求解Bellman方程（此处可能需要传入ω相关值到效用函数）
                    # 简化实现：先不传ω到Bellman求解中
                    from src.model.bellman import solve_bellman_equation_individual
                    converged_v_individual, converged = solve_bellman_equation_individual(
                        utility_function=None,
                        individual_data=individual_data,
                        params=type_params,
                        agent_type=int(k),
                        beta=beta,
                        transition_matrices=transition_matrices,
                        regions_df=regions_df,
                        distance_matrix=distance_matrix,
                        adjacency_matrix=adjacency_matrix,
                        verbose=False,
                        prov_to_idx=prov_to_idx,
                        initial_v=initial_v  # 热启动
                    )
                    
                    # 检查Bellman方程是否成功求解
                    if converged and converged_v_individual is not None:
                        # 存储到增强版缓存系统或向后兼容
                        if hasattr(bellman_cache, 'put'):  # 增强版缓存
                            # 增强版缓存使用智能存储接口
                            bellman_cache.put(individual_id, type_params, int(k), converged_v_individual)
                        else:  # 旧版缓存
                            # 为旧版缓存创建传统键
                            cache_key = (individual_id, tuple(sorted(type_params.items())), int(k))
                            bellman_cache[cache_key] = converged_v_individual
                        individual_cache_misses += 1
                    else:
                        logger.warning(f"E-step Bellman方程求解失败: {individual_id}, ω={omega_idx}, 类型={k}")
                        log_lik_matrix[omega_idx, k] = -1e6
                        continue  # 跳过这个组合
                
                # 计算似然（包含工资似然）
                from src.model.likelihood import calculate_likelihood_from_v_individual
                log_lik_obs = calculate_likelihood_from_v_individual(
                    converged_v_individual=converged_v_individual,
                    params=type_params,
                    individual_data=individual_data,
                    agent_type=int(k),
                    beta=beta,
                    transition_matrices=transition_matrices,
                    regions_df=regions_df,
                    distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix,
                    prov_to_idx=prov_to_idx
                )
                
                # 汇总该个体的对数似然
                individual_log_lik = np.sum(log_lik_obs)
                log_lik_matrix[omega_idx, k] = individual_log_lik
                
            except Exception as e:
                import traceback
                logger.error(f"Error computing likelihood for individual {individual_id}, "
                           f"omega_idx={omega_idx}, type={k}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                log_lik_matrix[omega_idx, k] = -1e6
    
    # 4. 计算后验概率 p(τ, ω | D_i)
    # log p(τ, ω | D) = log π_τ + log P(ω) + log L(D | τ, ω)
    log_pi_k = np.log(np.maximum(pi_k, 1e-10))
    log_omega_probs = np.log(np.maximum(omega_probs, 1e-10))
    
    # Broadcasting: (n_omega, K)
    log_joint = (
        log_omega_probs[:, np.newaxis] +  # (n_omega, 1)
        log_pi_k[np.newaxis, :] +          # (1, K)
        log_lik_matrix                      # (n_omega, K)
    )
    
    # 归一化
    max_log_joint = np.max(log_joint)
    joint_probs = np.exp(log_joint - max_log_joint)
    joint_probs = joint_probs / np.sum(joint_probs)
    
    # 5. 边缘化ω以获得p(τ|D_i)用于更新π_k和计算似然
    marginal_type_likelihood = np.sum(np.exp(log_lik_matrix), axis=0)
    
    # 更新缓存统计
    cache_stats['cache_hits'] += individual_cache_hits
    cache_stats['cache_misses'] += individual_cache_misses
    
    return individual_id, joint_probs, marginal_type_likelihood