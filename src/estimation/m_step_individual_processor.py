"""
M-step个体处理模块 - 支持并行化
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional

def process_single_individual_m_step(
    individual_id: Any,
    individual_data: Any,
    posterior_matrix: np.ndarray,
    omega_list: List[Dict],
    params: Dict[str, Any],
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    prov_to_idx: Dict[int, int],
    bellman_cache: Any,
    weight_threshold: float = 1e-10
) -> Tuple[float, int, int]:
    """
    处理单个个体的M-step计算
    
    返回:
        individual_contribution: 该个体的加权对数似然贡献
        n_valid_computations: 有效计算次数
        n_failed_computations: 失败计算次数
    """
    logger = logging.getLogger()
    
    individual_contribution = 0.0
    n_valid_computations = 0
    n_failed_computations = 0
    
    # 提取显著权重组合
    omega_indices, type_indices = np.where(posterior_matrix > weight_threshold)
    
    if len(omega_indices) == 0:
        n_failed_computations += 1
        return individual_contribution, n_valid_computations, n_failed_computations
    
    # 计算个体状态数量（用于热启动验证）
    n_individual_states = len(individual_data['visited_locations'].iloc[0]) * len(individual_data['age_t'].unique())
    
    # 遍历每个显著 (ω, τ) 组合
    for idx in range(len(omega_indices)):
        omega_idx = omega_indices[idx]
        k = type_indices[idx]
        weight = posterior_matrix[omega_idx, k]
        
        # 构建 type-specific 参数
        type_params = params.copy()
        if f'gamma_0_type_{k}' in params:
            type_params['gamma_0'] = params[f'gamma_0_type_{k}']
        sigma_epsilon = omega_list[omega_idx]['sigma']
        type_params['sigma_epsilon'] = sigma_epsilon
        
        # 参数有效性检查
        if not (np.isfinite(sigma_epsilon) and sigma_epsilon > 0):
            n_failed_computations += 1
            continue
        
        try:
            # 检查本地缓存 - 使用增强版缓存系统
            converged_v_individual = None
            initial_v = None
            converged = False
            
            logger.debug(f"M-step 缓存查询: {individual_id}, ω={omega_idx}, 类型={k}, 缓存类型={type(bellman_cache)}")
            
            if hasattr(bellman_cache, 'get'):  # 增强版缓存
                solution_shape = (n_individual_states, 3)  # 假设3个选择
                logger.debug(f"M-step 使用增强版缓存查询: individual_id={individual_id}, shape={solution_shape}, 关键参数={ {k:v for k,v in type_params.items() if 'gamma' in k or 'sigma' in k} }")
                converged_v_individual = bellman_cache.get(individual_id, type_params, int(k), solution_shape)
                logger.debug(f"M-step 缓存查询结果: converged_v_individual_is_none={converged_v_individual is None}")
                if converged_v_individual is not None:
                    converged = True
                    logger.debug(f"M-step缓存命中: {individual_id}, ω={omega_idx}, 类型={k}")
                else:
                    # 增强版缓存会自动处理热启动
                    logger.debug(f"M-step缓存未命中: {individual_id}, ω={omega_idx}, 类型={k}")
                    
                    # 检查缓存状态（调试用）
                    if hasattr(bellman_cache, 'get_stats'):
                        try:
                            cache_stats = bellman_cache.get_stats()
                            logger.debug(f"M-step 缓存状态: L1大小={cache_stats.get('l1_stats', {}).get('size', 'N/A')}, "
                                       f"L2大小={cache_stats.get('l2_cache_size', 'N/A')}, "
                                       f"总请求={cache_stats.get('total_requests', 'N/A')}")
                        except Exception as e:
                            logger.debug(f"M-step 获取缓存状态失败: {e}")
            else:
                # 向后兼容：旧版缓存逻辑（保持之前修复的正确逻辑）
                logger.debug(f"M-step 使用旧版缓存查询: individual_id={individual_id}")
                cache_key = (individual_id, tuple(sorted(type_params.items())), int(k))
                converged_v_individual = bellman_cache.get(cache_key)
                if converged_v_individual is not None:
                    converged = True
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
                    else:
                        # 普通dict
                        for key in reversed(list(bellman_cache.keys())):
                            cached_solution = bellman_cache[key]
                            if isinstance(cached_solution, np.ndarray) and cached_solution.shape[0] == n_individual_states:
                                initial_v = cached_solution
                                break
                
                # 求解 Bellman 方程
                logger.debug(f"M-step 开始求解Bellman方程: {individual_id}, ω={omega_idx}, 类型={k}")
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
                
                logger.debug(f"M-step Bellman方程求解完成: converged={converged}, v_individual_is_none={converged_v_individual is None}")
                
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
                else:
                    logger.warning(f"Bellman方程求解失败: {individual_id}, ω={omega_idx}, 类型={k}, converged={converged}, v_is_none={converged_v_individual is None}")
                    n_failed_computations += 1
                    continue
            
            if not converged:
                n_failed_computations += 1
                continue
            
            if np.any(~np.isfinite(converged_v_individual)):
                n_failed_computations += 1
                continue
            
            # 计算个体似然
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
            
            individual_log_lik = np.sum(log_lik_obs)
            if not np.isfinite(individual_log_lik):
                n_failed_computations += 1
                continue
            
            # 加权贡献
            weighted_contribution = weight * individual_log_lik
            individual_contribution += weighted_contribution
            n_valid_computations += 1
            
        except Exception as e:
            logger.debug(f"Error for ind={individual_id}, ω_idx={omega_idx}, τ={k}: {e}")
            n_failed_computations += 1
            continue  # 跳过该组合
    
    return individual_contribution, n_valid_computations, n_failed_computations