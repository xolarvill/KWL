"""
E-step并行处理模块 - v2.0 轻量级版本
彻底解决Windows pickle问题的全新实现
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional
from .e_step_individual_processor import process_single_individual_e_step
from ..utils.lightweight_parallel_logging import create_safe_worker_logger, log_worker_progress


def process_individual_parallel_v2(
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
    bellman_cache: Any
) -> Dict[str, Any]:
    """
    处理单个个体 - v2.0轻量级版本（完全避免pickle问题）
    
    核心改进：
    1. 不接收logger对象，避免序列化问题
    2. 返回包含日志数据的字典格式
    3. 使用可pickle的简单数据结构记录状态
    
    返回:
    ----
    字典格式：{
        'result': (individual_id, joint_probs, marginal_likelihood),
        'worker_log_data': WorkerLogData对象
    }
    """
    
    # 创建worker日志数据（可pickle的安全结构）
    worker_data = create_safe_worker_logger()
    start_time = time.time()
    
    # 缓存统计
    cache_stats = {'cache_hits': 0, 'cache_misses': 0}
    
    try:
        # 调用核心处理函数
        individual_id, joint_probs, marginal_likelihood = process_single_individual_e_step(
            individual_id=individual_id,
            individual_data=individual_data,
            omega_list=omega_list,
            omega_probs=omega_probs,
            params=params,
            pi_k=pi_k,
            K=K,
            beta=beta,
            transition_matrices=transition_matrices,
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            prov_to_idx=prov_to_idx,
            bellman_cache=bellman_cache,
            cache_stats=cache_stats
        )
        
        processing_time = time.time() - start_time
        
        # 记录成功处理
        total_cache_requests = cache_stats['cache_hits'] + cache_stats['cache_misses']
        cache_hit = (cache_stats['cache_hits'] > 0) if total_cache_requests > 0 else None
        
        log_worker_progress(
            worker_data=worker_data,
            individual_id=individual_id,
            success=True,
            cache_hit=cache_hit,
            processing_time=processing_time
        )
        
        # 返回新格式：结果 + 日志数据
        return {
            'result': (individual_id, joint_probs, marginal_likelihood),
            'worker_log_data': worker_data
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        # 记录处理错误
        log_worker_progress(
            worker_data=worker_data,
            individual_id=individual_id,
            success=False,
            error_msg=error_msg,
            processing_time=processing_time
        )
        
        # 返回错误结果和日志数据
        return {
            'result': (individual_id, np.array([]), np.full(K, -1e6)),
            'worker_log_data': worker_data
        }


def create_parallel_processing_data_v2(
    individual_omega_dict: Dict[Any, Tuple[List[Dict], np.ndarray]],
    params: Dict[str, Any],
    pi_k: np.ndarray,
    K: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    prov_to_idx: Dict[int, int],
    bellman_cache: Any
) -> Dict[str, Any]:
    """
    创建并行处理所需的数据包 - v2版本
    与v1版本相同，保持兼容性
    """
    return {
        'params': params,
        'pi_k': pi_k,
        'K': K,
        'beta': beta,
        'transition_matrices': transition_matrices,
        'regions_df': regions_df,
        'distance_matrix': distance_matrix,
        'adjacency_matrix': adjacency_matrix,
        'prov_to_idx': prov_to_idx,
        'bellman_cache': bellman_cache,
        'individual_omega_dict': individual_omega_dict
    }


def process_individual_with_data_package_v2(
    individual_id: Any,
    individual_data: Any,
    data_package: Dict[str, Any]
) -> Dict[str, Any]:
    """
    使用数据包处理单个个体 - v2.0版本
    
    这是新系统的核心函数，完全避免pickle序列化问题。
    
    返回:
    ----
    字典格式，包含处理结果和日志数据，完全可pickle
    """
    # 从数据包中提取omega信息
    omega_list, omega_probs = data_package['individual_omega_dict'][individual_id]
    
    # 调用v2版本的并行处理函数
    return process_individual_parallel_v2(
        individual_id=individual_id,
        individual_data=individual_data,
        omega_list=omega_list,
        omega_probs=omega_probs,
        params=data_package['params'],
        pi_k=data_package['pi_k'],
        K=data_package['K'],
        beta=data_package['beta'],
        transition_matrices=data_package['transition_matrices'],
        regions_df=data_package['regions_df'],
        distance_matrix=data_package['distance_matrix'],
        adjacency_matrix=data_package['adjacency_matrix'],
        prov_to_idx=data_package['prov_to_idx'],
        bellman_cache=data_package['bellman_cache']
    )