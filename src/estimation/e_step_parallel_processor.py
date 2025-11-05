"""
E-step并行处理模块 - 无闭包版本
解决序列化问题的根本方案
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from .e_step_individual_processor import process_single_individual_e_step
from ..utils.parallel_logger_registry import get_parallel_logger


def process_individual_parallel(
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
    parallel_logger_id: Optional[str] = None
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    处理单个个体 - 并行化版本（无闭包）
    
    这个函数不依赖任何外部闭包变量，所有需要的数据都通过参数传递，
    确保可以被pickle序列化，支持并行化处理。
    
    参数:
    ----
    individual_id: 个体ID
    individual_data: 个体数据(DataFrame)
    omega_list: ω列表
    omega_probs: ω概率
    params: 模型参数
    pi_k: 类型概率
    K: 类型数量
    beta: 贴现因子
    transition_matrices: 转移矩阵
    regions_df: 地区数据
    distance_matrix: 距离矩阵
    adjacency_matrix: 邻接矩阵
    prov_to_idx: 省份映射
    bellman_cache: Bellman方程缓存
    parallel_logger_id: 并行日志管理器ID
    
    返回:
    ----
    individual_id: 个体ID
    joint_probs: 联合后验概率
    marginal_likelihood: 边缘似然
    """
    
    # 缓存统计（线程局部）
    cache_stats = {'cache_hits': 0, 'cache_misses': 0}
    
    # 获取并行日志管理器（如果存在）
    parallel_logger = None
    worker_id = None
    if parallel_logger_id:
        parallel_logger = get_parallel_logger(parallel_logger_id)
        if parallel_logger:
            worker_id = parallel_logger._get_worker_id()
            parallel_logger.log_worker_start(worker_id, 1)
    
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
        
        # 记录成功处理
        if parallel_logger:
            total_cache_requests = cache_stats['cache_hits'] + cache_stats['cache_misses']
            cache_hit_rate = cache_stats['cache_hits'] / total_cache_requests if total_cache_requests > 0 else 0
            parallel_logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=True,
                cache_hit=cache_hit_rate > 0 if total_cache_requests > 0 else None
            )
        
        return individual_id, joint_probs, marginal_likelihood
        
    except Exception as e:
        # 记录处理错误
        if parallel_logger:
            parallel_logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=False,
                error_msg=str(e)
            )
        
        # 返回空结果，将在主函数中处理
        return individual_id, np.array([]), np.full(K, -1e10)


def create_parallel_processing_data(
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
    创建并行处理所需的数据包
    
    将所有需要的数据打包成一个字典，避免闭包依赖。
    
    返回:
    ----
    包含所有处理参数的数据字典
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


def process_individual_with_data_package(
    individual_id: Any,
    individual_data: Any,
    data_package: Dict[str, Any],
    parallel_logger_id: Optional[str] = None
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    使用数据包处理单个个体
    
    这是真正可以被joblib并行调用的函数，所有数据都通过参数传递。
    """
    # 从数据包中提取omega信息
    omega_list, omega_probs = data_package['individual_omega_dict'][individual_id]
    
    # 调用处理函数
    return process_individual_parallel(
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
        bellman_cache=data_package['bellman_cache'],
        parallel_logger_id=parallel_logger_id
    )