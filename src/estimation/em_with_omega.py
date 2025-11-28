"""
扩展的EM算法E-step和M-step

集成离散支撑点，计算p(τ, ω|D)后验概率
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, TYPE_CHECKING
import logging
import time
import os
from joblib import Parallel, delayed
from src.utils.memory_monitor import MemoryMonitor, create_memory_safe_config, log_memory_usage

# Avoid runtime import cycles; make the EstimationProgressTracker type available for type checkers only
if TYPE_CHECKING:
    from src.utils.progress_tracker import EstimationProgressTracker

from src.model.discrete_support import (
    DiscreteSupportGenerator,
    SimplifiedOmegaEnumerator
)
from src.model.likelihood import (
    calculate_likelihood_from_v_individual,
    _make_cache_key,
    clear_bellman_cache,
    _BELLMAN_CACHE as GLOBAL_BELLMAN_CACHE
)
from src.model.bellman import solve_bellman_equation_individual
from src.model.wage_equation import calculate_predicted_wage, calculate_reference_wage
from src.model.smart_cache import EnhancedBellmanCache, create_enhanced_cache
from src.utils.parallel_wrapper import ParallelConfig, parallel_individual_processor
from src.utils.parallel_logger_registry import register_parallel_logger, unregister_parallel_logger


def _calculate_individual_state_space_size(individual_data: pd.DataFrame) -> int:
    """
    计算个体状态空间大小，用于热启动验证
    """
    visited_locations = individual_data['visited_locations'].iloc[0]
    n_visited_locations = len(visited_locations)
    ages = sorted(individual_data['age_t'].unique())
    n_ages = len(ages)
    return n_ages * n_visited_locations


def _pack_params(params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    将参数字典打包为数组和名称列表

    参数:
    ----
    params : Dict[str, Any]
        参数字典

    返回:
    ----
    Tuple[np.ndarray, List[str]]
        (参数值数组, 参数名称列表)
    """
    # 显式排除 gamma_0_type_0（归一化为0）和 n_choices
    param_names = sorted([k for k in params.keys() if k not in ['n_choices', 'gamma_0_type_0']])
    param_values = np.array([params[name] for name in param_names])
    return param_values, param_names


def _unpack_params(param_values: np.ndarray, param_names: List[str], n_choices: int) -> Dict[str, Any]:
    """
    将参数数组解包为参数字典

    参数:
    ----
    param_values : np.ndarray
        参数值数组
    param_names : List[str]
        参数名称列表
    n_choices : int
        选择数量

    返回:
    ----
    Dict[str, Any]
        参数字典
    """
    params_dict = dict(zip(param_names, param_values))
    params_dict['n_choices'] = n_choices
    # 将被归一化的参数重新加回字典
    params_dict['gamma_0_type_0'] = 0.0
    return params_dict


def _calculate_logging_interval(total_individuals: int) -> int:
    """
    根据总个体数量计算动态日志输出间隔。
    间隔为总个体数量的5%，然后取最接近的10^N或5*10^N。

    参数:
    ----
    total_individuals : int
        总个体数量。

    返回:
    ----
    int
        计算出的日志输出间隔。
    """
    if total_individuals <= 100:
        return 10  # 对于小样本，保持较频繁的更新
    
    target_interval = max(1, int(total_individuals * 0.05))
    
    # 寻找最接近的10^N或5*10^N
    best_interval = 1
    min_diff = abs(target_interval - 1)

    for power in range(1, 10):  # 考虑足够大的范围
        val_10 = 10 ** power
        val_5_10 = 5 * (10 ** power)

        diff_10 = abs(target_interval - val_10)
        diff_5_10 = abs(target_interval - val_5_10)

        if diff_10 < min_diff:
            min_diff = diff_10
            best_interval = val_10
        
        if diff_5_10 < min_diff:
            min_diff = diff_5_10
            best_interval = val_5_10
        
        if val_10 > target_interval * 2 and val_5_10 > target_interval * 2: # 避免过大的间隔
            break

    # 确保间隔不会超过总个体数的一半，且至少为1
    return max(1, min(best_interval, total_individuals // 2))



# 计算后验概率
def e_step_with_omega(
    params: Dict[str, Any],
    pi_k: np.ndarray,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    support_generator: DiscreteSupportGenerator,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 1000,
    use_simplified_omega: bool = True,
    bellman_cache=None,  # 移除类型注解，支持多种缓存类型
    parallel_config: Optional[ParallelConfig] = None,  # 新增并行配置参数
    memory_safe_mode: bool = False  # 内存安全模式
) -> Tuple[Dict[Any, np.ndarray], np.ndarray]:
    """
    扩展的E-step：计算p(τ, ω_i | D_i)后验概率

    根据论文公式(1120-1123行):
        p(τ, ω_i | D_i) ∝ π_τ · P(ω_i) · L(D_i | τ, ω_i, Θ)

    参数:
    ----
    params : Dict[str, Any]
        当前参数估计值
    pi_k : np.ndarray
        类型概率 (K,)
    observed_data : pd.DataFrame
        观测数据
    state_space : pd.DataFrame
        状态空间
    support_generator : DiscreteSupportGenerator
        支撑点生成器
    n_types : int
        类型数量K
    prov_to_idx : Dict[int, int]
        省份ID到矩阵索引的映射
    max_omega_per_individual : int
        每个个体的最大ω组合数
    use_simplified_omega : bool
        是否使用简化策略（只在访问过的地区实例化ν和ξ）
    parallel_config : ParallelConfig, optional
        并行配置，如果为None则使用串行处理

    返回:
    ----
    individual_posteriors : Dict[individual_id, np.ndarray]
        每个个体的后验概率矩阵 (n_omega_i * K,)
        其中每行对应一个(ω, τ)组合
    log_likelihood_matrix : np.ndarray
        用于计算总似然的对数似然矩阵 (N, K)
    """
    logger = logging.getLogger()
    unique_individuals = observed_data['individual_id'].unique()
    N = len(unique_individuals)
    K = n_types

    logger.info(f"  [E-step with ω] Processing {N} individuals across {K} types...")
    start_time = time.time()
    
    # 设置默认并行配置
    if parallel_config is None:
        parallel_config = ParallelConfig(n_jobs=1)  # 默认串行处理
    
    logger.info(f"  [E-step with ω] Parallel configuration: {parallel_config}")
    
    # 记录初始内存使用情况
    log_memory_usage("E-step开始")
    
    # 内存安全模式：使用智能内存监控
    if memory_safe_mode and parallel_config.is_parallel_enabled():
        memory_config = create_memory_safe_config(parallel_config, N, memory_safe_mode=True)
        if memory_config:
            # 调整并行参数
            if memory_config['n_jobs'] < parallel_config.n_jobs:
                logger.info(f"  [E-step with ω] 内存安全模式：将工作进程数从 {parallel_config.n_jobs} 调整为 {memory_config['n_jobs']}")
                parallel_config.n_jobs = memory_config['n_jobs']
            
            # 调整缓存大小
            if hasattr(bellman_cache, 'memory_limit_mb') and bellman_cache.memory_limit_mb > memory_config.get('memory_limit_mb', 1000):
                old_limit = bellman_cache.memory_limit_mb
                bellman_cache.memory_limit_mb = memory_config['memory_limit_mb']
                logger.info(f"  [E-step with ω] 内存安全模式：调整缓存内存限制从 {old_limit}MB 到 {bellman_cache.memory_limit_mb}MB")
    
    # 初始化缓存 - 使用增强版缓存系统（大幅提升容量和智能性）
    if bellman_cache is None:
        # 创建增强版缓存实例，容量从500提升到2000
        from src.model.smart_cache import create_enhanced_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
        logger.info("使用增强版缓存系统: 容量2000, 内存限制2000MB")
    elif isinstance(bellman_cache, dict):
        # 向后兼容：如果是普通dict，转换为增强缓存
        from src.model.smart_cache import create_enhanced_cache
        old_cache = bellman_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
        logger.info(f"迁移旧缓存数据: {len(old_cache)} 个条目")
        # 迁移现有数据（如果可能）
        for key, value in old_cache.items():
            try:
                # 尝试解析旧键格式并迁移
                if isinstance(key, tuple) and len(key) >= 3:
                    individual_id, old_params, agent_type = key[0], {}, key[-1]
                    # 简化的参数迁移
                    bellman_cache.l1_cache.put(key, value)
            except Exception as e:
                logger.debug(f"缓存迁移跳过某个键: {e}")
    elif hasattr(bellman_cache, 'capacity') and bellman_cache.capacity < 1000:
        # 如果现有缓存容量太小，创建新的增强缓存
        from src.model.smart_cache import create_enhanced_cache
        old_cache = bellman_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
        logger.info(f"替换小容量缓存: {old_cache.capacity} -> 2000")
    
    cache_hits = 0
    cache_misses = 0
    hot_start_attempts = 0
    hot_start_successes = 0

    # 初始化结果容器
    individual_posteriors = {}
    log_likelihood_matrix = np.zeros((N, K))

    # 初始化枚举器
    enumerator = SimplifiedOmegaEnumerator(support_generator) if use_simplified_omega else None

    # **性能优化**: 并行枚举ω
    def enumerate_omega_for_individual_wrapper(individual_id):
        """并行ω枚举的包装函数"""
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        if use_simplified_omega:
            omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
                individual_data,
                max_combinations=max_omega_per_individual
            )
        else:
            logger.warning("Full omega enumeration not yet implemented, using simplified")
            omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
                individual_data,
                max_combinations=max_omega_per_individual
            )
        return individual_id, omega_list, omega_probs

    # 并行生成所有个体的ω，限制并行核心数以避免内存溢出
    logger.info(f"  Enumerating ω for {N} individuals (parallel)...")
    omega_results = Parallel(n_jobs=min(4, N), verbose=0)(
        delayed(enumerate_omega_for_individual_wrapper)(ind_id)
        for ind_id in unique_individuals
    )

    # 将结果组织到字典中
    individual_omega_dict = {
        ind_id: (omega_list, omega_probs)
        for ind_id, omega_list, omega_probs in omega_results
    }

    # 个体处理函数（用于并行化）
    def process_individual(individual_id, parallel_logger_id=None):
        """处理单个个体"""
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        omega_list, omega_probs = individual_omega_dict[individual_id]
        
        # 缓存统计（线程局部）
        cache_stats = {'cache_hits': 0, 'cache_misses': 0}
        
        # 从外部模块导入处理函数
        from src.estimation.e_step_individual_processor import process_single_individual_e_step
        from src.utils.parallel_logger_registry import get_parallel_logger
        
        # 获取并行日志管理器（如果存在）
        parallel_logger = None
        worker_id = None
        if parallel_logger_id:
            parallel_logger = get_parallel_logger(parallel_logger_id)
            if parallel_logger:
                worker_id = parallel_logger._get_worker_id()
                parallel_logger.log_worker_start(worker_id, 1)
        
        try:
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
            
            # 更新全局缓存统计（串行模式）
            if not parallel_logger:
                cache_hits += cache_stats['cache_hits']
                cache_misses += cache_stats['cache_misses']
            
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
            return individual_id, np.array([]), np.full(K, -1e6)

    # 根据配置选择处理方式
    # 内存安全模式：动态调整批次大小
    if memory_safe_mode:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < 8:
            BATCH_SIZE = 1000
        elif available_memory_gb < 16:
            BATCH_SIZE = 2000
        else:
            BATCH_SIZE = 3000
        
        # 大样本时进一步减小批次
        if N > 20000:
            BATCH_SIZE = min(BATCH_SIZE, 1500)
        
        logger.info(f"  内存安全模式：批次大小调整为 {BATCH_SIZE}")
    else:
        BATCH_SIZE = 5000  # 默认批次大小
    
    # 使用分批处理避免内存问题
    if parallel_config.is_parallel_enabled() and N > BATCH_SIZE:
        logger.info(f"  使用分批并行处理，{parallel_config.n_jobs} 个工作进程，每批 {BATCH_SIZE} 个个体")
        
        # 分批处理
        all_individual_posteriors = {}
        all_log_likelihood_matrix = np.zeros((N, K))
        
        for batch_start in range(0, N, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N)
            batch_individuals = unique_individuals[batch_start:batch_end]
            batch_size = len(batch_individuals)
            
            logger.info(f"  处理批次 {batch_start//BATCH_SIZE + 1}: 个体 {batch_start+1}-{batch_end}")
            
            # 为当前批次创建数据子集
            batch_omega_dict = {ind_id: individual_omega_dict[ind_id] for ind_id in batch_individuals}
            
            # 创建批次数据包
            from src.estimation.e_step_parallel_processor import create_parallel_processing_data, process_individual_with_data_package
            
            batch_data_package = create_parallel_processing_data(
                individual_omega_dict=batch_omega_dict,
                params=params,
                pi_k=pi_k,
                K=K,
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                prov_to_idx=prov_to_idx,
                bellman_cache=bellman_cache
            )
            
            # 处理当前批次
            def get_parallel_config():
                from src.utils.lightweight_parallel_wrapper import create_simple_parallel_config
                return create_simple_parallel_config(n_jobs=parallel_config.n_jobs)
                
            from src.utils.lightweight_parallel_wrapper import lightweight_parallel_processor
            @lightweight_parallel_processor(config_getter=get_parallel_config, quiet_mode=True, 
                                          enable_memory_monitoring=memory_safe_mode)
            def process_batch_individuals(batch_individual_ids):
                results = []
                for ind_id in batch_individual_ids:
                    result = process_individual_with_data_package(
                        ind_id,
                        observed_data[observed_data['individual_id'] == ind_id],
                        batch_data_package
                    )
                    results.append(result)
                return results
            
            # 执行批次处理
            try:
                batch_results = process_batch_individuals(batch_individuals)
                
                # 整理批次结果
                for i_idx, result_data in enumerate(batch_results):
                    if isinstance(result_data, dict) and 'result' in result_data:
                        individual_result = result_data['result']
                    else:
                        individual_result = result_data
                    
                    if isinstance(individual_result, tuple) and len(individual_result) == 3:
                        individual_id, joint_probs, marginal_likelihood = individual_result
                        if joint_probs.size > 0:
                            all_individual_posteriors[individual_id] = joint_probs
                            all_log_likelihood_matrix[batch_start + i_idx, :] = marginal_likelihood
                        else:
                            all_log_likelihood_matrix[batch_start + i_idx, :] = -1e6
                    else:
                        all_log_likelihood_matrix[batch_start + i_idx, :] = -1e6
                        
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                # 回退到串行处理当前批次
                logger.info("  回退到串行处理当前批次")
                for i_idx, individual_id in enumerate(batch_individuals):
                    individual_id, joint_probs, marginal_likelihood = process_individual(individual_id)
                    if joint_probs.size > 0:
                        all_individual_posteriors[individual_id] = joint_probs
                        all_log_likelihood_matrix[batch_start + i_idx, :] = marginal_likelihood
                    else:
                        all_log_likelihood_matrix[batch_start + i_idx, :] = -1e6
        
        return all_individual_posteriors, all_log_likelihood_matrix
    elif parallel_config.is_parallel_enabled():
        logger.info(f"  使用并行个体处理，{parallel_config.n_jobs} 个工作进程")
        
        # 创建并行处理数据包（v2.0轻量级版本，解决pickle问题）
        from src.estimation.e_step_parallel_processor import create_parallel_processing_data, process_individual_with_data_package
        
        data_package = create_parallel_processing_data(
            individual_omega_dict=individual_omega_dict,
            params=params,
            pi_k=pi_k,
            K=K,
            beta=beta,
            transition_matrices=transition_matrices,
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            prov_to_idx=prov_to_idx,
            bellman_cache=bellman_cache  # 传递缓存
        )
        
        # 使用新的轻量级并行处理装饰器，完全避免Windows pickle问题
        from src.utils.lightweight_parallel_wrapper import create_simple_parallel_config, lightweight_parallel_processor
        
        # 创建并行配置
        parallel_config_wrapper = create_simple_parallel_config(n_jobs=parallel_config.n_jobs)
        
        # 创建包装后的处理函数，直接使用配置
        def get_parallel_config():
            return parallel_config_wrapper
            
        @lightweight_parallel_processor(config_getter=get_parallel_config, quiet_mode=True,
                                      enable_memory_monitoring=memory_safe_mode, 
                                      disable_stats_output=True)  # 【修复】禁用统计输出，避免与E-step统计重复
        def process_individuals_batch(individual_ids):
            """批量处理个体的包装函数"""
            results = []
            for ind_id in individual_ids:
                # 调用v2版本的并行处理函数
                result = process_individual_with_data_package(
                    ind_id,
                    observed_data[observed_data['individual_id'] == ind_id],
                    data_package
                )
                results.append(result)
            return results
        
        try:
            # 使用新的轻量级并行系统处理所有个体
            individual_results = process_individuals_batch(unique_individuals)
            
            # 整理结果（适配新系统的返回格式）
            # 新系统已经处理了日志聚合，直接提取结果
            for i_idx, result_data in enumerate(individual_results):
                # 提取个体结果
                if isinstance(result_data, dict) and 'result' in result_data:
                    individual_result = result_data['result']
                else:
                    individual_result = result_data
                
                # 提取个体结果
                if isinstance(individual_result, tuple) and len(individual_result) == 3:
                    individual_id, joint_probs, marginal_likelihood = individual_result
                    if joint_probs.size > 0:  # 确保有有效结果
                        individual_posteriors[individual_id] = joint_probs
                        log_likelihood_matrix[i_idx, :] = marginal_likelihood
                    else:
                        log_likelihood_matrix[i_idx, :] = -1e6
                else:
                    # 处理错误情况
                    log_likelihood_matrix[i_idx, :] = -1e6
            
        except Exception as e:
            logger.error(f"并行处理失败，回退到串行模式: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 回退到串行处理
            parallel_config = ParallelConfig(n_jobs=1)
    
    if not parallel_config.is_parallel_enabled():
        logger.info("  使用串行个体处理模式")
        # 原有的串行处理逻辑
        logging_interval = _calculate_logging_interval(N)
        
        for i_idx, individual_id in enumerate(unique_individuals):
            if (i_idx + 1) % logging_interval == 0:
                current_time = time.time()
                if i_idx > 0:
                    elapsed = current_time - start_time
                    rate = (i_idx + 1) / elapsed
                    remaining = (N - i_idx - 1) / rate
                    logger.info(f"    Processing individual {i_idx+1}/{N} "
                              f"(rate: {rate:.1f} ind/s, est. remaining: {remaining:.1f}s)")
                else:
                    logger.info(f"    Processing individual {i_idx+1}/{N}")

            # 处理单个个体（原有的串行逻辑）
            individual_id, joint_probs, marginal_likelihood = process_individual(individual_id)
            
            if joint_probs.size > 0:  # 确保有有效结果
                individual_posteriors[individual_id] = joint_probs
                log_likelihood_matrix[i_idx, :] = marginal_likelihood
            else:
                log_likelihood_matrix[i_idx, :] = -1e6

    total_time = time.time() - start_time
    cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
    hot_start_rate = hot_start_successes / hot_start_attempts if hot_start_attempts > 0 else 0
    logger.info(f"  [E-step with ω] Completed. Total time: {total_time:.1f}s, "
                f"Rate: {N/total_time:.1f} individuals/second")
    logger.info(f"  [E-step with ω] Cache hit rate: {cache_hit_rate:.1%} ({cache_hits}/{cache_hits + cache_misses})")
    logger.info(f"  [E-step with ω] Hot-start success rate: {hot_start_rate:.1%} ({hot_start_successes}/{hot_start_attempts})")
    
    # 增强版缓存统计信息
    if hasattr(bellman_cache, 'get_stats'):
        cache_stats = bellman_cache.get_stats()
        logger.info(f"  [E-step with ω] 增强缓存统计: L1命中率={cache_stats['l1_hit_rate']:.1%}, "
                   f"L2相似性命中率={cache_stats['l2_hit_rate']:.1%}, "
                   f"总命中率={cache_stats['total_hit_rate']:.1%}, "
                   f"相似性匹配={cache_stats['similarity_hits']}")
        # 详细的L1缓存统计
        if cache_stats.get('l1_stats'):
            l1_stats = cache_stats['l1_stats']
            logger.info(f"  [E-step with ω] L1缓存状态: {l1_stats['size']}/{l1_stats['max_size']}条目, "
                       f"{l1_stats['memory_mb']:.1f}/{l1_stats['max_memory_mb']:.1f}MB, "
                       f"淘汰: {l1_stats['eviction_count']}次")
    else:
        # 向后兼容：旧版缓存统计
        if hasattr(bellman_cache, 'get_stats'):
            cache_stats = bellman_cache.get_stats()
            logger.info(f"  [E-step with ω] LRU缓存状态: {cache_stats['size']}/{cache_stats['max_size']}条目, "
                       f"{cache_stats['memory_mb']:.1f}/{cache_stats['max_memory_mb']:.1f}MB, "
                       f"淘汰: {cache_stats['eviction_count']}次, 命中率: {cache_stats['hit_rate']:.1%}")

    # 记录总体处理统计
    total_time = time.time() - start_time
    cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
    hot_start_rate = hot_start_successes / hot_start_attempts if hot_start_attempts > 0 else 0
    
    logger.info(f"  [E-step with ω] Completed. Total time: {total_time:.1f}s, "
                f"Rate: {N/total_time:.1f} individuals/second")
    logger.info(f"  [E-step with ω] Cache hit rate: {cache_hit_rate:.1%} ({cache_hits}/{cache_hits + cache_misses})")
    logger.info(f"  [E-step with ω] Hot-start success rate: {hot_start_rate:.1%} ({hot_start_successes}/{hot_start_attempts})")

    return individual_posteriors, log_likelihood_matrix


def _m_step_objective_worker(
    individual_id: Any,
    data_package: Dict[str, Any]
) -> Dict[str, Any]:
    """
    M-step worker function to compute the weighted log-likelihood for a single individual.
    Returns the individual's contribution and cache statistics.
    """
    # 1. Unpack data from the package
    params = data_package['params']
    observed_data = data_package['observed_data']
    individual_posteriors = data_package['individual_posteriors']
    individual_omega_lists = data_package['individual_omega_lists']
    beta = data_package['beta']
    transition_matrices = data_package['transition_matrices']
    regions_df_np = data_package['regions_df_np']
    distance_matrix = data_package['distance_matrix']
    adjacency_matrix = data_package['adjacency_matrix']
    prov_to_idx = data_package['prov_to_idx']
    bellman_cache = data_package['bellman_cache']
    shared_cache_dict = data_package.get('shared_cache_dict', None)

    # 2. Logic from the original for-loop body
    individual_data = observed_data[observed_data['individual_id'] == individual_id]
    posterior_matrix = individual_posteriors[individual_id]
    omega_list = individual_omega_lists[individual_id]
    
    individual_contribution = 0.0
    weight_threshold = 1e-10
    omega_indices, type_indices = np.where(posterior_matrix > weight_threshold)

    # 缓存统计
    cache_hits = 0
    cache_misses = 0

    if len(omega_indices) == 0:
        return {
            'contribution': 0.0,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses
        }

    n_individual_states = _calculate_individual_state_space_size(individual_data)

    for idx in range(len(omega_indices)):
        omega_idx = omega_indices[idx]
        k = type_indices[idx]
        weight = posterior_matrix[omega_idx, k]

        type_params = params.copy()
        if f'gamma_0_type_{k}' in params:
            type_params['gamma_0'] = params[f'gamma_0_type_{k}']
        sigma_epsilon = omega_list[omega_idx]['sigma']
        type_params['sigma_epsilon'] = sigma_epsilon

        if not (np.isfinite(sigma_epsilon) and sigma_epsilon > 0):
            continue

        try:
            # Assuming 3 choices for now. This might need to be made dynamic if n_choices varies.
            solution_shape = (n_individual_states, 3) 
            
            # 首先尝试从共享缓存中获取
            converged_v_individual = None
            if shared_cache_dict is not None:
                # 创建缓存键
                try:
                    import pickle
                    from src.model.smart_cache import SmartCacheKey
                    cache_key_generator = SmartCacheKey()
                    cache_key = cache_key_generator.create_key(individual_id, type_params, int(k))
                    serialized_key = pickle.dumps(cache_key)
                    
                    # 从共享缓存中获取
                    if serialized_key in shared_cache_dict:
                        serialized_value = shared_cache_dict[serialized_key]
                        converged_v_individual = pickle.loads(serialized_value)
                        cache_hits += 1
                    else:
                        cache_misses += 1
                except Exception:
                    # 如果共享缓存访问失败，继续使用原始缓存
                    cache_misses += 1
            else:
                # 使用原始缓存
                converged_v_individual = bellman_cache.get(individual_id, type_params, int(k), solution_shape)
                if converged_v_individual is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1

            converged = converged_v_individual is not None

            if not converged:
                converged_v_individual, converged = solve_bellman_equation_individual(
                    utility_function=None,
                    individual_data=individual_data,
                    params=type_params,
                    agent_type=int(k),
                    beta=beta,
                    transition_matrices=transition_matrices,
                    regions_df=regions_df_np,
                    distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix,
                    verbose=False,
                    prov_to_idx=prov_to_idx,
                    initial_v=None
                )
                
                # 如果求解成功，更新缓存
                if converged and converged_v_individual is not None:
                    cache_misses += 1  # 这是一次缓存未命中后的成功计算
                    # 将结果存储到共享缓存中（如果可用）
                    if shared_cache_dict is not None:
                        try:
                            import pickle
                            from src.model.smart_cache import SmartCacheKey
                            cache_key_generator = SmartCacheKey()
                            cache_key = cache_key_generator.create_key(individual_id, type_params, int(k))
                            serialized_key = pickle.dumps(cache_key)
                            serialized_value = pickle.dumps(converged_v_individual)
                            shared_cache_dict[serialized_key] = serialized_value
                        except Exception:
                            # 如果无法存储到共享缓存，继续执行
                            pass

            if not converged or converged_v_individual is None or np.any(~np.isfinite(converged_v_individual)):
                continue

            log_lik_obs = calculate_likelihood_from_v_individual(
                converged_v_individual=converged_v_individual,
                params=type_params,
                individual_data=individual_data,
                agent_type=int(k),
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df_np,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                prov_to_idx=prov_to_idx
            )

            individual_log_lik = np.sum(log_lik_obs)
            if np.isfinite(individual_log_lik):
                individual_contribution += weight * individual_log_lik

        except Exception:
            # In a parallel worker, it's better to fail gracefully for one individual
            continue

    return {
        'contribution': individual_contribution,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses
    }


def m_step_with_omega(
    individual_posteriors: Dict[Any, np.ndarray],
    initial_params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    support_generator: DiscreteSupportGenerator,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 1000,
    lbfgsb_maxiter: int = 15,
    bellman_cache: Dict = None,
    parallel_config: Optional['ParallelConfig'] = None,  # 新增并行配置参数
    lbfgsb_gtol: float = None,  # 临时调整L-BFGS-B梯度容差
    lbfgsb_ftol: float = None,  # 临时调整L-BFGS-B函数值容差
    m_step_backend: str = 'threading',  # M步并行化后端
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    扩展的M-step：对ω进行加权求和来更新参数
    """
    logger = logging.getLogger()
    logger.info(f"\n  [M-step with ω] Starting parameter optimization...")

    from scipy.optimize import minimize
    from src.config.model_config import ModelConfig
    
    # 设置默认并行配置
    if parallel_config is None:
        parallel_config = ParallelConfig(n_jobs=1)  # 默认串行处理
    
    logger.info(f"  [M-step with ω] Parallel configuration: {parallel_config}")
    
    # 传递缓存给目标函数 - 使用增强版缓存系统
    if bellman_cache is None:
        # 创建增强版缓存实例（与E步保持一致）
        from src.model.smart_cache import create_enhanced_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)  # M步专用缓存
    elif isinstance(bellman_cache, dict):
        # 向后兼容：如果是普通dict，转换为增强版缓存
        from src.model.smart_cache import create_enhanced_cache
        old_cache = bellman_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
        logger.info(f"M-step迁移缓存数据: {len(old_cache)} 个条目")
        # 迁移现有数据（使用增强版缓存的标准接口）
        for key, value in old_cache.items():
            try:
                if hasattr(bellman_cache, 'put'):
                    # 如果可能，解析旧键格式进行智能迁移
                    bellman_cache.put(key, value)  # 简化迁移，使用标准接口
                else:
                    bellman_cache[key] = value  # 向后兼容
            except Exception as e:
                logger.debug(f"M-step缓存迁移跳过某个键: {e}")
    elif not hasattr(bellman_cache, 'get'):  # 如果是旧版LRUCache或其他类型
        # 转换为增强版缓存
        from src.model.smart_cache import create_enhanced_cache
        old_cache = bellman_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
        logger.info(f"M-step升级缓存系统: {type(old_cache)} -> 增强版缓存")

    config = ModelConfig()
    unique_individuals = list(individual_posteriors.keys())
    N = len(unique_individuals)
    K = n_types

    individual_to_idx = {ind_id: idx for idx, ind_id in enumerate(unique_individuals)}

    # 预枚举所有个体的ω
    enumerator = SimplifiedOmegaEnumerator(support_generator)
    individual_omega_lists = {}

    logger.info("  Pre-enumerating omega for all individuals...")
    for individual_id in unique_individuals:
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        omega_list, _ = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=max_omega_per_individual
        )
        individual_omega_lists[individual_id] = omega_list

    # --- 目标函数类 ---
    class Objective:
        def __init__(self, shared_bellman_cache=None, parallel_config=None, m_step_backend='threading'):
            self.call_count = 0
            self.last_call_end_time = time.time()
            # 在目标函数调用之间缓存Bellman解
            self.bellman_cache = shared_bellman_cache if shared_bellman_cache is not None else {}
            self.cache_hits = 0
            self.cache_misses = 0
            self.hot_start_attempts = 0
            self.hot_start_successes = 0
            # 并行配置
            self.parallel_config = parallel_config or ParallelConfig(n_jobs=1)
            # M步并行化后端
            self.m_step_backend = m_step_backend
            logger.info(f"  [Debug] Objective.__init__中接收到的m_step_backend参数: {m_step_backend}")
            
            # 确保是增强版缓存实例
            log_msg_header = "  [M-step]"
            logger.info(f"{log_msg_header} Objective初始化，缓存类型: {type(self.bellman_cache)}")
            
            if isinstance(self.bellman_cache, dict) and len(self.bellman_cache) > 0:
                # 如果是普通dict且不为空，升级为增强版缓存
                logger.info(f"{log_msg_header} 升级普通dict为增强版缓存系统")
                from src.model.smart_cache import create_enhanced_cache
                old_cache = self.bellman_cache
                self.bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
                # 迁移数据（使用标准接口）
                for key, value in old_cache.items():
                    try:
                        if hasattr(self.bellman_cache, 'put'):
                            self.bellman_cache.put(key, value)  # 使用标准接口
                        else:
                            self.bellman_cache[key] = value  # 向后兼容
                    except Exception as e:
                        logger.debug(f"缓存迁移跳过: {e}")
            elif str(type(self.bellman_cache)).find('LRUCache') >= 0:  # 如果是旧版LRUCache
                # 旧版缓存，升级到增强版
                logger.info(f"{log_msg_header} 升级LRUCache为增强版缓存系统")
                from src.model.smart_cache import create_enhanced_cache
                self.bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)
            elif str(type(self.bellman_cache)).find('EnhancedBellmanCache') >= 0:  # 如果是增强版缓存
                # 增强版缓存，记录初始统计
                try:
                    initial_stats = self.bellman_cache.get_stats()
                    l1_hit_rate = initial_stats.get('l1_hit_rate', 0)
                    total_hit_rate = initial_stats.get('total_hit_rate', 0)
                    l2_cache_size = initial_stats.get('l2_cache_size', 0)
                    logger.info(f"{log_msg_header} 使用增强版缓存系统，初始状态: L1命中率={l1_hit_rate:.1%}, "
                               f"总命中率={total_hit_rate:.1%}, L2容量={l2_cache_size}")
                except Exception as e:
                    logger.info(f"{log_msg_header} 使用增强版缓存系统（统计信息暂不可用: {e}）")
            else:
                # 创建新的增强版缓存
                logger.info(f"{log_msg_header} 创建增强版缓存系统")
                from src.model.smart_cache import create_enhanced_cache
                self.bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)

        def _function_parallel(self, param_values: np.ndarray, param_names: List[str]) -> float:
            """并行版本的M-step目标函数"""
            start_time = time.time()
            log_msg_header = f"    [M-step Objective Call #{self.call_count}]"
            logger.info(f"{log_msg_header} Starting (PARALLEL MODE)...")
            
            # 解包参数
            params = _unpack_params(param_values, param_names, initial_params['n_choices'])
            
            # 根据参数选择并行化后端
            backend = self.m_step_backend
            # 【修复】当使用多进程后端时，不能传递包含锁的缓存对象。
            # 每个worker将创建自己的缓存实例。
            if backend in ['loky', 'multiprocessing']:
                logger.info(f"{log_msg_header} 使用'{backend}'后端，进程间隔离但稳定性更好。缓存将在每个worker中独立创建。")
                # 传递一个空的dict，避免序列化错误
                worker_bellman_cache = {}
            else:
                logger.info(f"{log_msg_header} 使用'{backend}'后端，worker共享主进程缓存（避免死锁+保留加速）")
                worker_bellman_cache = self.bellman_cache
            
            # 创建数据包供worker使用 - 传递bellman_cache供共享
            data_package = {
                'params': params,
                'observed_data': observed_data,
                'individual_posteriors': individual_posteriors,
                'individual_omega_lists': individual_omega_lists,
                'beta': beta,
                'transition_matrices': transition_matrices,
                'regions_df_np': regions_df,  # 注意：使用regions_df_np命名，与worker函数一致
                'distance_matrix': distance_matrix,
                'adjacency_matrix': adjacency_matrix,
                'prov_to_idx': prov_to_idx,
                'bellman_cache': worker_bellman_cache,  # 【关键】根据后端传递合适的缓存
                'shared_cache_dict': None  # 不使用multiprocessing共享字典
            }
            
            # 使用joblib进行并行处理
            from joblib import Parallel, delayed
            
            logger.info(f"{log_msg_header} Starting parallel processing with {self.parallel_config.n_jobs} jobs...")
            
            # 并行调用worker函数 - 使用threading后端实现缓存共享
            worker_results = Parallel(
                n_jobs=self.parallel_config.n_jobs,
                verbose=0,
                backend=backend,  # threading后端：所有worker在同一个进程，共享bellman_cache
                timeout=3600  # 【新增】设置1小时超时防止死锁
            )(
                delayed(_m_step_objective_worker)(
                    individual_id,
                    data_package
                ) for individual_id in unique_individuals
            )
            
            # 聚合结果和缓存统计
            total_weighted_log_lik = 0.0
            n_valid_computations = 0
            n_failed_computations = 0
            individual_log_liks = []
            total_cache_hits = 0
            total_cache_misses = 0
            
            for i_idx, result in enumerate(worker_results):
                if isinstance(result, dict):
                    # 新格式：包含贡献和缓存统计
                    individual_contribution = result['contribution']
                    total_cache_hits += result['cache_hits']
                    total_cache_misses += result['cache_misses']
                else:
                    # 旧格式：只有贡献值
                    individual_contribution = result
                    
                if np.isfinite(individual_contribution):
                    total_weighted_log_lik += individual_contribution
                    individual_log_liks.append(individual_contribution)
                    n_valid_computations += 1
                else:
                    n_failed_computations += 1
                
                # 进度日志（每logging_interval个个体）
                if (i_idx + 1) % max(1, len(unique_individuals) // 10) == 0 or i_idx == 0:
                    logger.info(f"{log_msg_header} Processed {i_idx+1}/{len(unique_individuals)} individuals in parallel")
            
            # 更新主进程的缓存统计
            self.cache_hits = total_cache_hits
            self.cache_misses = total_cache_misses
            
            # 计算负对数似然
            neg_ll = -total_weighted_log_lik
            duration = time.time() - start_time
            time_since_last = time.time() - self.last_call_end_time
            self.last_call_end_time = time.time()
            
            # 日志输出（保持与串行版本一致）
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            logger.info(f"{log_msg_header} Finished (PARALLEL). Duration: {duration:.2f}s, "
                        f"Since last: {time_since_last:.2f}s, NegLL: {neg_ll:.4f}")
            logger.info(f"{log_msg_header} Valid: {n_valid_computations}, Failed: {n_failed_computations}")
            if individual_log_liks:
                logger.info(f"{log_msg_header} Mean ind loglik: {np.mean(individual_log_liks):.4f}, "
                          f"Min: {np.min(individual_log_liks):.4f}, Max: {np.max(individual_log_liks):.4f}")
            
            if duration > 0 and n_valid_computations > 0:
                processing_rate = n_valid_computations / duration
                logger.info(f"{log_msg_header} Processing rate: {processing_rate:.1f} individuals/second, "
                            f"Cache hit rate: {cache_hit_rate:.1%} ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
            
            # 增强版缓存统计信息
            if hasattr(self.bellman_cache, 'get_stats'):
                try:
                    cache_stats = self.bellman_cache.get_stats()
                    logger.info(f"{log_msg_header} 增强缓存统计: L1命中率={cache_stats['l1_hit_rate']:.1%}, "
                               f"L2相似性命中率={cache_stats['l2_hit_rate']:.1%}, "
                               f"总命中率={cache_stats['total_hit_rate']:.1%}, "
                               f"相似性匹配={cache_stats['similarity_hits']}")
                except Exception as e:
                    logger.warning(f"{log_msg_header} 获取增强缓存统计失败: {e}")
            
            if n_valid_computations == 0:
                logger.error(f"{log_msg_header} All computations failed!")
                return 1e10
            
            return neg_ll

        def function(self, param_values: np.ndarray, param_names: List[str]) -> float:
            self.call_count += 1
            start_time = time.time()
            log_msg_header = f"    [M-step Objective Call #{self.call_count}]"
            logger.info(f"{log_msg_header} Starting...")
            
            # 记录开始处理的时间
            individual_start_time = time.time()

            # 解包参数
            params = _unpack_params(param_values, param_names, initial_params['n_choices'])

            # 根据并行配置选择处理模式
            if self.parallel_config.is_parallel_enabled() and self.parallel_config.n_jobs > 1:
                logger.info(f"{log_msg_header} Using PARALLEL processing with {self.parallel_config.n_jobs} jobs")
                return self._function_parallel(param_values, param_names)
            else:
                logger.info(f"{log_msg_header} Using SERIAL processing")

            # 提前定义统计变量
            total_weighted_log_lik = 0.0
            n_valid_computations = 0
            n_failed_computations = 0
            individual_log_liks = []
            all_weights = []
            all_log_liks = []

            weight_threshold = 1e-10

            # 主循环：遍历每个个体
            total_individuals = len(unique_individuals)
            processed_individuals = 0
            
            logging_interval = _calculate_logging_interval(total_individuals)

            for i_idx, individual_id in enumerate(unique_individuals):
                # 计算个体状态数量（用于热启动验证）
                individual_data = observed_data[observed_data['individual_id'] == individual_id]
                n_individual_states = _calculate_individual_state_space_size(individual_data)
                
                # 添加进度提示（每K个个体或开始/结束时）
                if (i_idx + 1) % logging_interval == 0 or i_idx == 0 or i_idx == total_individuals - 1:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if processed_individuals > 0:
                        avg_time_per_individual = elapsed / processed_individuals
                        remaining_individuals = total_individuals - processed_individuals
                        estimated_remaining = avg_time_per_individual * remaining_individuals
                        logger.info(f"{log_msg_header} Processing individual {i_idx+1}/{total_individuals} "
                                  f"(elapsed: {elapsed:.1f}s, est. remaining: {estimated_remaining:.1f}s)")
                    else:
                        logger.info(f"{log_msg_header} Processing individual {i_idx+1}/{total_individuals}")
                
                individual_data = observed_data[observed_data['individual_id'] == individual_id]
                posterior_matrix = individual_posteriors[individual_id]  # (n_omega, K)
                omega_list = individual_omega_lists[individual_id]

                individual_contribution = 0.0

                # 提取显著权重组合
                omega_indices, type_indices = np.where(posterior_matrix > weight_threshold)

                if len(omega_indices) == 0:
                    n_failed_computations += 1
                    continue  # 跳过该个体

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
                        
                        logger.debug(f"M-step 缓存查询: {individual_id}, ω={omega_idx}, 类型={k}, 缓存类型={type(self.bellman_cache)}")
                        
                        if hasattr(self.bellman_cache, 'get'):  # 增强版缓存
                            solution_shape = (n_individual_states, 3)  # 假设3个选择
                            logger.debug(f"M-step 使用增强版缓存查询: individual_id={individual_id}, shape={solution_shape}, 关键参数={ {k:v for k,v in type_params.items() if 'gamma' in k or 'sigma' in k} }")
                            converged_v_individual = self.bellman_cache.get(individual_id, type_params, int(k), solution_shape)
                            logger.debug(f"M-step 缓存查询结果: converged_v_individual_is_none={converged_v_individual is None}")
                            if converged_v_individual is not None:
                                converged = True
                                self.cache_hits += 1
                                logger.debug(f"M-step缓存命中: {individual_id}, ω={omega_idx}, 类型={k}")
                            else:
                                self.cache_misses += 1
                                logger.debug(f"M-step缓存未命中: {individual_id}, ω={omega_idx}, 类型={k}, 开始求解...")
                                
                                # 热启动由增强缓存内部处理，此处直接求解
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
                                    initial_v=None  # 增强缓存会自动处理热启动
                                )
                                
                                logger.debug(f"M-step Bellman求解完成: converged={converged}, v_is_none={converged_v_individual is None}")

                                if converged and converged_v_individual is not None:
                                    # 存入缓存
                                    self.bellman_cache.put(individual_id, type_params, int(k), converged_v_individual)
                                else:
                                    logger.warning(f"Bellman方程求解失败: {individual_id}, ω={omega_idx}, 类型={k}")
                                    n_failed_computations += 1
                                    continue
                        else:
                            # 向后兼容：旧版缓存逻辑（保持之前修复的正确逻辑）
                            logger.debug(f"M-step 使用旧版缓存查询: individual_id={individual_id}")
                            cache_key = (individual_id, tuple(sorted(type_params.items())), int(k))
                            converged_v_individual = self.bellman_cache.get(cache_key)
                            if converged_v_individual is not None:
                                converged = True
                                self.cache_hits += 1
                            else:
                                # 热启动：尝试使用相似参数的解作为初始值
                                self.hot_start_attempts += 1
                                # 保持之前修复的形状匹配逻辑（吸取之前的教训）
                                if hasattr(self.bellman_cache, 'cache') and len(self.bellman_cache.cache) > 0:
                                    try:
                                        for key, cached_solution in self.bellman_cache.cache.items():
                                            if isinstance(cached_solution, np.ndarray) and cached_solution.shape[0] == n_individual_states:
                                                initial_v = cached_solution
                                                self.hot_start_successes += 1
                                                break
                                    except Exception:
                                        pass
                                else:
                                    # 普通dict
                                    for key in reversed(list(self.bellman_cache.keys())):
                                        cached_solution = self.bellman_cache[key]
                                        if isinstance(cached_solution, np.ndarray) and cached_solution.shape[0] == n_individual_states:
                                            initial_v = cached_solution
                                            self.hot_start_successes += 1
                                            break
                            
                            # 求解 Bellman 方程
                            logger.debug(f"M-step 开始求解Bellman方程: {individual_id}, ω={omega_idx}, 类型={k}")
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
                            
                            logger.debug(f"M-step Bellman方程求解完成: converged={converged}, v_is_none={converged_v_individual is None}")
                            
                            # 检查Bellman方程是否成功求解
                            if converged and converged_v_individual is not None:
                                # 存储到增强版缓存系统或向后兼容
                                if hasattr(self.bellman_cache, 'put'):  # 增强版缓存
                                    # 增强版缓存使用智能存储接口
                                    self.bellman_cache.put(individual_id, type_params, int(k), converged_v_individual)
                                else:  # 旧版缓存
                                    # 为旧版缓存创建传统键
                                    cache_key = (individual_id, tuple(sorted(type_params.items())), int(k))
                                    self.bellman_cache[cache_key] = converged_v_individual
                                self.cache_misses += 1
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

                        # 首次调用时收集调试信息
                        if self.call_count == 1:
                            all_weights.append(weight)
                            all_log_liks.append(individual_log_lik)

                    except Exception as e:
                        logger.debug(f"{log_msg_header} Error for ind={individual_id}, ω_idx={omega_idx}, τ={k}: {e}")
                        n_failed_computations += 1
                        continue  # 跳过该组合

                # 累加该个体总贡献
                total_weighted_log_lik += individual_contribution
                individual_log_liks.append(individual_contribution)
                processed_individuals += 1

            # 计算负对数似然
            neg_ll = -total_weighted_log_lik
            duration = time.time() - start_time
            time_since_last = time.time() - self.last_call_end_time
            self.last_call_end_time = time.time()

            # 日志输出
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            logger.info(f"{log_msg_header} Finished. Duration: {duration:.2f}s, "
                        f"Since last: {time_since_last:.2f}s, NegLL: {neg_ll:.4f}")
            logger.info(f"{log_msg_header} Valid: {n_valid_computations}, Failed: {n_failed_computations}")
            if individual_log_liks:
                logger.info(f"{log_msg_header} Mean ind loglik: {np.mean(individual_log_liks):.4f}, "
                          f"Min: {np.min(individual_log_liks):.4f}, Max: {np.max(individual_log_liks):.4f}")
            # 安全计算处理速率，避免除零错误
            if duration > 0 and processed_individuals > 0:
                processing_rate = processed_individuals / duration
                logger.info(f"{log_msg_header} Processing rate: {processing_rate:.1f} individuals/second, "
                            f"Cache hit rate: {cache_hit_rate:.1%} ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
            elif duration > 0:
                logger.info(f"{log_msg_header} Processing rate: 0.0 individuals/second (no individuals processed), "
                            f"Cache hit rate: {cache_hit_rate:.1%} ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
            else:
                logger.info(f"{log_msg_header} Processing rate: N/A (processing time too short), "
                            f"Cache hit rate: {cache_hit_rate:.1%} ({self.cache_hits}/{self.cache_misses})")
            
            # 增强版缓存统计信息
            if hasattr(self.bellman_cache, 'get_stats'):
                try:
                    cache_stats = self.bellman_cache.get_stats()
                    # 只在第一次调用或每10次调用时显示详细统计
                    if self.call_count <= 2 or self.call_count % 10 == 0:
                        logger.info(f"{log_msg_header} 增强缓存统计: L1命中率={cache_stats['l1_hit_rate']:.1%}, "
                                   f"L2相似性命中率={cache_stats['l2_hit_rate']:.1%}, "
                                   f"总命中率={cache_stats['total_hit_rate']:.1%}, "
                                   f"相似性匹配={cache_stats['similarity_hits']}")
                    else:
                        logger.debug(f"{log_msg_header} 缓存命中率: {cache_stats['total_hit_rate']:.1%}")
                except Exception as e:
                    logger.warning(f"{log_msg_header} 获取增强缓存统计失败: {e}")
                    # 使用基本统计信息
                    if hasattr(self.bellman_cache, 'stats'):
                        basic_stats = self.bellman_cache.stats
                        logger.info(f"{log_msg_header} 基础缓存统计: {basic_stats}")
            else:
                # 向后兼容：旧版缓存统计
                cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                logger.info(f"{log_msg_header} Cache hit rate: {cache_hit_rate:.1%} ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
            
            # 首次调用时添加热启动统计
            if self.call_count == 1:
                logger.info(f"{log_msg_header} Hot-start enabled: Using previous solutions as initial values for Bellman iteration")
            
            # 添加热启动统计（增强版缓存会自动处理）
            if hasattr(self.bellman_cache, 'get_stats'):
                try:
                    # 增强版缓存：相似性匹配就是热启动
                    cache_stats = self.bellman_cache.get_stats()
                    similarity_rate = cache_stats.get('l2_hit_rate', 0) if cache_stats.get('total_requests', 0) > 0 else 0
                    logger.info(f"{log_msg_header} 相似性匹配率: {similarity_rate:.1%} (L2缓存)")
                except Exception as e:
                    logger.debug(f"{log_msg_header} 获取热启动统计失败: {e}")
            else:
                # 向后兼容：旧版热启动统计
                hot_start_rate = self.hot_start_successes / self.hot_start_attempts if self.hot_start_attempts > 0 else 0
                logger.info(f"{log_msg_header} Hot-start success rate: {hot_start_rate:.1%} ({self.hot_start_successes}/{self.hot_start_attempts})")
            
            # 缓存容量和性能统计
            if hasattr(self.bellman_cache, 'l1_cache') and hasattr(self.bellman_cache.l1_cache, 'get_stats'):
                try:
                    l1_stats = self.bellman_cache.l1_cache.get_stats()
                    logger.info(f"{log_msg_header} L1缓存: {l1_stats['size']}/{l1_stats['max_size']}条目, "
                               f"{l1_stats['memory_mb']:.1f}/{l1_stats['max_memory_mb']:.1f}MB, "
                               f"淘汰: {l1_stats['eviction_count']}次")
                except Exception as e:
                    logger.debug(f"{log_msg_header} 获取L1缓存统计失败: {e}")
            elif hasattr(self.bellman_cache, 'get_stats'):
                try:
                    cache_stats = self.bellman_cache.get_stats()
                    size = cache_stats.get('l1_stats', {}).get('size', 'N/A')
                    max_size = cache_stats.get('l1_stats', {}).get('max_size', 'N/A')
                    logger.info(f"{log_msg_header} 缓存容量: {size}/{max_size}条目")
                except Exception as e:
                    logger.debug(f"{log_msg_header} 获取缓存统计失败: {e}")

            if n_valid_computations == 0:
                logger.error(f"{log_msg_header} All computations failed!")
                return 1e10

            # 首次调用输出权重/似然统计
            if self.call_count == 1 and all_weights:
                logger.debug(f"{log_msg_header} Weight stats: "
                             f"mean={np.mean(all_weights):.2e}, std={np.std(all_weights):.2e}, "
                             f"min={np.min(all_weights):.2e}, max={np.max(all_weights):.2e}")
                logger.debug(f"{log_msg_header} Loglik stats: "
                             f"mean={np.mean(all_log_liks):.2f}, std={np.std(all_log_liks):.2f}, "
                             f"min={np.min(all_log_liks):.2f}, max={np.max(all_log_liks):.2f}")

            return neg_ll

    # --- 优化执行 ---
    logger.info(f"  [Debug] 传递给Objective的m_step_backend参数: {m_step_backend}")
    objective = Objective(shared_bellman_cache=bellman_cache, parallel_config=parallel_config, m_step_backend=m_step_backend)
    initial_param_values, param_names = _pack_params(initial_params)
    param_bounds = config.get_parameter_bounds(param_names)

    logger.info(f"  Optimizing {len(param_names)} parameters with L-BFGS-B (maxiter={lbfgsb_maxiter})")
    initial_neg_ll = objective.function(initial_param_values, param_names)
    logger.info(f"  Initial objective: {initial_neg_ll:.4f}")
    objective.call_count = 0  # 重置计数

    # 使用传入的容差参数，如果没有提供则使用默认值
    gtol = lbfgsb_gtol if lbfgsb_gtol is not None else 1e-5
    ftol = lbfgsb_ftol if lbfgsb_ftol is not None else 1e-6
    
    logger.info(f"  L-BFGS-B convergence tolerances: gtol={gtol}, ftol={ftol}")

    try:
        result = minimize(
            objective.function,
            x0=initial_param_values,
            args=(param_names,),
            method='L-BFGS-B',
            bounds=param_bounds,
            options={
                'disp': False,
                'maxiter': lbfgsb_maxiter,
                'gtol': gtol,
                'ftol': ftol,
                'eps': 1e-8,
            }
        )

        final_neg_ll = result.fun
        logger.info(f"  Optimization finished: success={result.success}, nit={result.nit}, "
                    f"ΔNegLL={initial_neg_ll - final_neg_ll:+.4f}")

        if result.success or result.nit > 0:
            updated_params = _unpack_params(result.x, param_names, initial_params['n_choices'])
        else:
            logger.warning("  Optimization failed to converge, keeping initial params")
            updated_params = initial_params

    except Exception as e:
        logger.error(f"  Optimization error: {e}", exc_info=True)
        updated_params = initial_params

    # --- 更新类型概率 π_k ---
    updated_pi_k = np.zeros(K)
    for individual_id in unique_individuals:
        posterior_matrix = individual_posteriors[individual_id]
        marginal_type_prob = np.sum(posterior_matrix, axis=0)  # 边缘化ω
        updated_pi_k += marginal_type_prob
    updated_pi_k /= N
    updated_pi_k = np.clip(updated_pi_k, 0.01, None)
    updated_pi_k /= updated_pi_k.sum()

    logger.info(f"  M-step completed. π_k = {np.round(updated_pi_k, 4)}")
    return updated_params, updated_pi_k

def _prepare_numpy_region_data(regions_df: pd.DataFrame, prov_to_idx: Dict[int, int]) -> Dict[str, np.ndarray]:
    """
    Converts the regions DataFrame to a dictionary of NumPy arrays, ordered by prov_to_idx.
    """
    # Ensure the DataFrame is sorted according to the indexer to guarantee correct alignment
    idx_to_prov = {v: k for k, v in prov_to_idx.items()}
    sorted_provs = [idx_to_prov[i] for i in range(len(prov_to_idx))]
    
    regions_df_sorted = regions_df.set_index('provcd').loc[sorted_provs].reset_index()

    regions_df_np = {col: regions_df_sorted[col].to_numpy() for col in regions_df_sorted.columns}
    return regions_df_np

def run_em_algorithm_with_omega(
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    n_types: int,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    support_generator: 'DiscreteSupportGenerator',
    prov_to_idx: Dict[int, int],
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    n_choices: int = 31,
    initial_params: Dict[str, Any] = None,
    initial_pi_k: np.ndarray = None,
    max_omega_per_individual: int = 1000,
    use_simplified_omega: bool = True,
    lbfgsb_maxiter: int = 15,
    parallel_config: Optional['ParallelConfig'] = None,  # 新增并行配置参数
    lbfgsb_gtol: float = None,  # 临时调整L-BFGS-B梯度容差
    lbfgsb_ftol: float = None,  # 临时调整L-BFGS-B函数值容差
    memory_safe_mode: bool = False,  # 内存安全模式
    progress_tracker: Optional['EstimationProgressTracker'] = None,  # 进度跟踪器
    start_iteration: int = 0,  # 起始迭代次数（用于恢复）
    m_step_backend: str = 'threading',  # M步并行化后端 ('threading', 'loky', 'multiprocessing')
) -> Dict[str, Any]:
    """
    EM-NFXP算法主循环（带离散支撑点ω）
    
    这是run_em_algorithm()的扩展版本，集成了离散支撑点枚举。
    EM-NFXP算法主循环（带离散支撑点ω）
    
    参数:
    ----
    # ... 其他参数 ...
    m_step_backend : str, optional
        M步并行化后端类型，可选值: 'threading', 'loky', 'multiprocessing'，默认'threading'
        
    参数:
    ----
    observed_data : pd.DataFrame
        观测数据
    state_space : pd.DataFrame
        状态空间
    transition_matrices : Dict[str, np.ndarray]
        转移矩阵
    beta : float
        贴现因子
    n_types : int
        类型数量K
    regions_df : pd.DataFrame
        地区数据
    distance_matrix : np.ndarray
        距离矩阵
    adjacency_matrix : np.ndarray
        邻接矩阵
    support_generator : DiscreteSupportGenerator
        离散支撑点生成器
    prov_to_idx : Dict[int, int]
        省份ID到矩阵索引的映射
    max_iterations : int
        EM最大迭代次数
    tolerance : float
        收敛容忍度
    n_choices : int
        选择数量
    initial_params : Dict[str, Any], optional
        初始参数
    initial_pi_k : np.ndarray, optional
        初始类型概率
    max_omega_per_individual : int
        每个个体的最大ω组合数
    use_simplified_omega : bool
        是否使用简化ω策略
    lbfgsb_maxiter : int
        M-step中L-BFGS-B最大迭代次数
    progress_tracker : EstimationProgressTracker, optional
        进度跟踪器 用于定期保存EM算法中间状态
    start_iteration : int, optional
        起始迭代次数，用于从中间状态恢复EM算法
    m_step_backend : str, optional
        M步并行化后端类型，可选值: 'threading', 'loky', 'multiprocessing'，默认'threading'

    返回:
    ----
    Dict包含:
        - 'structural_params': 估计的结构参数
        - 'type_probabilities': 估计的类型概率
        - 'final_log_likelihood': 最终对数似然
        - 'converged': 是否收敛
        - 'n_iterations': 迭代次数
        - 'individual_posteriors': 个体后验概率
    """
    # 添加调试信息
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("EM-NFXP Algorithm with Discrete Support Points (ω)")
    logger.info("="*80)
    
    # 添加调试信息
    logger.info(f"[Debug] run_em_algorithm_with_omega接收到的m_step_backend参数: {m_step_backend}")

    # --- OPTIMIZATION: Pre-convert region data to NumPy ---
    logger.info("Pre-converting regional data to NumPy for performance...")
    regions_df_np = _prepare_numpy_region_data(regions_df, prov_to_idx)
    logger.info("Conversion complete.")

    # 初始化
    if initial_params is None:
        from src.config.model_config import ModelConfig
        config = ModelConfig()
        initial_params = config.get_initial_params(use_type_specific=True)
        initial_params['n_choices'] = n_choices

    if initial_pi_k is None:
        initial_pi_k = np.full(n_types, 1.0 / n_types)

    current_params = initial_params.copy()
    current_pi_k = initial_pi_k.copy()

    # 缓存预热：使用初始参数预热缓存
    logger.info("预热缓存...")
    for k in range(n_types):
        cache_key = _make_cache_key(current_params, k)
        logger.debug(f"预热类型 {k} 的缓存")

    prev_log_likelihood = -np.inf
    converged = False

    # EM迭代
    for iteration in range(start_iteration, max_iterations):
        actual_iteration = iteration + 1
        logger.info(f"\n{'='*80}")
        logger.info(f"EM Iteration {actual_iteration}/{max_iterations}")
        if start_iteration > 0 and iteration == start_iteration:
            logger.info(f"(从保存的第{start_iteration}次迭代继续)")
        logger.info(f"{'='*80}")

        # E-step
        logger.info("\n[E-step with ω]")
        # 创建增强版缓存系统 - 为这次EM迭代创建共享缓存，具有内存管理功能
        from src.model.smart_cache import create_enhanced_cache
        bellman_cache = create_enhanced_cache(capacity=2000, memory_limit_mb=2000)  # EM迭代专用增强缓存
        
        # 设置并行配置（如果未提供则使用默认配置）
        if parallel_config is None:
            parallel_config = ParallelConfig(n_jobs=1)  # 默认串行处理
        
        individual_posteriors, log_likelihood_matrix = e_step_with_omega(
            bellman_cache=bellman_cache,
            params=current_params,
            pi_k=current_pi_k,
            observed_data=observed_data,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=beta,
            regions_df=regions_df_np,  # Pass NumPy version
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_generator,
            n_types=n_types,
            prov_to_idx=prov_to_idx,
            max_omega_per_individual=max_omega_per_individual,
            use_simplified_omega=use_simplified_omega,
            parallel_config=parallel_config,  # 传递并行配置
            memory_safe_mode=memory_safe_mode  # 传递内存安全模式
        )

        # 计算当前对数似然
        current_log_likelihood = np.sum(np.log(np.sum(
            log_likelihood_matrix * current_pi_k[np.newaxis, :],
            axis=1
        ) + 1e-300))

        logger.info(f"\n  Current log-likelihood: {current_log_likelihood:.4f}")
        
        # E步缓存性能汇总（只在第一次或每5次迭代时显示）
        if iteration == 0 or (iteration + 1) % 5 == 0:
            if hasattr(bellman_cache, 'get_stats'):
                try:
                    e_step_stats = bellman_cache.get_stats()
                    logger.info(f"  E步缓存性能: 总命中率={e_step_stats['total_hit_rate']:.1%}, "
                               f"L1命中率={e_step_stats['l1_hit_rate']:.1%}, "
                               f"L2相似性命中率={e_step_stats['l2_hit_rate']:.1%}")
                except Exception as e:
                    logger.debug(f"  获取E步缓存统计失败: {e}")

        # 检查收敛
        if iteration > 0:
            ll_change = current_log_likelihood - prev_log_likelihood
            logger.info(f"  Log-likelihood change: {ll_change:.6f}")
            
            # **诊断增强**: 添加相对变化
            if abs(prev_log_likelihood) > 1e-10:
                relative_change = abs(ll_change / prev_log_likelihood)
                logger.info(f"  Relative log-likelihood change: {relative_change:.6f}")
            
            if abs(ll_change) < tolerance:
                logger.info(f"\n  ✓ Converged! (Δ log-likelihood < {tolerance})")
                converged = True
                break
        
        # 保存EM算法进度（如果提供了进度跟踪器）
        if progress_tracker is not None:
            em_state = {
                'current_params': current_params.copy(),
                'current_pi_k': current_pi_k.copy(),
                'current_log_likelihood': current_log_likelihood,
                'iteration': actual_iteration,
                'converged': converged
            }
            progress_tracker.save_em_progress(em_state)

        prev_log_likelihood = current_log_likelihood

        # M-step
        logger.info("\n[M-step with ω]")
        current_params, current_pi_k = m_step_with_omega(
            individual_posteriors=individual_posteriors,
            initial_params=current_params,
            observed_data=observed_data,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=beta,
            regions_df=regions_df_np,  # Pass NumPy version
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_generator,
            n_types=n_types,
            prov_to_idx=prov_to_idx,
            max_omega_per_individual=max_omega_per_individual,
            lbfgsb_maxiter=lbfgsb_maxiter,
            bellman_cache=bellman_cache,  # 传递E步的缓存给M步
            parallel_config=parallel_config,  # 传递并行配置给M步
            lbfgsb_gtol=lbfgsb_gtol,  # 临时收敛容差控制
            lbfgsb_ftol=lbfgsb_ftol,  # 临时收敛容差控制
            m_step_backend=m_step_backend
        )

    # 汇总结果
    logger.info("\n" + "="*80)
    total_iterations = iteration + 1 if 'iteration' in locals() else start_iteration
    if converged:
        logger.info(f"✓ EM Algorithm Converged after {total_iterations} iterations")
    else:
        logger.info(f"⚠ EM Algorithm reached max iterations ({max_iterations})")
    logger.info(f"Final log-likelihood: {prev_log_likelihood:.4f}")
    logger.info(f"Final type probabilities: {current_pi_k}")
    logger.info("="*80)
    
    # 保存最终EM算法状态
    if progress_tracker is not None:
        final_em_state = {
            'current_params': current_params.copy(),
            'current_pi_k': current_pi_k.copy(),
            'current_log_likelihood': prev_log_likelihood,
            'iteration': total_iterations,
            'converged': converged
        }
        progress_tracker.save_em_progress(final_em_state, force=True)  # 强制保存最终状态

    results = {
        'structural_params': current_params,
        'type_probabilities': current_pi_k,
        'final_log_likelihood': prev_log_likelihood,
        'converged': converged,
        'n_iterations': iteration + 1,
        'individual_posteriors': individual_posteriors,
        'posterior_probs': log_likelihood_matrix  # 为兼容性保留
    }

    return results