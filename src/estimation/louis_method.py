"""
该模块实现Louis (1982)方法计算EM算法估计参数的标准误。

Louis方法利用EM算法的副产品（完全数据对数似然的梯度和Hessian的期望）
来计算观测信息矩阵，从而避免了计算量巨大的Bootstrap方法。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
from numdifftools import Hessian, Gradient
from scipy.stats import norm

from src.model.likelihood import calculate_likelihood_from_v_individual
from src.model.bellman import solve_bellman_equation_individual
from src.estimation.em_with_omega import _pack_params, _unpack_params # 导入参数打包解包工具

logger = logging.getLogger(__name__)

def _individual_log_likelihood_for_louis(
    packed_params: np.ndarray,
    param_names: List[str],
    n_choices: int,
    individual_data: pd.DataFrame,
    agent_type: int,
    omega_values: Dict[str, float],
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    prov_to_idx: Dict[int, int]
) -> float:
    """
    计算单个个体在给定类型和omega下的对数似然。
    此函数用于数值微分以获取梯度和Hessian。

    参数:
    ----
    packed_params : np.ndarray
        打包后的参数数组
    param_names : List[str]
        参数名称列表
    n_choices : int
        选择数量
    individual_data : pd.DataFrame
        单个个体的观测数据
    agent_type : int
        个体类型 (τ)
    omega_values : Dict[str, float]
        当前omega组合的eta和sigma值
    beta : float
        贴现因子
    transition_matrices : Dict[str, np.ndarray]
        转移矩阵
    regions_df : Dict[str, np.ndarray]
        地区数据 (NumPy版本)
    distance_matrix : np.ndarray
        距离矩阵
    adjacency_matrix : np.ndarray
        邻接矩阵
    prov_to_idx : Dict[int, int]
        省份ID到矩阵索引的映射

    返回:
    ----
    float
        该个体在该类型和omega组合下的对数似然
    """
    import time
    start_time = time.time()
    
    params = _unpack_params(packed_params, param_names, n_choices)

    # 提取omega相关参数
    eta_i = omega_values['eta']
    sigma_epsilon = omega_values['sigma']

    # 构建type-specific参数
    type_params = params.copy()
    if f'gamma_0_type_{agent_type}' in params:
        type_params['gamma_0'] = params[f'gamma_0_type_{agent_type}']
    type_params['sigma_epsilon'] = sigma_epsilon # 将sigma_epsilon作为参数传入

    try:
        # 求解Bellman方程
        converged_v, _ = solve_bellman_equation_individual(
            utility_function=None, # 效用函数在内部构建
            individual_data=individual_data,
            params=type_params,
            agent_type=agent_type,
            beta=beta,
            transition_matrices=transition_matrices,
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            verbose=False,
            prov_to_idx=prov_to_idx
        )

        # 计算似然（包含工资似然）
        log_lik_obs = calculate_likelihood_from_v_individual(
            converged_v_individual=converged_v,
            params=type_params,
            individual_data=individual_data,
            agent_type=agent_type,
            beta=beta,
            transition_matrices=transition_matrices,
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            prov_to_idx=prov_to_idx
        )
        total_likelihood = np.sum(log_lik_obs)
        
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # 如果计算时间超过1秒，记录警告
            logger.debug(f"个体似然计算耗时: {elapsed:.2f}s, 类型: {agent_type}, omega: {omega_values}, 似然: {total_likelihood:.4f}")
        
        return total_likelihood
    except Exception as e:
        elapsed = time.time() - start_time
        logger.debug(f"个体似然计算失败: {e}, 耗时: {elapsed:.2f}s, 类型: {agent_type}, omega: {omega_values}")
        return -1e10 # 返回一个非常小的数表示计算失败

def louis_method_standard_errors(
    estimated_params: Dict[str, Any],
    type_probabilities: np.ndarray,
    individual_posteriors: Dict[Any, np.ndarray],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    support_generator: Any, # DiscreteSupportGenerator
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 1000,
    use_simplified_omega: bool = True,
    h_step: float = 1e-4 # 数值微分步长
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    使用Louis (1982)方法计算EM算法估计参数的标准误、t统计量和p值。

    参数:
    ----
    estimated_params : Dict[str, Any]
        EM算法估计得到的最终参数
    type_probabilities : np.ndarray
        EM算法估计得到的最终类型概率 π_k
    individual_posteriors : Dict[Any, np.ndarray]
        E-step计算得到的个体后验概率 p(τ, ω | D_i)
    observed_data : pd.DataFrame
        观测数据
    state_space : pd.DataFrame
        状态空间
    transition_matrices : Dict[str, np.ndarray]
        转移矩阵
    beta : float
        贴现因子
    regions_df : Dict[str, np.ndarray]
        地区数据 (NumPy版本)
    distance_matrix : np.ndarray
        距离矩阵
    adjacency_matrix : np.ndarray
        邻接矩阵
    support_generator : Any
        离散支撑点生成器 (DiscreteSupportGenerator)
    n_types : int
        类型数量K
    prov_to_idx : Dict[int, int]
        省份ID到矩阵索引的映射
    max_omega_per_individual : int
        每个个体的最大ω组合数
    use_simplified_omega : bool
        是否使用简化ω策略
    h_step : float
        数值微分步长

    返回:
    ----
    Tuple[Dict, Dict, Dict]: (标准误字典, t统计量字典, p值字典)
    """
    logger.info("\n开始使用Louis (1982)方法计算标准误...")
    logger.info(f"  参数配置: h_step={h_step}, max_omega_per_individual={max_omega_per_individual}")

    # 1. 准备参数
    packed_estimated_params, param_names = _pack_params(estimated_params)
    n_params = len(packed_estimated_params)
    n_choices = estimated_params['n_choices']
    unique_individuals = observed_data['individual_id'].unique()
    N = len(unique_individuals)
    
    logger.info(f"  样本信息: {N} 个个体, {n_params} 个参数, {n_types} 种类型")

    # 2. 初始化信息矩阵和Score矩阵
    expected_complete_information = np.zeros((n_params, n_params))
    sum_of_individual_scores_outer_product = np.zeros((n_params, n_params))

    # 预枚举所有个体的ω
    from src.model.discrete_support import SimplifiedOmegaEnumerator
    enumerator = SimplifiedOmegaEnumerator(support_generator)
    individual_omega_lists = {}
    individual_omega_probs = {}

    logger.info("  预枚举所有个体的omega...")
    start_time = pd.Timestamp.now()
    for i, individual_id in enumerate(unique_individuals):
        if (i + 1) % max(1, N // 10) == 0 or i < 5:  # 显示前5个和每10%的进度
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"    预枚举进度: {i+1}/{N} 个体, 用时: {elapsed:.1f}s")
        
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=max_omega_per_individual
        )
        individual_omega_lists[individual_id] = omega_list
        individual_omega_probs[individual_id] = omega_probs
    
    enum_time = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"  预枚举完成: 用时 {enum_time:.1f}s, 平均每个个体 {enum_time/N:.2f}s")

    # 3. 遍历每个个体，计算其对信息矩阵和Score矩阵的贡献
    logger.info(f"  开始计算信息矩阵和Score矩阵...")
    total_omega_combinations = sum(len(individual_omega_lists[ind_id]) for ind_id in unique_individuals)
    logger.info(f"  总omega组合数: {total_omega_combinations}, 平均每个个体: {total_omega_combinations/N:.1f}")
    
    start_time = pd.Timestamp.now()
    last_log_time = start_time
    
    for i_idx, individual_id in enumerate(unique_individuals):
        # 更频繁的日志输出
        current_time = pd.Timestamp.now()
        if (i_idx + 1) % 1 == 0 or (current_time - last_log_time).total_seconds() > 10:  # 每1个个体或每10秒输出一次
            elapsed = (current_time - start_time).total_seconds()
            progress = (i_idx + 1) / N
            eta = elapsed / progress - elapsed if progress > 0 else 0
            logger.info(f"    处理个体 {i_idx+1}/{N} ({progress*100:.1f}%), 用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
            last_log_time = current_time

        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        posterior_matrix = individual_posteriors[individual_id] # (n_omega, K)
        omega_list = individual_omega_lists[individual_id]

        individual_score_sum = np.zeros(n_params) # S_i

        # 遍历每个(ω, τ)组合
        for omega_idx, omega in enumerate(omega_list):
            for k in range(n_types):
                weight = posterior_matrix[omega_idx, k] # p(τ, ω | D_i)

                if weight < 1e-10: # 忽略权重过小的组合
                    continue

                # 定义一个lambda函数，用于传递给numdifftools
                # 这里的func只接受一个参数：packed_params
                func_to_diff = lambda p: _individual_log_likelihood_for_louis(
                    p, param_names, n_choices, individual_data, k, omega, beta,
                    transition_matrices, regions_df, distance_matrix, adjacency_matrix, prov_to_idx
                )

                # 计算梯度 (Score)
                try:
                    grad_calculator = Gradient(func_to_diff, step=h_step)
                    score_vector = grad_calculator(packed_estimated_params)
                    if not np.all(np.isfinite(score_vector)):
                        logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: Score包含非有限值，跳过。")
                        continue
                except Exception as e:
                    logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: 计算Score失败: {e}")
                    continue

                # 计算Hessian
                try:
                    hess_calculator = Hessian(func_to_diff, step=h_step)
                    hessian_matrix = hess_calculator(packed_estimated_params)
                    if not np.all(np.isfinite(hessian_matrix)):
                        logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: Hessian包含非有限值，跳过。")
                        continue
                except Exception as e:
                    logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: 计算Hessian失败: {e}")
                    continue

                # 累加到期望完全信息矩阵
                expected_complete_information -= weight * hessian_matrix

                # 累加到个体Score和
                individual_score_sum += weight * score_vector

        # 累加到Score协方差矩阵
        sum_of_individual_scores_outer_product += np.outer(individual_score_sum, individual_score_sum)
        
        # 记录当前个体的处理统计
        if (i_idx + 1) % max(1, N // 20) == 0:  # 每5%输出一次统计
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"    进度: {i_idx+1}/{N} 个体, 总用时: {elapsed:.1f}s")

    # 计算完成统计
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"  信息矩阵计算完成: 总用时 {total_time:.1f}s, 平均每个个体 {total_time/N:.2f}s")

    # 4. 计算观测信息矩阵
    observed_information = expected_complete_information - sum_of_individual_scores_outer_product

    # 5. 计算协方差矩阵
    try:
        cov_matrix = np.linalg.inv(observed_information)
    except np.linalg.LinAlgError:
        logger.warning("观测信息矩阵奇异，使用伪逆计算协方差矩阵。")
        cov_matrix = np.linalg.pinv(observed_information)

    # 6. 提取标准误、t统计量和p值
    std_errors = {}
    t_stats = {}
    p_values = {}

    for i, name in enumerate(param_names):
        if i < cov_matrix.shape[0] and i < cov_matrix.shape[1]:
            std_err = np.sqrt(max(0, cov_matrix[i, i]))
            std_errors[name] = std_err

            if std_err > 1e-10: # 避免除以零
                t_stat = packed_estimated_params[i] / std_err
                t_stats[name] = t_stat
                p_values[name] = 2 * (1 - norm.cdf(abs(t_stat)))
            else:
                t_stats[name] = np.nan
                p_values[name] = np.nan
        else:
            std_errors[name] = np.nan
            t_stats[name] = np.nan
            p_values[name] = np.nan

    logger.info("Louis (1982)方法标准误计算完成。")
    logger.info(f"  结果概览: {len(std_errors)} 个参数, 信息矩阵条件数: {np.linalg.cond(observed_information):.2e}")
    return std_errors, t_stats, p_values
