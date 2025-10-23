"""
扩展的EM算法E-step和M-step

集成离散支撑点，计算p(τ, ω|D)后验概率
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
import time
from joblib import Parallel, delayed

from src.model.discrete_support import (
    DiscreteSupportGenerator,
    SimplifiedOmegaEnumerator,
    extract_omega_values_for_state
)
from src.model.likelihood import (
    solve_bellman_for_params,
    calculate_likelihood_from_v,
    calculate_likelihood_from_v_individual
)
from src.model.bellman import solve_bellman_equation_individual
from src.model.wage_equation import calculate_predicted_wage, calculate_reference_wage


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




def e_step_with_omega(
    params: Dict[str, Any],
    pi_k: np.ndarray,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    support_generator: DiscreteSupportGenerator,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 1000,
    use_simplified_omega: bool = True
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

    # 初始化结果容器
    individual_posteriors = {}
    log_likelihood_matrix = np.zeros((N, K))

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

    # 并行生成所有个体的ω
    logger.info(f"  Enumerating ω for {N} individuals (parallel)...")
    omega_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(enumerate_omega_for_individual_wrapper)(ind_id)
        for ind_id in unique_individuals
    )

    # 将结果组织到字典中
    individual_omega_dict = {
        ind_id: (omega_list, omega_probs)
        for ind_id, omega_list, omega_probs in omega_results
    }

    for i_idx, individual_id in enumerate(unique_individuals):
        if (i_idx + 1) % 100 == 0:
            logger.info(f"    Processing individual {i_idx+1}/{N}")

        # 1. 提取该个体的数据和ω
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        omega_list, omega_probs = individual_omega_dict[individual_id]

        n_omega = len(omega_list)

        # 3. 计算每个(τ, ω)组合的似然
        # 结构: posterior_matrix[omega_idx, type_idx]
        log_lik_matrix = np.zeros((n_omega, K))

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

                    # 求解Bellman方程（此处可能需要传入ω相关值到效用函数）
                    # 简化实现：先不传ω到Bellman求解中
                    converged_v, _ = solve_bellman_equation_individual(
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
                        prov_to_idx=prov_to_idx
                    )

                    # 计算似然（包含工资似然）
                    log_lik_obs = calculate_likelihood_from_v_individual(
                        converged_v_individual=converged_v,
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
                    log_lik_matrix[omega_idx, k] = -1e10

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

        # 存储
        individual_posteriors[individual_id] = joint_probs

        # 5. 边缘化ω以获得p(τ|D_i)用于更新π_k和计算似然
        marginal_type_likelihood = np.sum(np.exp(log_lik_matrix), axis=0)
        log_likelihood_matrix[i_idx, :] = marginal_type_likelihood

    logger.info(f"  [E-step with ω] Completed.")

    return individual_posteriors, log_likelihood_matrix


def aggregate_omega_posteriors_for_parameter_update(
    individual_posteriors: Dict[Any, np.ndarray],
    observed_data: pd.DataFrame,
    support_generator: DiscreteSupportGenerator
) -> Dict[str, np.ndarray]:
    """
    汇总ω后验分布，用于参数更新

    从individual_posteriors中提取有用的统计量，
    例如E[η_i | D_i], E[σ_ε | D_i]等

    参数:
    ----
    individual_posteriors : Dict
        每个个体的后验概率
    observed_data : pd.DataFrame
        观测数据
    support_generator : DiscreteSupportGenerator
        支撑点生成器

    返回:
    ----
    Dict with aggregated statistics
    """
    logger = logging.getLogger()
    logger.info("  Aggregating omega posterior statistics...")

    # TODO: 实现后验期望值计算
    # 例如: E[η_i | D_i] = Σ_ω η(ω) · p(ω | D_i)

    aggregated_stats = {
        'eta_posterior_mean': [],
        'sigma_posterior_mean': []
    }

    logger.info("  Omega aggregation completed.")

    return aggregated_stats


def m_step_with_omega(
    individual_posteriors: Dict[Any, np.ndarray],
    initial_params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    support_generator: DiscreteSupportGenerator,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 1000,
    lbfgsb_maxiter: int = 15
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    扩展的M-step：对ω进行加权求和来更新参数

    根据论文公式(1132-1138行):
        Θ^(k+1) = argmax_Θ Σ_i Σ_τ Σ_ω p(τ,ω|D_i,Θ^(k)) · ln L(D_i|τ,ω,Θ)

    参数:
    ----
    individual_posteriors : Dict[individual_id, np.ndarray]
        E-step计算出的后验概率，每个个体是一个(n_omega, K)矩阵
    initial_params : Dict[str, Any]
        当前参数值
    ... (其他参数同e_step_with_omega)
    prov_to_idx : Dict[int, int]
        省份ID到矩阵索引的映射
    lbfgsb_maxiter : int
        L-BFGS-B最大迭代次数

    返回:
    ----
    updated_params : Dict[str, Any]
        更新后的参数
    updated_pi_k : np.ndarray
        更新后的类型概率
    """
    logger = logging.getLogger()
    logger.info(f"\n  [M-step with ω] Starting parameter optimization...")

    from scipy.optimize import minimize
    from src.config.model_config import ModelConfig

    # 创建配置对象以获取参数边界
    config = ModelConfig()

    unique_individuals = list(individual_posteriors.keys())
    N = len(unique_individuals)
    K = n_types

    # 创建个体ID到索引的映射
    individual_to_idx = {ind_id: idx for idx, ind_id in enumerate(unique_individuals)}

    # 为每个个体预先枚举ω（与E-step保持一致）
    enumerator = SimplifiedOmegaEnumerator(support_generator)
    individual_omega_lists = {}

    logger.info("  Pre-enumerating omega for all individuals...")
    for individual_id in unique_individuals:
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=max_omega_per_individual
        )
        individual_omega_lists[individual_id] = omega_list

    # --- Objective Function with detailed logging ---
    class Objective:
        def __init__(self):
            self.call_count = 0
            self.last_call_end_time = time.time()

        def function(self, param_values: np.ndarray, param_names: List[str]) -> float:
            self.call_count += 1
            start_time = time.time()

            log_msg_header = f"    [M-step Objective Call #{self.call_count}]"
            logger.info(f"{log_msg_header} Starting...")

            params = _unpack_params(param_values, param_names, initial_params['n_choices'])
            
            # **调试增强**: 输出当前参数值
            logger.debug(f"{log_msg_header} Current params: {params}")
            
            # **调试增强**: 检查参数值是否有效
            invalid_params = {k: v for k, v in params.items() if np.isnan(v) or np.isinf(v)}
            if invalid_params:
                logger.warning(f"{log_msg_header} Invalid params detected: {invalid_params}")
                return 1e10
            
            total_weighted_log_lik = 0.0

            # **调试增强**: 统计计数器
            n_valid_computations = 0
            n_failed_computations = 0
            individual_log_liks = []
            
            # **调试增强**: 记录权重和似然值
            all_weights = []
            all_log_liks = []

            try:
                # This is the triply nested loop causing the long runtime
                for i_idx, individual_id in enumerate(unique_individuals):
                    individual_data = observed_data[observed_data['individual_id'] == individual_id]
                    posterior_matrix = individual_posteriors[individual_id]
                    omega_list = individual_omega_lists[individual_id]

                    individual_contribution = 0.0

                    # **调试增强**: 输出当前个体的信息
                    if self.call_count == 1 and i_idx == 0:
                        logger.debug(f"{log_msg_header} Processing individual {individual_id}, posterior_matrix shape: {posterior_matrix.shape}")

                    for omega_idx, omega in enumerate(omega_list):
                        for k in range(K):
                            weight = posterior_matrix[omega_idx, k]
                            if weight < 1e-10: continue

                            # **调试增强**: 输出权重信息
                            if self.call_count <= 2 and i_idx == 0 and omega_idx == 0:
                                logger.debug(f"{log_msg_header} ind={individual_id}, omega_idx={omega_idx}, type={k}, weight={weight:.6f}")

                            type_params = params.copy()
                            if f'gamma_0_type_{k}' in params:
                                type_params['gamma_0'] = params[f'gamma_0_type_{k}']
                            type_params['sigma_epsilon'] = omega['sigma']

                            # **调试增强**: 输出前几个参数值
                            if self.call_count == 1 and i_idx == 0 and omega_idx == 0 and k == 0:
                                debug_params = {k: v for k, v in type_params.items() if k in ['alpha_w', 'gamma_0', 'sigma_epsilon']}
                                logger.debug(f"{log_msg_header} Type {k} params sample: {debug_params}")

                            try:
                                # --- OPTIMIZATION: Use individual solver ---
                                converged_v_individual, converged = solve_bellman_equation_individual(
                                    utility_function=None, # Not needed, called internally
                                    individual_data=individual_data,
                                    params=type_params,
                                    agent_type=int(k),
                                    beta=beta,
                                    transition_matrices=transition_matrices,
                                    regions_df=regions_df,
                                    distance_matrix=distance_matrix,
                                    adjacency_matrix=adjacency_matrix,
                                    verbose=False,
                                    prov_to_idx=prov_to_idx
                                )

                                if not converged:
                                    logger.warning(f"{log_msg_header} Bellman did not converge for ind={individual_id}, omega_idx={omega_idx}, type={k}")
                                    n_failed_computations += 1
                                    continue

                                # --- OPTIMIZATION: Use individual likelihood calculator ---
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

                                # **调试增强**: 检查似然值是否有效
                                if np.isnan(individual_log_lik) or np.isinf(individual_log_lik) or individual_log_lik < -1e9:
                                    logger.warning(f"{log_msg_header} Invalid log-lik={individual_log_lik:.2f} for ind={individual_id}, omega_idx={omega_idx}, type={k}")
                                    n_failed_computations += 1
                                    continue

                                # **调试增强**: 输出前几个计算的似然值
                                if self.call_count <= 2 and i_idx == 0 and omega_idx == 0 and k <= 2:
                                    logger.debug(f"{log_msg_header} ind={individual_id}, omega_idx={omega_idx}, type={k}, weight={weight:.6f}, log_lik={individual_log_lik:.6f}")

                                # **调试增强**: 收集权重和似然值用于统计
                                if self.call_count == 1:
                                    all_weights.append(weight)
                                    all_log_liks.append(individual_log_lik)

                                weighted_contribution = weight * individual_log_lik
                                individual_contribution += weighted_contribution
                                n_valid_computations += 1

                            except Exception as inner_e:
                                logger.warning(f"{log_msg_header} Inner exception for ind={individual_id}, omega_idx={omega_idx}, type={k}: {inner_e}")
                                n_failed_computations += 1
                                continue

                    total_weighted_log_lik += individual_contribution
                    individual_log_liks.append(individual_contribution)

                neg_ll = -total_weighted_log_lik
                duration = time.time() - start_time
                time_since_last = time.time() - self.last_call_end_time
                self.last_call_end_time = time.time()

                logger.info(f"{log_msg_header} Finished. Duration: {duration:.2f}s. Total time since last call: {time_since_last:.2f}s. NegLogLik: {neg_ll:.4f}")
                logger.info(f"{log_msg_header} Valid computations: {n_valid_computations}, Failed: {n_failed_computations}")
                logger.info(f"{log_msg_header} Mean individual log-lik: {np.mean(individual_log_liks):.4f}" if individual_log_liks else f"{log_msg_header} No valid individual log-liks")
                param_subset = param_values[:3]
                logger.info(f"{log_msg_header} Params (first 3): {np.round(param_subset, 4)}")
                
                # **调试增强**: 输出权重和似然值的统计信息
                if self.call_count == 1 and all_weights and all_log_liks:
                    logger.debug(f"{log_msg_header} Weight stats - Mean: {np.mean(all_weights):.6f}, Std: {np.std(all_weights):.6f}, Min: {np.min(all_weights):.6f}, Max: {np.max(all_weights):.6f}")
                    logger.debug(f"{log_msg_header} Log-lik stats - Mean: {np.mean(all_log_liks):.6f}, Std: {np.std(all_log_liks):.6f}, Min: {np.min(all_log_liks):.6f}, Max: {np.max(all_log_liks):.6f}")

                # **调试增强**: 如果所有计算都失败，返回大惩罚值
                if n_valid_computations == 0:
                    logger.error(f"{log_msg_header} All computations failed! Returning penalty.")
                    return 1e10

                return neg_ll

            except Exception as e:
                logger.error(f"{log_msg_header} Error: {e}", exc_info=True)
                return 1e10

    # 打包参数并优化
    objective = Objective()
    initial_param_values, param_names = _pack_params(initial_params)
    
    # **调试增强**: 输出打包后的参数信息
    logger.debug(f"  Packed params - names: {param_names[:5]}..., values: {initial_param_values[:5]}...")
    
    # **关键修复**: 从ModelConfig获取参数边界约束
    param_bounds = config.get_parameter_bounds(param_names)
    logger.info(f"  Optimizing {len(param_names)} parameters with L-BFGS-B...")
    logger.info(f"  Parameter bounds set for {len(param_bounds)} parameters")

    # Run initial call to log starting value
    initial_neg_ll = objective.function(initial_param_values, param_names)
    logger.info(f"  Initial objective value: {initial_neg_ll:.4f}")
    objective.call_count = 0 # Reset counter for the actual optimization

    try:
        result = minimize(
            objective.function,
            initial_param_values,
            args=(param_names,),
            method='L-BFGS-B',
            bounds=param_bounds,  # **关键修复**: 添加边界约束
            options={
                'disp': False,
                'maxiter': lbfgsb_maxiter,
                'gtol': 1e-4,      # 降低梯度容忍度以增加敏感性
                'ftol': 1e-4,      # 降低函数值容忍度
                'eps': 1e-7        # 增加数值微分精度
            }
        )

        final_neg_ll = result.fun
        logger.info(f"  L-BFGS-B result: success={result.success}, nit={result.nit}")
        logger.info(f"  Objective change: {initial_neg_ll:.4f} -> {final_neg_ll:.4f} "
                   f"(Δ {initial_neg_ll - final_neg_ll:.4f})")
        
        # **调试增强**: 输出优化结果的详细信息
        if hasattr(result, 'nit') and result.nit > 0:
            logger.debug(f"  Optimization details - nit: {result.nit}, nfev: {getattr(result, 'nfev', 'N/A')}, njev: {getattr(result, 'njev', 'N/A')}")
        
        if result.success or result.nit > 0:
            updated_params = _unpack_params(result.x, param_names, initial_params['n_choices'])
            
            # **调试增强**: 输出参数变化
            param_changes = {}
            for i, param_name in enumerate(param_names):
                initial_val = initial_param_values[i]
                final_val = result.x[i]
                if abs(final_val - initial_val) > 1e-6:
                    param_changes[param_name] = (initial_val, final_val, final_val - initial_val)
            
            if param_changes:
                logger.debug(f"  Significant parameter changes:")
                for param_name, (initial, final, change) in list(param_changes.items())[:5]:
                    logger.debug(f"    {param_name}: {initial:.6f} -> {final:.6f} (Δ {change:.6f})")
            else:
                logger.debug(f"  No significant parameter changes detected")
        else:
            logger.warning(f"  M-step optimization did not converge, using initial params")
            updated_params = initial_params

    except Exception as e:
        logger.error(f"  Error in M-step optimization: {e}", exc_info=True)
        updated_params = initial_params

    # 更新类型概率 π_k
    # π_k^(k+1) = (1/N) Σ_i Σ_ω p(τ, ω | D_i)
    updated_pi_k = np.zeros(K)
    for individual_id in unique_individuals:
        posterior_matrix = individual_posteriors[individual_id]  # (n_omega, K)
        # 边缘化ω
        marginal_type_prob = np.sum(posterior_matrix, axis=0)  # (K,)
        updated_pi_k += marginal_type_prob

    updated_pi_k = updated_pi_k / N
    updated_pi_k = np.maximum(updated_pi_k, 0.01)  # 最小概率
    updated_pi_k = updated_pi_k / np.sum(updated_pi_k)

    logger.info(f"  M-step completed. Updated type probabilities: {updated_pi_k}")

    return updated_params, updated_pi_k


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
    lbfgsb_maxiter: int = 15
) -> Dict[str, Any]:
    """
    EM-NFXP算法主循环（带离散支撑点ω）

    这是run_em_algorithm()的扩展版本，集成了离散支撑点枚举。

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
    logger = logging.getLogger()
    logger.info("\n" + "="*80)
    logger.info("EM-NFXP Algorithm with Discrete Support Points (ω)")
    logger.info("="*80)

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

    prev_log_likelihood = -np.inf
    converged = False

    # EM迭代
    for iteration in range(max_iterations):
        logger.info(f"\n{'='*80}")
        logger.info(f"EM Iteration {iteration + 1}/{max_iterations}")
        logger.info(f"{'='*80}")

        # E-step
        logger.info("\n[E-step with ω]")
        individual_posteriors, log_likelihood_matrix = e_step_with_omega(
            params=current_params,
            pi_k=current_pi_k,
            observed_data=observed_data,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=beta,
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_generator,
            n_types=n_types,
            prov_to_idx=prov_to_idx,
            max_omega_per_individual=max_omega_per_individual,
            use_simplified_omega=use_simplified_omega
        )

        # 计算当前对数似然
        current_log_likelihood = np.sum(np.log(np.sum(
            log_likelihood_matrix * current_pi_k[np.newaxis, :],
            axis=1
        ) + 1e-300))

        logger.info(f"\n  Current log-likelihood: {current_log_likelihood:.4f}")

        # 检查收敛
        if iteration > 0:
            ll_change = current_log_likelihood - prev_log_likelihood
            logger.info(f"  Log-likelihood change: {ll_change:.6f}")

            if abs(ll_change) < tolerance:
                logger.info(f"\n  ✓ Converged! (Δ log-likelihood < {tolerance})")
                converged = True
                break

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
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_generator,
            n_types=n_types,
            prov_to_idx=prov_to_idx,
            max_omega_per_individual=max_omega_per_individual,
            lbfgsb_maxiter=lbfgsb_maxiter
        )

    # 汇总结果
    logger.info("\n" + "="*80)
    if converged:
        logger.info(f"✓ EM Algorithm Converged after {iteration + 1} iterations")
    else:
        logger.info(f"⚠ EM Algorithm reached max iterations ({max_iterations})")
    logger.info(f"Final log-likelihood: {prev_log_likelihood:.4f}")
    logger.info(f"Final type probabilities: {current_pi_k}")
    logger.info("="*80)

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
