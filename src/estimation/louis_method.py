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
from src.config.model_config import ModelConfig

logger = logging.getLogger(__name__)

def _compute_individual_contribution_streaming(
    packed_params: np.ndarray,
    param_names: List[str],
    n_choices: int,
    individual_data: pd.DataFrame,
    agent_type: int,
    omega_values: Dict[str, float],
    omega_prob: float,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    prov_to_idx: Dict[int, int],
    h_step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    流式计算单个个体在特定(omega, type)组合下的统计贡献
    
    返回:
        information_matrix: 信息矩阵贡献 (p×p)
        score_vector: 分数向量贡献 (p,)
    """
    try:
        # 构建type-specific参数
        params = _unpack_params(packed_params, param_names, n_choices)
        type_params = params.copy()
        if f'gamma_0_type_{agent_type}' in params:
            type_params['gamma_0'] = params[f'gamma_0_type_{agent_type}']
        type_params['sigma_epsilon'] = omega_values['sigma']

        # 求解Bellman方程
        converged_v, _ = solve_bellman_equation_individual(
            utility_function=None,
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

        # 定义似然函数用于数值微分
        def likelihood_func(p):
            temp_params = _unpack_params(p, param_names, n_choices)
            temp_type_params = temp_params.copy()
            if f'gamma_0_type_{agent_type}' in temp_params:
                temp_type_params['gamma_0'] = temp_params[f'gamma_0_type_{agent_type}']
            temp_type_params['sigma_epsilon'] = omega_values['sigma']
            
            log_lik_obs = calculate_likelihood_from_v_individual(
                converged_v_individual=converged_v,
                params=temp_type_params,
                individual_data=individual_data,
                agent_type=agent_type,
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                prov_to_idx=prov_to_idx
            )
            return np.sum(log_lik_obs)

        # 计算梯度和Hessian
        grad_calculator = Gradient(likelihood_func, step=h_step)
        score_vector = grad_calculator(packed_params)
        
        hessian_calculator = Hessian(likelihood_func, step=h_step)
        hessian_matrix = hessian_calculator(packed_params)
        
        # 检查数值稳定性
        if not np.all(np.isfinite(score_vector)) or not np.all(np.isfinite(hessian_matrix)):
            logger.debug(f"数值不稳定: agent_type={agent_type}, omega_prob={omega_prob}")
            return np.zeros((len(packed_params), len(packed_params))), np.zeros(len(packed_params))
        
        return -hessian_matrix, score_vector  # 信息矩阵 = -E[Hessian]
        
    except Exception as e:
        logger.debug(f"计算个体贡献失败: {e}")
        n_params = len(packed_params)
        return np.zeros((n_params, n_params)), np.zeros(n_params)


def louis_method_standard_errors_streaming(
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
    support_generator: Any,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 100,  # 保持完整数量！
    use_simplified_omega: bool = True,
    h_step: float = 1e-3
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    流式Louis方法 - 保持统计完整性的内存优化版本
    
    核心思想：
    1. 不存储所有个体的omega数据
    2. 逐个体计算，累积统计量矩阵
    3. 保持完整的omega枚举数量
    """
    logger.info("\n开始流式Louis方法计算标准误（保持统计完整性）...")
    logger.info(f"  内存优化: 流式处理，保持完整omega枚举 ({max_omega_per_individual} per individual)")
    
    # 1. 准备参数
    packed_estimated_params, param_names = _pack_params(estimated_params)
    n_params = len(packed_estimated_params)
    n_choices = estimated_params['n_choices']
    unique_individuals = observed_data['individual_id'].unique()
    N = len(unique_individuals)
    
    logger.info(f"  样本信息: {N} 个个体, {n_params} 个参数, {n_types} 种类型")
    logger.info(f"  内存使用: O(p²) = {n_params*n_params*8/1024/1024:.1f}MB (仅统计量矩阵)")
    
    # 2. 初始化累积矩阵（只占O(p²)内存）
    expected_complete_information = np.zeros((n_params, n_params))
    sum_of_individual_scores_outer_product = np.zeros((n_params, n_params))
    
    # 3. 获取配置中的权重阈值
    config = ModelConfig()
    weight_threshold = config.weight_threshold
    
    # 4. 创建omega枚举器
    from src.model.discrete_support import SimplifiedOmegaEnumerator
    enumerator = SimplifiedOmegaEnumerator(support_generator)
    
    # 5. 流式处理每个个体
    start_time = pd.Timestamp.now()
    processed_combinations = 0
    skipped_combinations = 0
    
    for i_idx, individual_id in enumerate(unique_individuals):
        if (i_idx + 1) % max(1, N // 10) == 0:
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"    进度: {i_idx+1}/{N} ({(i_idx+1)/N*100:.1f}%), 用时: {elapsed:.1f}s")
        
        # 获取个体数据
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        posterior_matrix = individual_posteriors[individual_id]  # (n_omega, K)
        
        # 关键：按需枚举omega，保持完整数量但不存储！
        omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=max_omega_per_individual  # 保持完整数量！
        )
        
        # 初始化个体级累积变量
        individual_info_sum = np.zeros((n_params, n_params))
        individual_score_sum = np.zeros(n_params)
        individual_total_weight = 0.0
        
        # 检查posterior_matrix的维度是否匹配
        if posterior_matrix.shape[0] != len(omega_list):
            logger.warning(f"个体 {individual_id}: posterior矩阵维度不匹配 {posterior_matrix.shape[0]} vs {len(omega_list)}，调整posterior矩阵")
            # 重新构建posterior矩阵以匹配实际的omega列表
            actual_posterior = np.zeros((len(omega_list), n_types))
            min_size = min(posterior_matrix.shape[0], len(omega_list))
            actual_posterior[:min_size, :] = posterior_matrix[:min_size, :]
            posterior_matrix = actual_posterior
        
        # 遍历每个(omega, type)组合
        for omega_idx, omega in enumerate(omega_list):
            for k in range(n_types):
                weight = posterior_matrix[omega_idx, k]  # p(τ, ω | D_i)
                
                if weight < weight_threshold:  # 忽略极小权重
                    skipped_combinations += 1
                    continue
                
                # 流式计算该组合的统计贡献
                info_contrib, score_contrib = _compute_individual_contribution_streaming(
                    packed_params=packed_estimated_params,
                    param_names=param_names,
                    n_choices=n_choices,
                    individual_data=individual_data,
                    agent_type=k,
                    omega_values=omega,
                    omega_prob=weight,
                    beta=beta,
                    transition_matrices=transition_matrices,
                    regions_df=regions_df,
                    distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix,
                    prov_to_idx=prov_to_idx,
                    h_step=h_step
                )
                
                # 加权累积
                individual_info_sum += weight * info_contrib
                individual_score_sum += weight * score_contrib
                individual_total_weight += weight
                processed_combinations += 1
        
        # 检查权重总和（应该接近1）
        if abs(individual_total_weight - 1.0) > 0.1:
            logger.warning(f"  个体 {individual_id} 权重总和异常: {individual_total_weight:.3f}")
        
        # 累积到总矩阵
        expected_complete_information += individual_info_sum
        sum_of_individual_scores_outer_product += np.outer(individual_score_sum, individual_score_sum)
    
    # 计算完成统计
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"  流式计算完成: 总用时 {total_time:.1f}s, 平均每个个体 {total_time/N:.2f}s")
    logger.info(f"  处理组合数: {processed_combinations}, 跳过组合数: {skipped_combinations}")
    
    # 5. 最终计算（标准Louis公式）
    observed_information = expected_complete_information - sum_of_individual_scores_outer_product
    
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
            
            if std_err > 1e-10:
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
    
    logger.info("流式Louis方法标准误计算完成。")
    logger.info(f"  结果概览: {len(std_errors)} 个参数, 信息矩阵条件数: {np.linalg.cond(observed_information):.2e}")
    
    return std_errors, t_stats, p_values

def _louis_method_standard_errors_stratified_deprecated(
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
    support_generator: Any,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 100,
    use_simplified_omega: bool = True,
    h_step: float = 1e-3,
    n_strata: int = 16,
    strata_size: int = 1000
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    [已弃用] 分层抽样Louis方法
    
    警告：此方法已被流式Louis方法取代，仅保留用于向后兼容。
    建议使用louis_method_standard_errors_streaming()获得更好的性能和统计完整性。
    """
    import warnings
    warnings.warn(
        "_louis_method_standard_errors_stratified_deprecated已弃用，"
        "请使用louis_method_standard_errors_streaming()代替，"
        "后者提供更好的性能和统计完整性",
        DeprecationWarning, stacklevel=2
    )
    logger.info(f"\n开始大样本Louis方法（分层抽样）计算标准误...")
    logger.info(f"  分层策略: {n_strata}层，每层{strata_size}个个体")
    logger.info(f"  内存优化模式: 流式处理，避免大对象创建")
    
    # 1. 创建分层抽样（内存优化版）
    try:
        sampled_data, sampled_posteriors = create_stratified_sample_deprecated(
            observed_data, individual_posteriors, n_strata, strata_size
        )
    except MemoryError as e:
        logger.error(f"分层抽样内存不足: {e}")
        logger.info("降级为系统抽样...")
        # 降级为简单的系统抽样
        sampled_data, sampled_posteriors = create_systematic_sample_deprecated(
            observed_data, individual_posteriors, n_strata * strata_size
        )
    except Exception as e:
        logger.error(f"分层抽样失败: {e}")
        logger.info("使用原始数据...")
        return _louis_method_standard_errors_core(
            estimated_params=estimated_params,
            type_probabilities=type_probabilities,
            individual_posteriors=individual_posteriors,
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
            use_simplified_omega=use_simplified_omega,
            h_step=h_step
        )
    
    n_sampled = len(sampled_data['individual_id'].unique())
    logger.info(f"  分层抽样完成: 选择了{n_sampled}个代表性个体")
    
    # 2. 使用标准Louis方法处理抽样数据
    return _louis_method_standard_errors_core(
        estimated_params=estimated_params,
        type_probabilities=type_probabilities,
        individual_posteriors=sampled_posteriors,
        observed_data=sampled_data,
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
        use_simplified_omega=use_simplified_omega,
        h_step=h_step
    )

def create_stratified_sample_deprecated(observed_data: pd.DataFrame, 
                           individual_posteriors: Dict[str, np.ndarray],
                           n_strata: int = 16,
                           strata_size: int = 1000) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    [已弃用] 分层抽样函数
    
    警告：此方法已被流式方法取代，仅保留用于向后兼容。
    流式方法提供了更好的内存效率和统计完整性。
    """
    import warnings
    warnings.warn(
        "create_stratified_sample_deprecated已弃用，"
        "流式方法提供了更好的内存效率和统计完整性",
        DeprecationWarning, stacklevel=2
    )
    logger.info(f"创建{n_strata}层分层抽样，每层{strata_size}个个体")
    
    # 获取所有个体ID - 使用numpy数组减少内存
    all_individuals = observed_data['individual_id'].unique()
    n_total = len(all_individuals)
    
    if n_total <= n_strata * strata_size:
        logger.info(f"总样本量{n_total}较小，使用全部数据")
        return observed_data, individual_posteriors
    
    logger.info(f"大样本优化: {n_total}个体 -> {n_strata * strata_size}代表性样本")
    
    # === 内存优化的分层策略 ===
    
    # 1. 预计算分层映射（避免重复pandas操作）
    strata_mappings = {}
    
    # 地理分层（内存优化版）
    if 'provcd_t' in observed_data.columns:
        regions = observed_data['provcd_t'].unique()
        if len(regions) >= n_strata:
            # 预创建映射字典
            region_map = {region: i % n_strata for i, region in enumerate(sorted(regions))}
            strata_mappings['region'] = region_map
    
    # 年龄分层（使用numpy分位数，避免pandas qcut）
    if 'age' in observed_data.columns:
        ages = observed_data['age'].values
        n_age_groups = min(n_strata, 10)
        # 使用numpy分位数代替pandas qcut
        quantiles = np.linspace(0, 1, n_age_groups + 1)
        age_thresholds = np.quantile(ages, quantiles)
        # 创建简单的年龄组映射函数
        def get_age_group(age):
            for i in range(n_age_groups):
                if age <= age_thresholds[i + 1]:
                    return i % n_strata
            return (n_age_groups - 1) % n_strata
        strata_mappings['age_group'] = get_age_group
    
    # 教育分层（同样使用numpy方法）
    if 'edu' in observed_data.columns:
        edus = observed_data['edu'].values
        n_edu_groups = min(n_strata, 5)
        quantiles = np.linspace(0, 1, n_edu_groups + 1)
        edu_thresholds = np.quantile(edus, quantiles)
        def get_edu_group(edu):
            for i in range(n_edu_groups):
                if edu <= edu_thresholds[i + 1]:
                    return i % n_strata
            return (n_edu_groups - 1) % n_strata
        strata_mappings['edu_group'] = get_edu_group
    
    # 2. 流式分层分配（逐行处理，避免大DataFrame操作）
    individual_to_stratum = {}
    
    # 只遍历一次数据，逐行分配分层
    for _, row in observed_data[['individual_id', 'provcd_t', 'age', 'edu']].drop_duplicates('individual_id').iterrows():
        ind_id = row['individual_id']
        
        # 计算综合分层（使用简单的模运算避免hash）
        stratum_factors = []
        
        if 'region' in strata_mappings and pd.notna(row.get('provcd_t')):
            region_stratum = strata_mappings['region'].get(row['provcd_t'], 0)
            stratum_factors.append(region_stratum)
        
        if 'age_group' in strata_mappings and pd.notna(row.get('age')):
            age_stratum = strata_mappings['age_group'](row['age'])
            stratum_factors.append(age_stratum)
            
        if 'edu_group' in strata_mappings and pd.notna(row.get('edu')):
            edu_stratum = strata_mappings['edu_group'](row['edu'])
            stratum_factors.append(edu_stratum)
        
        # 综合分层（简单求和模运算）
        if stratum_factors:
            final_stratum = sum(stratum_factors) % n_strata
        else:
            # 备用方案：使用个体ID
            final_stratum = hash(str(ind_id)) % n_strata
            
        individual_to_stratum[ind_id] = final_stratum
    
    # 3. 内存优化的分层抽样（使用numpy数组）
    stratum_individuals = [[] for _ in range(n_strata)]
    
    # 按分层分组（使用列表，避免pandas操作）
    for ind_id, stratum in individual_to_stratum.items():
        if 0 <= stratum < n_strata:
            stratum_individuals[stratum].append(ind_id)
    
    # 4. 在每层内抽样（使用numpy随机抽样）
    selected_individuals = []
    
    for stratum in range(n_strata):
        individuals_in_stratum = stratum_individuals[stratum]
        n_in_stratum = len(individuals_in_stratum)
        
        if n_in_stratum >= strata_size:
            # 使用numpy随机抽样（比pandas高效）
            selected = np.random.choice(individuals_in_stratum, size=strata_size, replace=False)
        else:
            # 层内个体不足
            selected = individuals_in_stratum
            logger.warning(f"层{stratum}个体不足({n_in_stratum}<{strata_size})，使用全部个体")
        
        selected_individuals.extend(selected)
    
    # 5. 最终筛选（内存优化版）
    target_size = n_strata * strata_size
    if len(selected_individuals) < target_size:
        # 从剩余个体中补充（使用集合差集）
        all_set = set(all_individuals)
        selected_set = set(selected_individuals)
        remaining = list(all_set - selected_set)
        
        n_needed = target_size - len(selected_individuals)
        if remaining and n_needed > 0:
            additional = np.random.choice(remaining, size=min(n_needed, len(remaining)), replace=False)
            selected_individuals.extend(additional)
    
    # 截断到目标大小
    selected_individuals = selected_individuals[:target_size]
    
    # 6. 高效的数据筛选（使用isin，但先转换为集合）
    selected_set = set(selected_individuals)
    
    # 筛选数据（使用布尔索引，避免大DataFrame复制）
    mask = observed_data['individual_id'].isin(selected_set)
    selected_data = observed_data[mask].copy()
    
    # 筛选后验概率（字典推导，避免大字典复制）
    selected_posteriors = {
        ind_id: posterior for ind_id, posterior in individual_posteriors.items() 
        if ind_id in selected_set
    }
    
    logger.info(f"分层抽样完成: {len(selected_individuals)}个体，来自{n_strata}层，内存优化模式")
    
    return selected_data, selected_posteriors

def create_systematic_sample_deprecated(observed_data: pd.DataFrame, 
                           individual_posteriors: Dict[str, np.ndarray],
                           sample_size: int = 16000) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    [已弃用] 系统抽样函数
    
    警告：此方法已被流式方法取代，仅保留用于向后兼容。
    流式方法可以在不牺牲统计完整性的情况下处理超大样本。
    """
    import warnings
    warnings.warn(
        "create_systematic_sample_deprecated已弃用，"
        "流式方法可以在不牺牲统计完整性的情况下处理超大样本",
        DeprecationWarning, stacklevel=2
    )
    logger.info(f"创建系统抽样，目标样本量{sample_size}（内存超简单模式）")
    
    all_individuals = observed_data['individual_id'].unique()
    n_total = len(all_individuals)
    
    if n_total <= sample_size:
        return observed_data, individual_posteriors
    
    # 超简单系统抽样：直接按顺序选取，避免复杂计算
    step = max(1, n_total // sample_size)
    start = np.random.randint(0, min(step, max(1, n_total - sample_size)))
    
    # 直接索引，避免大列表
    selected_indices = list(range(start, min(n_total, start + sample_size * step), step))
    selected_individuals = [all_individuals[i] for i in selected_indices[:sample_size]]
    
    logger.info(f"系统抽样完成: 选择了{len(selected_individuals)}个个体，间隔{step}（超简单模式）")
    
    # 内存优化的数据筛选
    selected_set = set(selected_individuals)
    mask = observed_data['individual_id'].isin(selected_set)
    selected_data = observed_data[mask].copy()
    
    # 后验概率筛选
    selected_posteriors = {
        ind_id: posterior for ind_id, posterior in individual_posteriors.items() 
        if ind_id in selected_set
    }
    
    return selected_data, selected_posteriors

def _louis_method_standard_errors_core(
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
    max_omega_per_individual: int = 100,
    use_simplified_omega: bool = True,
    h_step: float = 1e-3 # 数值微分步长
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
    
    # 流式方法：保持统计完整性，不再减少omega数量！
    adjusted_max_omega = max_omega_per_individual  # 保持完整omega枚举
    logger.info(f"  流式方法: 保持完整omega枚举 ({adjusted_max_omega} per individual)")
    
    for i, individual_id in enumerate(unique_individuals):
        if (i + 1) % max(1, N // 5) == 0 or i < 3:  # 显示前3个和每20%的进度
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"    预枚举进度: {i+1}/{N} 个体, 用时: {elapsed:.1f}s")
        
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=adjusted_max_omega
        )
        individual_omega_lists[individual_id] = omega_list
        individual_omega_probs[individual_id] = omega_probs
    
    enum_time = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"  预枚举完成: 用时 {enum_time:.1f}s, 平均每个个体 {enum_time/N:.2f}s")

    # 流式方法：处理所有个体，不再限制数量！
    logger.info(f"  开始流式计算信息矩阵和Score矩阵...")
    logger.info(f"  将处理所有 {N} 个个体，保持统计完整性")
    
    start_time = pd.Timestamp.now()
    last_log_time = start_time
    
    for i_idx, individual_id in enumerate(unique_individuals):
        # 更频繁的日志输出
        current_time = pd.Timestamp.now()
        if (i_idx + 1) % max(1, N // 10) == 0 or (current_time - last_log_time).total_seconds() > 5:  # 每10%或每5秒
            elapsed = (current_time - start_time).total_seconds()
            progress = (i_idx + 1) / N
            eta = elapsed / progress - elapsed if progress > 0 else 0
            logger.info(f"    处理个体 {i_idx+1}/{N} ({progress*100:.1f}%), 用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
            last_log_time = current_time

        # 流式方法：按需获取omega数据，不预存储！
        individual_data = observed_data[observed_data['individual_id'] == individual_id]
        posterior_matrix = individual_posteriors[individual_id]  # (n_omega, K)
        
        # 关键：按需枚举omega，保持完整数量！
        omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
            individual_data,
            max_combinations=max_omega_per_individual  # 保持完整数量！
        )
        
        individual_score_sum = np.zeros(n_params)  # S_i
        
        # 遍历每个(ω, τ)组合
        total_combinations = len(omega_list) * n_types
        processed_combinations = 0
        skipped_combinations = 0
        
        # 检查posterior_matrix的维度是否匹配
        if posterior_matrix.shape[0] != len(omega_list):
            logger.warning(f"个体 {individual_id}: posterior矩阵维度不匹配 {posterior_matrix.shape[0]} vs {len(omega_list)}，调整posterior矩阵")
            # 重新构建posterior矩阵以匹配实际的omega列表
            actual_posterior = np.zeros((len(omega_list), n_types))
            min_size = min(posterior_matrix.shape[0], len(omega_list))
            actual_posterior[:min_size, :] = posterior_matrix[:min_size, :]
            posterior_matrix = actual_posterior
        
        # 先计算所有权重的总和，用于相对权重阈值
        total_weight = np.sum(posterior_matrix)
        cumulative_weight = 0.0
        config = ModelConfig()
        weight_threshold = config.weight_threshold
        
        for omega_idx, omega in enumerate(omega_list):
            for k in range(n_types):
                weight = posterior_matrix[omega_idx, k]  # p(τ, ω | D_i)
                
                if weight < weight_threshold:  # 忽略权重过小的组合
                    skipped_combinations += 1
                    continue
                    
                # 使用相对权重和累积权重进行更智能的剪枝
                relative_weight = weight / total_weight if total_weight > 0 else 0
                cumulative_weight += relative_weight
                
                # 如果累积权重已经达到99%，跳过剩余的低权重组合
                if cumulative_weight > 0.90 and relative_weight < 0.01:
                    skipped_combinations += 1
                    continue

                # 定义似然函数用于数值微分
                def likelihood_func(p):
                    temp_params = _unpack_params(p, param_names, n_choices)
                    temp_type_params = temp_params.copy()
                    if f'gamma_0_type_{k}' in temp_params:
                        temp_type_params['gamma_0'] = temp_params[f'gamma_0_type_{k}']
                    temp_type_params['sigma_epsilon'] = omega['sigma']
                    
                    # 求解Bellman方程
                    converged_v, _ = solve_bellman_equation_individual(
                        utility_function=None,
                        individual_data=individual_data,
                        params=temp_type_params,
                        agent_type=k,
                        beta=beta,
                        transition_matrices=transition_matrices,
                        regions_df=regions_df,
                        distance_matrix=distance_matrix,
                        adjacency_matrix=adjacency_matrix,
                        verbose=False,
                        prov_to_idx=prov_to_idx
                    )
                    
                    # 计算似然
                    log_lik_obs = calculate_likelihood_from_v_individual(
                        converged_v_individual=converged_v,
                        params=temp_type_params,
                        individual_data=individual_data,
                        agent_type=k,
                        beta=beta,
                        transition_matrices=transition_matrices,
                        regions_df=regions_df,
                        distance_matrix=distance_matrix,
                        adjacency_matrix=adjacency_matrix,
                        prov_to_idx=prov_to_idx
                    )
                    return np.sum(log_lik_obs)

                # 计算梯度 (Score)
                try:
                    grad_calculator = Gradient(likelihood_func, step=h_step)
                    score_vector = grad_calculator(packed_estimated_params)
                    if not np.all(np.isfinite(score_vector)):
                        logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: Score包含非有限值，跳过。weight={weight:.2e}")
                        skipped_combinations += 1
                        continue
                except Exception as e:
                    logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: 计算Score失败: {e}, weight={weight:.2e}")
                    skipped_combinations += 1
                    continue

                # 计算Hessian
                try:
                    hessian_calculator = Hessian(likelihood_func, step=h_step)
                    hessian_matrix = hessian_calculator(packed_estimated_params)
                    if not np.all(np.isfinite(hessian_matrix)):
                        logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: Hessian包含非有限值，跳过。weight={weight:.2e}")
                        skipped_combinations += 1
                        continue
                except Exception as e:
                    logger.debug(f"  个体 {individual_id}, ω_idx {omega_idx}, 类型 {k}: 计算Hessian失败: {e}, weight={weight:.2e}")
                    skipped_combinations += 1
                    continue

                # 累加到期望完全信息矩阵
                expected_complete_information -= weight * hessian_matrix

                # 累加到个体Score和
                individual_score_sum += weight * score_vector
                processed_combinations += 1
        
        # 记录当前个体的处理统计
        if processed_combinations > 0:
            logger.debug(f"  个体 {individual_id}: 处理 {processed_combinations}/{total_combinations} 组合, 跳过 {skipped_combinations} 组合")

        # 累加到Score协方差矩阵
        sum_of_individual_scores_outer_product += np.outer(individual_score_sum, individual_score_sum)
        
        # 记录当前个体的处理统计
        if (i_idx + 1) % max(1, N // 10) == 0:  # 每10%输出一次统计
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

def louis_method_standard_errors_safe(
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
    support_generator: Any,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 100,
    use_simplified_omega: bool = True,
    h_step: float = 1e-3,
    force_standard_method: bool = False,  # 强制使用标准方法
    memory_safe_mode: bool = True,        # 内存安全模式
    large_sample_threshold: int = 1000,
    use_stratified_sampling: bool = True,
    n_strata: int = 16,
    strata_size: int = 1000
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Louis方法的安全版本 - 专门解决大样本内存问题
    
    针对16,000个体大样本的内存优化版本，提供多层保护机制
    """
    logger.info("\n开始使用Louis方法安全版本计算标准误...")
    
    # 获取样本大小
    n_individuals = len(observed_data['individual_id'].unique())
    logger.info(f"  样本信息: {n_individuals} 个个体")
    
    # 内存安全检查
    if memory_safe_mode and n_individuals > 10000:
        logger.warning(f"  大样本警告: {n_individuals}个体，启用内存安全模式")
        logger.info(f"  将使用保守的抽样策略和内存优化")
        
        # 强制降低分层参数
        safe_n_strata = min(n_strata, 8)      # 最多8层
        safe_strata_size = min(strata_size, 500)  # 每层最多500
        target_sample_size = safe_n_strata * safe_strata_size
        
        logger.info(f"  安全参数: {safe_n_strata}层 × {safe_strata_size} = {target_sample_size}样本")
        
        # 使用系统抽样（最简单，最省内存）
        try:
            sampled_data, sampled_posteriors = create_systematic_sample(
                observed_data, individual_posteriors, target_sample_size
            )
            
            n_sampled = len(sampled_data['individual_id'].unique())
            logger.info(f"  系统抽样完成: {n_sampled}个代表性个体（安全模式）")
            
            # 使用流式Louis方法处理安全抽样数据，保持统计完整性
            return louis_method_standard_errors_streaming(
                estimated_params=estimated_params,
                type_probabilities=type_probabilities,
                individual_posteriors=sampled_posteriors,
                observed_data=sampled_data,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=beta,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                support_generator=support_generator,
                n_types=n_types,
                prov_to_idx=prov_to_idx,
                max_omega_per_individual=max_omega_per_individual,  # 保持完整数量！
                use_simplified_omega=use_simplified_omega,
                h_step=h_step
            )
            
        except Exception as e:
            logger.error(f"  安全模式也失败: {e}")
            logger.info("  降级到最简模式: 使用固定小样本")
            
            # 最极端的降级：使用流式方法处理小样本，但保持omega完整性
            return louis_method_standard_errors_streaming(
                estimated_params=estimated_params,
                type_probabilities=type_probabilities,
                individual_posteriors=individual_posteriors,
                observed_data=observed_data.head(5000),  # 只用前5000行
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=beta,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                support_generator=support_generator,
                n_types=n_types,
                prov_to_idx=prov_to_idx,
                max_omega_per_individual=max_omega_per_individual,  # 保持完整数量！
                use_simplified_omega=use_simplified_omega,
                h_step=h_step
            )
    
    # 正常模式（原逻辑）
    return louis_method_standard_errors(
        estimated_params=estimated_params,
        type_probabilities=type_probabilities,
        individual_posteriors=individual_posteriors,
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
        use_simplified_omega=use_simplified_omega,
        h_step=h_step,
        large_sample_threshold=large_sample_threshold,
        use_stratified_sampling=use_stratified_sampling,
        n_strata=n_strata,
        strata_size=strata_size
    )

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
    support_generator: Any,
    n_types: int,
    prov_to_idx: Dict[int, int],
    max_omega_per_individual: int = 100,
    use_simplified_omega: bool = True,
    h_step: float = 1e-3,
    large_sample_threshold: int = 1000,
    use_stratified_sampling: bool = True,
    n_strata: int = 16,
    strata_size: int = 1000
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    综合Louis方法标准误计算函数 - 自动选择最优策略
    
    根据样本大小自动选择标准方法或大样本方法（分层抽样）
    
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
    large_sample_threshold : int
        大样本阈值，超过此值使用分层抽样
    use_stratified_sampling : bool
        是否使用分层抽样策略
    n_strata : int
        分层数量（大样本时）
    strata_size : int
        每层样本大小（大样本时）

    返回:
    ----
    Tuple[Dict, Dict, Dict]: (标准误字典, t统计量字典, p值字典)
    """
    logger.info("\n开始使用综合Louis方法计算标准误...")
    
    # 获取样本大小
    n_individuals = len(observed_data['individual_id'].unique())
    logger.info(f"  样本信息: {n_individuals} 个个体")
    
    # 根据样本大小选择策略 - 统一使用流式方法保持统计完整性
    if n_individuals > large_sample_threshold:
        logger.info(f"  大样本检测: {n_individuals} > {large_sample_threshold}，使用流式方法保持统计完整性")
        logger.info(f"  流式配置: 保持完整omega枚举 ({max_omega_per_individual} per individual)")
        
        return louis_method_standard_errors_streaming(
            estimated_params=estimated_params,
            type_probabilities=type_probabilities,
            individual_posteriors=individual_posteriors,
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
            max_omega_per_individual=max_omega_per_individual,  # 保持完整数量！
            use_simplified_omega=use_simplified_omega,
            h_step=h_step
        )
    else:
        if n_individuals > large_sample_threshold:
            logger.info(f"  样本较大但禁用分层抽样，使用标准优化方法")
        else:
            logger.info(f"  小样本: {n_individuals}个个体，使用标准优化方法")
        
        return _louis_method_standard_errors_core(
            estimated_params=estimated_params,
            type_probabilities=type_probabilities,
            individual_posteriors=individual_posteriors,
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
            use_simplified_omega=use_simplified_omega,
            h_step=h_step
        )