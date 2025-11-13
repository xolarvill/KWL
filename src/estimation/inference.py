"""
该模块实现参数估计的标准误计算和统计推断
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Any, Tuple
import warnings
import logging
from joblib import Parallel, delayed
from src.estimation.louis_method import louis_method_standard_errors, louis_method_standard_errors_safe # 导入Louis方法


def compute_hessian_numerical(
    log_likelihood_func,
    params: Dict[str, float],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    agent_type: int,
    beta: float,
    regions_df: pd.DataFrame = None,
    distance_matrix: np.ndarray = None,
    adjacency_matrix: np.ndarray = None,
    h: float = 1e-4
) -> np.ndarray:
    """
    使用数值方法计算对数似然函数的海塞矩阵
    
    Args:
        log_likelihood_func: 对数似然函数
        params: 参数字典
        observed_data: 观测数据
        state_space: 状态空间
        transition_matrices: 转移矩阵
        agent_type: 代理人类型
        beta: 折现因子
        h: 数值微分步长
    
    Returns:
        np.ndarray: 海塞矩阵
    """
    # 提取参数值并记录参数名称（排除n_choices）
    param_names = sorted([k for k in params.keys() if k != 'n_choices'])
    param_values = np.array([params[name] for name in param_names])
    
    n_params = len(param_values)
    hessian = np.zeros((n_params, n_params))
    
    # 保存n_choices（如果存在）用于传递给似然函数
    n_choices_value = params.get('n_choices', None)

    # 计算梯度函数（一阶导数）
    def gradient(theta):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            # 创建正向和负向扰动的参数字典
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += h
            theta_minus[i] -= h

            # 重构参数字典，并添加n_choices
            params_plus = {name: val for name, val in zip(param_names, theta_plus)}
            params_minus = {name: val for name, val in zip(param_names, theta_minus)}
            if n_choices_value is not None:
                params_plus['n_choices'] = n_choices_value
                params_minus['n_choices'] = n_choices_value
            
            # 计算数值导数
            ll_plus_vector = log_likelihood_func(
                params=params_plus,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=agent_type,
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                verbose=False
            )
            ll_minus_vector = log_likelihood_func(
                params=params_minus,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=agent_type,
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                verbose=False
            )

            ll_plus = np.sum(ll_plus_vector)
            ll_minus = np.sum(ll_minus_vector)
            
            grad[i] = (ll_plus - ll_minus) / (2 * h)
        return grad
    
    # 计算海塞矩阵（二阶导数）
    base_grad = gradient(param_values)
    for i in range(n_params):
        for j in range(i, n_params):  # 只计算上三角矩阵
            if i == j:
                # 对角元素
                theta_plus = param_values.copy()
                theta_minus = param_values.copy()
                theta_plus[i] += h
                theta_minus[i] -= h
                
                grad_plus = gradient(theta_plus)[i]
                grad_minus = gradient(theta_minus)[i]
                
                hessian[i, j] = (grad_plus - grad_minus) / (2 * h)
            else:
                # 非对角元素
                theta_plus = param_values.copy()
                theta_minus = param_values.copy()
                theta_plus[i] += h
                theta_minus[i] -= h
                
                grad_plus = gradient(theta_plus)[j]
                grad_minus = gradient(theta_minus)[j]
                
                hessian[i, j] = (grad_plus - grad_minus) / (2 * h)
                hessian[j, i] = hessian[i, j]  # 海塞矩阵是对称的
    
    return hessian


def estimate_standard_errors(
    log_likelihood_func,
    params: Dict[str, float],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    agent_type: int,
    beta: float,
    regions_df: pd.DataFrame = None,
    distance_matrix: np.ndarray = None,
    adjacency_matrix: np.ndarray = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    估计参数的标准误、t统计量和p值
    
    Args:
        log_likelihood_func: 对数似然函数
        params: 估计得到的参数
        observed_data: 观测数据
        state_space: 状态空间
        transition_matrices: 转移矩阵
        agent_type: 代理人类型
        beta: 折现因子
    
    Returns:
        Tuple[Dict, Dict, Dict]: (标准误, t统计量, p值)
    """
    warnings.warn("使用数值方法计算海塞矩阵可能计算量较大，建议在参数收敛后再运行")

    # 计算海塞矩阵
    hessian = compute_hessian_numerical(
        log_likelihood_func, params, observed_data, state_space,
        transition_matrices, agent_type, beta,
        regions_df=regions_df,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix
    )
    
    # 信息矩阵是负海塞矩阵（负的二阶导数矩阵）
    info_matrix = -hessian
    
    # 计算协方差矩阵（信息矩阵的逆）
    try:
        cov_matrix = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        cov_matrix = np.linalg.pinv(info_matrix)
        warnings.warn("使用伪逆计算协方差矩阵，可能存在多重共线性")
    
    # 提取参数名称
    param_names = sorted(params.keys())
    
    # 计算标准误（协方差矩阵对角线的平方根）
    std_errors = {}
    for i, name in enumerate(param_names):
        std_errors[name] = np.sqrt(max(0, cov_matrix[i, i])) if i < cov_matrix.shape[0] and i < cov_matrix.shape[1] else 0.0
    
    # 计算t统计量
    t_stats = {}
    for name in param_names:
        if std_errors[name] > 0:
            t_stats[name] = params[name] / std_errors[name]
        else:
            t_stats[name] = np.inf if params[name] > 0 else -np.inf
    
    # 计算p值（双侧检验）
    p_values = {}
    for name in param_names:
        # 使用标准正态分布近似（大样本）
        p_values[name] = 2 * (1 - norm.cdf(abs(t_stats[name])))
    
    return std_errors, t_stats, p_values


def compute_information_criteria(
    log_likelihood: float,
    n_params: int,
    n_observations: int
) -> Dict[str, float]:
    """
    计算信息准则（AIC, BIC）

    Args:
        log_likelihood: 对数似然值
        n_params: 参数数量
        n_observations: 观测值数量

    Returns:
        Dict[str, float]: 信息准则
    """
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_observations)

    return {"AIC": aic, "BIC": bic}




def estimate_mixture_model_standard_errors(
    estimated_params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_types: int = 3,
    method: str = "louis", # default: "louis"; other: "shared_only", "all_numerical"
    individual_posteriors: Dict[Any, np.ndarray] = None, # Louis方法需要
    support_generator: Any = None, # Louis方法需要
    max_omega_per_individual: int = 100, # Louis方法需要
    use_simplified_omega: bool = True, # Louis方法需要
    h_step: float = 1e-3 # Louis方法需要
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    估计混合模型参数的标准误

    Args:
        estimated_params: 估计得到的所有参数（包括type-specific）
        observed_data: 观测数据
        state_space: 状态空间
        transition_matrices: 转移矩阵
        beta: 折现因子
        regions_df: 区域数据
        distance_matrix: 距离矩阵
        adjacency_matrix: 邻接矩阵
        n_types: 类型数量
        method: 计算方法
            - "louis": 使用Louis (1982)方法 (推荐)
            - "shared_only": 只计算共享参数的标准误（快速）
            - "all_numerical": 计算所有参数（非常耗时）
        individual_posteriors: E-step计算得到的个体后验概率 p(τ, ω | D_i) (Louis方法需要)
        support_generator: 离散支撑点生成器 (DiscreteSupportGenerator) (Louis方法需要)
        max_omega_per_individual: 每个个体的最大ω组合数 (Louis方法需要)
        use_simplified_omega: 是否使用简化ω策略 (Louis方法需要)
        h_step: 数值微分步长 (Louis方法需要)

    Returns:
        Tuple[Dict, Dict, Dict]: (标准误, t统计量, p值)
    """
    from src.estimation.em_with_omega import _prepare_numpy_region_data # 导入预处理函数

    std_errors, t_stats, p_values = {}, {}, {}

    # 预处理regions_df为NumPy数组，以兼容Louis方法
    prov_to_idx = {prov_id: idx for idx, prov_id in enumerate(regions_df['provcd'].unique())}
    regions_df_np = _prepare_numpy_region_data(regions_df, prov_to_idx)


    if method == "louis":
        if individual_posteriors is None or support_generator is None:
            raise ValueError("使用Louis方法时，必须提供individual_posteriors和support_generator。")
        
        # 确保regions_df是NumPy版本
        # 使用安全版本处理大样本内存问题
        return louis_method_standard_errors_safe(
            estimated_params=estimated_params,
            type_probabilities=None, # Louis方法中不需要，但为了兼容性保留
            individual_posteriors=individual_posteriors,
            observed_data=observed_data,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=beta,
            regions_df=regions_df_np, # 传入NumPy版本
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_generator,
            n_types=n_types,
            prov_to_idx=prov_to_idx,
            max_omega_per_individual=max_omega_per_individual,
            use_simplified_omega=use_simplified_omega,
            h_step=h_step
        )
        
    else:
        raise ValueError(f"未知或不支持的标准误计算方法: {method}")


def bootstrap_standard_errors(
    estimated_params: Dict[str, Any],
    posterior_probs: np.ndarray,
    type_probabilities: np.ndarray,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_types: int = 3,
    n_bootstrap: int = 100,
    max_em_iterations: int = 5,
    em_tolerance: float = 1e-4,
    seed: int = 42,
    n_jobs: int = 1,
    verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
    """
    使用参数化Bootstrap方法估计混合模型参数的标准误和置信区间

    该方法基于估计的参数和后验概率生成bootstrap样本，然后重新估计模型。
    参数估计值的标准差即为标准误，分位数即为置信区间。
    
    注意：此方法使用个体后验概率p(τ|D_i)进行重采样，适用于EM-with-ω算法的结果。

    Args:
        estimated_params: 原始估计的参数
        posterior_probs: 个体级别的类型后验概率 (N × K)
        type_probabilities: 类型概率 π_k
        observed_data: 原始观测数据
        state_space: 状态空间
        transition_matrices: 转移矩阵
        beta: 折现因子
        regions_df: 区域数据
        distance_matrix: 距离矩阵
        adjacency_matrix: 邻接矩阵
        n_types: 类型数量
        n_bootstrap: Bootstrap重复次数
        max_em_iterations: 每个bootstrap样本的最大EM迭代次数
        em_tolerance: EM算法的收敛容差
        seed: 随机种子
        n_jobs: 并行任务数（-1表示使用所有CPU核心）
        verbose: 是否打印进度信息

    Returns:
        Tuple[Dict, Dict, Dict, Dict]: (标准误字典, 95%置信区间字典, t统计量字典, p值字典)
    """
    logger = logging.getLogger()
    np.random.seed(seed)

    if verbose:
        print(f"\n开始 Bootstrap 标准误估计 ({n_bootstrap} 次重复)...", flush=True)
        print(f"原始样本大小: {len(observed_data.groupby('individual_id'))} 个个体", flush=True)
        print(f"EM配置: max_iterations={max_em_iterations}, tolerance={em_tolerance}", flush=True)

        # 显示原始估计的前5个参数
        param_sample = [k for k in estimated_params.keys() if k != 'n_choices'][:5]
        print(f"\n原始估计的参数值（前5个）:", flush=True)
        for pname in param_sample:
            print(f"  {pname}: {estimated_params[pname]:.6f}", flush=True)

    # 提取个体ID
    individual_ids = observed_data['individual_id'].unique()
    N = len(individual_ids)

    def generate_bootstrap_sample(b: int) -> pd.DataFrame:
        """
        生成一个bootstrap样本

        基于后验概率重新采样个体，并为每个个体分配类型
        """
        np.random.seed(seed + b)  # 确保可重现性

        # 重采样个体（有放回）
        bootstrap_individual_ids = np.random.choice(individual_ids, size=N, replace=True)

        # 为每个个体分配类型（基于后验概率）
        bootstrap_data_list = []
        for new_id, orig_id in enumerate(bootstrap_individual_ids):
            # 获取该个体的数据
            individual_data = observed_data[observed_data['individual_id'] == orig_id].copy()

            # 获取该个体的后验概率
            orig_id_idx = np.where(individual_ids == orig_id)[0][0]
            probs = posterior_probs[orig_id_idx, :]

            # 基于后验概率采样类型（虽然不直接使用，但保持数据结构一致性）
            # assigned_type = np.random.choice(n_types, p=probs)

            # 更新individual_id为新的ID
            individual_data['individual_id'] = new_id

            bootstrap_data_list.append(individual_data)

        return pd.concat(bootstrap_data_list, ignore_index=True)

    def run_bootstrap_iteration(b: int) -> Dict[str, float]:
        """
        运行单次bootstrap迭代

        Returns:
            Dict[str, float]: 该次bootstrap估计的参数值
        """
        try:
            # 使用print而非logger，这样输出会立即显示到控制台
            print(f"\n{'='*60}", flush=True)
            print(f"Bootstrap 迭代 {b+1}/{n_bootstrap}", flush=True)
            print(f"{'='*60}", flush=True)

            # 生成bootstrap样本
            bootstrap_data = generate_bootstrap_sample(b)
            n_individuals = len(bootstrap_data['individual_id'].unique())
            n_obs = len(bootstrap_data)
            print(f"Bootstrap样本生成完成: {n_individuals}个体, {n_obs}条观测", flush=True)

            # 运行EM算法（使用原始参数作为初始值以加快收敛）
            from src.estimation.em_with_omega import run_em_algorithm_with_omega
            
            print(f"开始EM估计（最大{max_em_iterations}轮迭代）...", flush=True)

            results = run_em_algorithm_with_omega(
                observed_data=bootstrap_data,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=beta,
                n_types=n_types,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                max_iterations=max_em_iterations,
                tolerance=em_tolerance,
                n_choices=estimated_params['n_choices'],
                use_migration_behavior_init=False,
                initial_params=estimated_params.copy(),  # **使用原始、干净的参数**
                initial_pi_k=type_probabilities.copy()
            )

            bootstrap_params = results["structural_params"]

            # **关键修复**: 检查参数是否真的被更新了
            # 如果参数和初始值几乎一样，说明这一轮EM没有收敛，结果无效
            initial_p_values = np.array(list(estimated_params.values()))
            final_p_values = np.array(list(bootstrap_params.values()))
            param_change = np.sum(np.abs(initial_p_values - final_p_values))

            if param_change < 1e-5:
                print(f"\n警告: Bootstrap 迭代 {b+1} 未能有效收敛 (参数变化极小)，丢弃此次结果。", flush=True)
                return None

            # 打印前5个参数的估计值
            param_sample = list(bootstrap_params.keys())[:5]
            print(f"\n估计完成！示例参数值:", flush=True)
            for param_name in param_sample:
                if param_name != 'n_choices':
                    orig_val = estimated_params.get(param_name, 'N/A')
                    boot_val = bootstrap_params.get(param_name, 'N/A')
                    print(f"  {param_name}: {orig_val:.4f} → {boot_val:.4f}", flush=True)

            return bootstrap_params

        except Exception as e:
            print(f"\n警告: Bootstrap iteration {b+1} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    # 并行或串行运行bootstrap
    if n_jobs != 1:
        if verbose:
            print(f"使用 {n_jobs} 个并行任务...")
        bootstrap_results = Parallel(n_jobs=n_jobs)(
            delayed(run_bootstrap_iteration)(b) for b in range(n_bootstrap)
        )
    else:
        bootstrap_results = [run_bootstrap_iteration(b) for b in range(n_bootstrap)]

    # 过滤失败的迭代
    bootstrap_results = [r for r in bootstrap_results if r is not None]
    n_successful = len(bootstrap_results)

    if n_successful == 0:
        raise RuntimeError("所有Bootstrap迭代都失败了")

    print(f"\n成功完成 {n_successful}/{n_bootstrap} 次Bootstrap迭代", flush=True)

    # 诊断：比较第一个bootstrap结果和原始估计
    if n_successful > 0:
        print("\n" + "="*60, flush=True)
        print("第一个Bootstrap样本的估计结果对比:", flush=True)
        print("="*60, flush=True)
        first_boot = bootstrap_results[0]
        param_sample = [k for k in estimated_params.keys() if k != 'n_choices'][:10]
        for pname in param_sample:
            orig = estimated_params.get(pname, np.nan)
            boot = first_boot.get(pname, np.nan)
            diff = boot - orig
            pct_diff = (diff / orig * 100) if orig != 0 else 0
            print(f"{pname:30s}: {orig:10.6f} → {boot:10.6f} (diff: {diff:+10.6f}, {pct_diff:+6.2f}%)", flush=True)
        print("="*60 + "\n", flush=True)

    # 计算标准误和置信区间
    param_names = [k for k in estimated_params.keys() if k != 'n_choices']
    bootstrap_matrix = np.zeros((n_successful, len(param_names)))

    for i, result in enumerate(bootstrap_results):
        for j, param_name in enumerate(param_names):
            bootstrap_matrix[i, j] = result.get(param_name, np.nan)

    # 诊断：检查参数是否有变化
    print("\n" + "="*60, flush=True)
    print("Bootstrap参数变异性诊断", flush=True)
    print("="*60, flush=True)
    print(f"成功的bootstrap样本数: {n_successful}/{n_bootstrap}", flush=True)
    print(f"参数数量: {len(param_names)}", flush=True)
    print("\n前10个参数的变异性:", flush=True)
    for j, param_name in enumerate(param_names[:10]):  # 检查前10个参数
        values = bootstrap_matrix[:, j]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            param_mean = np.mean(valid_values)
            param_std = np.std(valid_values, ddof=1)
            param_min = np.min(valid_values)
            param_max = np.max(valid_values)
            param_range = param_max - param_min
            orig_val = estimated_params.get(param_name, np.nan)
            print(f"{param_name:30s}: orig={orig_val:8.4f}, mean={param_mean:8.4f}, std={param_std:8.6f}, range={param_range:8.6f}", flush=True)
    print("="*60 + "\n", flush=True)

    # 计算标准误（标准差）
    std_errors = {}
    for j, param_name in enumerate(param_names):
        values = bootstrap_matrix[:, j]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            std_errors[param_name] = np.std(valid_values, ddof=1)
        else:
            std_errors[param_name] = np.nan

    # 计算95%置信区间
    confidence_intervals = {}
    for j, param_name in enumerate(param_names):
        values = bootstrap_matrix[:, j]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            lower = np.percentile(valid_values, 2.5)
            upper = np.percentile(valid_values, 97.5)
            confidence_intervals[param_name] = (lower, upper)
        else:
            confidence_intervals[param_name] = (np.nan, np.nan)

    # 计算t统计量和p值
    t_stats = {}
    p_values = {}
    for param_name in param_names:
        if std_errors[param_name] > 0 and not np.isnan(std_errors[param_name]):
            t_stats[param_name] = estimated_params[param_name] / std_errors[param_name]
            p_values[param_name] = 2 * (1 - norm.cdf(abs(t_stats[param_name])))
        else:
            t_stats[param_name] = np.nan
            p_values[param_name] = np.nan

    if verbose:
        print("\nBootstrap 标准误估计完成！")
        print(f"示例结果（前5个参数）：")
        for i, param_name in enumerate(param_names[:5]):
            point_est = estimated_params[param_name]
            se = std_errors[param_name]
            ci = confidence_intervals[param_name]
            print(f"  {param_name}: {point_est:.4f} (SE: {se:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

    return std_errors, confidence_intervals, t_stats, p_values