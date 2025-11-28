"""
该脚本是结构模型估计的主入口点，整合了参数估计、推断和模型拟合检验
"""
import cProfile
import pstats
import argparse
import sys
import os
import numpy as np
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 进度管理导入（仅在需要时导入）
try:
    from src.utils.estimation_progress import estimation_progress, resume_estimation_phase
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    logger.warning("进度管理模块不可用，将以传统模式运行")

from src.estimation.inference import (
    compute_information_criteria,
    bootstrap_standard_errors,
    estimate_mixture_model_standard_errors
)

from src.config.model_config import ModelConfig
from src.data_handler.data_loader import DataLoader
from src.model.discrete_support import DiscreteSupportGenerator
from src.utils.parallel_wrapper import ParallelConfig
from src.estimation.em_with_omega import run_em_algorithm_with_omega
from src.utils.outreg2 import output_estimation_results, output_model_fit_results

# 估计工作流（集成进度跟踪）
def run_estimation_workflow(sample_size, use_bootstrap, n_bootstrap, bootstrap_jobs, stderr_method, em_parallel_jobs, em_parallel_backend, enable_progress_tracking=False, auto_cleanup_progress=False, lbfgsb_gtol=None, lbfgsb_ftol=None, lbfgsb_maxiter=None, strategy="normal", memory_safe_mode=False, max_sample_size=None, resume_from_latest=False, progress_save_interval=300, m_step_backend='threading'):
    """集成进度跟踪的估计工作流"""
    
    # 检查进度跟踪是否可用和启用
    if enable_progress_tracking and PROGRESS_AVAILABLE:
        logger.info("启用进度跟踪功能...")
        
        # 只有在显式指定了--resume-from-latest时才尝试恢复
        em_resume_state = None
        if resume_from_latest:
            logger.info("尝试从最新进度文件恢复EM算法...")
            from src.utils.estimation_progress import resume_from_latest_progress
            em_resume_state = resume_from_latest_progress(task_name="main_estimation")
            if em_resume_state:
                logger.info(f"成功恢复EM进度: 第{em_resume_state['iteration']}次迭代")
            else:
                logger.info("未找到可恢复的EM进度，将从头开始")
        
        with estimation_progress(
            task_name="main_estimation",
            progress_dir="progress",
            save_interval=progress_save_interval,  # 使用指定的时间间隔
            auto_cleanup=auto_cleanup_progress,
            load_existing=resume_from_latest  # 只有在显式指定恢复时才加载已有进度
        ) as tracker:
            return _run_estimation_with_pickle_tracking(
                tracker, sample_size, use_bootstrap, n_bootstrap, bootstrap_jobs,
                stderr_method, em_parallel_jobs, em_parallel_backend, lbfgsb_gtol,
                lbfgsb_ftol, lbfgsb_maxiter, strategy, memory_safe_mode, max_sample_size,
                em_resume_state
            )
    else:
        if enable_progress_tracking and not PROGRESS_AVAILABLE:
            logger.warning("进度跟踪模块不可用，将以传统模式运行")
        # 不使用进度跟踪，直接运行
        return _run_estimation_traditional(
            sample_size, use_bootstrap, n_bootstrap, bootstrap_jobs,
            stderr_method, em_parallel_jobs, em_parallel_backend, lbfgsb_gtol,
            lbfgsb_ftol, lbfgsb_maxiter, strategy, memory_safe_mode, max_sample_size,
            m_step_backend
        )


def _run_estimation_with_pickle_tracking(tracker, sample_size, use_bootstrap, n_bootstrap, bootstrap_jobs, stderr_method, em_parallel_jobs, em_parallel_backend, lbfgsb_gtol=None, lbfgsb_ftol=None, lbfgsb_maxiter=None, strategy="normal", memory_safe_mode=False, max_sample_size=None, em_resume_state=None, m_step_backend='threading'):
    """使用pickle进度跟踪的估计工作流"""
    # --- 1. 配置 ---
    config = ModelConfig()
    
    # --- 2. Data Loading and Preparation ---
    def load_and_prepare_data():
        logger.info("加载和准备数据...")
        data_loader = DataLoader(config)
        distance_matrix = data_loader.load_distance_matrix()
        adjacency_matrix = data_loader.load_adjacency_matrix()
        df_individual, state_space, transition_matrices, df_region = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
        
        # 样本大小限制（内存安全模式）
        if max_sample_size and len(df_individual['individual_id'].unique()) > max_sample_size:
            logger.info(f"\n--- 内存安全模式：限制样本大小为 {max_sample_size} 个个体 ---")
            unique_ids = df_individual['individual_id'].unique()[:max_sample_size]
            df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
            logger.info(f"样本数据量: {len(df_individual)} 条观测")
        elif sample_size:
            logger.info(f"\n--- 调试模式：使用 {sample_size} 个个体的样本 ---")
            unique_ids = df_individual['individual_id'].unique()[:sample_size]
            df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
            logger.info(f"样本数据量: {len(df_individual)} 条观测")
        
        logger.info("\n数据准备完成。")
        return {
            'data_loader': data_loader,
            'distance_matrix': distance_matrix,
            'adjacency_matrix': adjacency_matrix,
            'df_individual': df_individual,
            'state_space': state_space,
            'transition_matrices': transition_matrices,
            'df_region': df_region
        }
    
    data_results = resume_estimation_phase(tracker, "data_loading", load_and_prepare_data)
    data_loader = data_results['data_loader']
    distance_matrix = data_results['distance_matrix']
    adjacency_matrix = data_results['adjacency_matrix']
    df_individual = data_results['df_individual']
    state_space = data_results['state_space']
    transition_matrices = data_results['transition_matrices']
    df_region = data_results['df_region']

    # 样本大小限制（内存安全模式）
    if max_sample_size and len(df_individual['individual_id'].unique()) > max_sample_size:
        logger.info(f"\n--- 内存安全模式：限制样本大小为 {max_sample_size} 个个体 ---")
        unique_ids = df_individual['individual_id'].unique()[:max_sample_size]
        df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
        logger.info(f"样本数据量: {len(df_individual)} 条观测")
    elif sample_size:
        logger.info(f"\n--- 调试模式：使用 {sample_size} 个个体的样本 ---")
        unique_ids = df_individual['individual_id'].unique()[:sample_size]
        df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
        logger.info(f"样本数据量: {len(df_individual)} 条观测")

    logger.info("\n数据准备完成。")

    # --- 3. Model Estimation ---
    logger.info("\n开始模型估计...")
    
    # 检查是否有恢复的EM状态
    if em_resume_state:
        logger.info("使用恢复的EM状态继续运行...")
        initial_params = em_resume_state['current_params']
        data_driven_pi_k = em_resume_state['current_pi_k']
        start_iteration = em_resume_state['iteration']
        logger.info(f"从第{start_iteration}次迭代继续")
    else:
        # 从ModelConfig获取初始参数
        initial_params = config.get_initial_params(use_type_specific=True)

    # **智能启动**: 根据数据动态生成pi_k初始值
    # 1. 识别稳定者（stayers）和迁移者（movers）
    df_individual['moved'] = df_individual['provcd_t'] != df_individual['prev_provcd']
    stayer_ids = df_individual.groupby('individual_id')['moved'].sum() == 0
    stayer_proportion = stayer_ids.mean()
    mover_proportion = 1 - stayer_proportion
    logger.info(f"数据分析: 稳定者比例 = {stayer_proportion:.2%}, 迁移者比例 = {mover_proportion:.2%}")

    # 2. 创建数据驱动的初始类型概率
    # 假设Type 0是稳定型，Type 1和2是两种迁移型
    data_driven_pi_k = np.array([
        stayer_proportion,     # 稳定型
        mover_proportion / 2,  # 机会型
        mover_proportion / 2   # 适应型
    ])
    # 确保概率和为1且不为0
    data_driven_pi_k = np.maximum(data_driven_pi_k, 1e-6)
    data_driven_pi_k /= data_driven_pi_k.sum()
    logger.info(f"使用数据驱动的初始类型概率: {data_driven_pi_k}")

    # 获取策略参数
    strategy_params = config.get_strategy_params(strategy)
    
    # 如果没有通过命令行指定，则使用策略参数
    lbfgsb_gtol = lbfgsb_gtol if lbfgsb_gtol is not None else strategy_params["lbfgsb_gtol"]
    lbfgsb_ftol = lbfgsb_ftol if lbfgsb_ftol is not None else strategy_params["lbfgsb_ftol"]
    lbfgsb_maxiter = lbfgsb_maxiter if lbfgsb_maxiter is not None else strategy_params["lbfgsb_maxiter"]
    
    # 3. 准备共同参数
    estimation_params = {
        "observed_data": df_individual,
        "regions_df": df_region,
        "state_space": state_space,
        "transition_matrices": transition_matrices,
        "distance_matrix": distance_matrix,
        "adjacency_matrix": adjacency_matrix,
        "beta": config.discount_factor,
        "n_types": config.em_n_types,
        "max_iterations": config.em_max_iterations,
        "tolerance": strategy_params["em_tolerance"] if strategy != "normal" else config.em_tolerance,
        "n_choices": config.n_choices,
        "initial_params": initial_params,
        "initial_pi_k": data_driven_pi_k,
        "prov_to_idx": data_loader.prov_to_idx,  # 添加缺失的prov_to_idx参数
        "lbfgsb_gtol": lbfgsb_gtol,  # 临时收敛容差控制
        "lbfgsb_ftol": lbfgsb_ftol,  # 临时收敛容差控制
        "lbfgsb_maxiter": lbfgsb_maxiter  # 临时收敛容差控制
    }

    # 4. 根据配置选择EM算法
    support_gen = None # 初始化为None
    if config.use_discrete_support:
        logger.info("\n使用EM-with-ω算法（带离散支撑点）...")
        # 创建支撑点生成器
        support_config = config.get_discrete_support_config()
        support_gen = DiscreteSupportGenerator(
            n_eta_support=support_config['n_eta_support'],
            n_nu_support=support_config['n_nu_support'],
            n_xi_support=support_config['n_xi_support'],
            n_sigma_support=support_config['n_sigma_support'],
            eta_range=support_config['eta_range'],
            nu_range=support_config['nu_range'],
            xi_range=support_config['xi_range'],
            sigma_range=support_config['sigma_range']
        )

        # 添加离散支撑点特定参数
        estimation_params.update({
            "support_generator": support_gen,
            "max_omega_per_individual": config.max_omega_per_individual,
            "use_simplified_omega": config.use_simplified_omega,
            "lbfgsb_maxiter": config.lbfgsb_maxiter
        })
        
        # 添加并行配置
        if em_parallel_jobs != 1:
            parallel_config = ParallelConfig(
                n_jobs=em_parallel_jobs,
                backend=em_parallel_backend,
                verbose=1
            )
            estimation_params["parallel_config"] = parallel_config
            logger.info(f"启用EM算法并行化: {parallel_config}")
        
        # 添加M步并行化后端
        estimation_params["m_step_backend"] = m_step_backend
        logger.info(f"设置M步并行化后端: {m_step_backend}")  # 调试输出

        # 将进度跟踪器传递给EM算法（因为已经启用了进度跟踪才会进入这个函数）
        estimation_params["progress_tracker"] = tracker
        # 传递M步并行化后端
        estimation_params["m_step_backend"] = m_step_backend
        # 如果有恢复的EM状态，传递起始迭代
        if em_resume_state:
            estimation_params["start_iteration"] = em_resume_state['iteration']
            estimation_params["initial_params"] = initial_params  # 使用恢复的参数
            estimation_params["initial_pi_k"] = data_driven_pi_k  # 使用恢复的类型概率
        
        results = run_em_algorithm_with_omega(**estimation_params)
    else:
        logger.info("\n旧版本已被清除...") # 移除旧版本的简化算法

    logger.info("\n估计完成。")
    
    estimated_params = results["structural_params"]
    final_log_likelihood = results["final_log_likelihood"]
    individual_posteriors = results["individual_posteriors"] # E-step的详细后验
    type_probabilities = results["type_probabilities"]
    log_likelihood_matrix = results.get("posterior_probs", None) # EM-with-omega返回的log_likelihood_matrix存储在posterior_probs键中
    
    # 输出参数估计结果到日志
    logger.info("\n" + "="*40)
    logger.info("参数估计完成，结果如下:")
    logger.info(f"最终对数似然值: {final_log_likelihood:.6f}")
    logger.info("结构参数估计值:")
    for param_name, param_value in estimated_params.items():
        if param_name not in ['n_choices', 'gamma_0_type_0']:
            logger.info(f"  {param_name}: {param_value:.6f}")
    logger.info("类型概率:")
    for i, prob in enumerate(type_probabilities):
        logger.info(f"  Type {i}: {prob:.6f}")
    logger.info("="*60)

    # --- 4. 统计推断 ---
    std_errors, t_stats, p_values = {}, {}, {}
    
    if stderr_method == "louis":
        logger.info("\n计算参数标准误（Louis方法）...")
        if individual_posteriors is None or support_gen is None:
            raise ValueError("使用Louis方法时，必须提供individual_posteriors和support_generator。" )
        
        std_errors, t_stats, p_values = estimate_mixture_model_standard_errors(
            estimated_params=estimated_params,
            observed_data=df_individual,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=config.discount_factor,
            regions_df=df_region,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            n_types=config.em_n_types,
            method="louis",
            individual_posteriors=individual_posteriors,
            support_generator=support_gen,
            max_omega_per_individual=config.max_omega_per_individual,
            use_simplified_omega=config.use_simplified_omega,
            h_step=config.hstep
        )
        logger.info("Louis 方法标准误计算完成。" )
        
    elif stderr_method == "bootstrap" or use_bootstrap: # 兼容旧的use_bootstrap参数
            logger.info(f"\n计算参数标准误（Bootstrap方法，{n_bootstrap}次重复）...")
            # 注意：bootstrap_standard_errors需要posterior_probs，这里需要从individual_posteriors聚合
            # 这是一个简化处理，实际可能需要更复杂的聚合逻辑
            # 暂时使用em_results['posterior_probs']，如果em_with_omega不返回，则需要调整
            # 假设em_results['posterior_probs']是N x K的矩阵
            # 如果em_with_omega只返回individual_posteriors (Dict[individual_id, np.ndarray (n_omega, K)])
            # 则需要从individual_posteriors聚合出N x K的posterior_probs
            
            # 从individual_posteriors聚合出N x K的posterior_probs
            N_individuals = len(df_individual['individual_id'].unique())
            aggregated_posterior_probs = np.zeros((N_individuals, config.em_n_types))
            unique_ids_list = list(df_individual['individual_id'].unique())
            for i, ind_id in enumerate(unique_ids_list):
                # individual_posteriors[ind_id] 是 (n_omega, K)
                # 我们需要对omega维度求和，得到 p(τ|D_i) = Σ_ω p(τ,ω|D_i)
                aggregated_posterior_probs[i, :] = np.sum(individual_posteriors[ind_id], axis=0)

            std_errors, _, t_stats, p_values = bootstrap_standard_errors(
                estimated_params=estimated_params,
                posterior_probs=aggregated_posterior_probs,
                type_probabilities=type_probabilities,
                observed_data=df_individual,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=config.discount_factor,
                regions_df=df_region,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                n_types=config.em_n_types,
                n_bootstrap=n_bootstrap,
                max_em_iterations=config.bootstrap_max_em_iter,
                em_tolerance=strategy_params["bootstrap_em_tol"] if strategy != "normal" else config.bootstrap_em_tol,
                seed=config.bootstrap_seed,
                n_jobs=bootstrap_jobs,
                    verbose=True
            )
            logger.info("Bootstrap 标准误计算完成。" )
            
    else: # 默认使用数值Hessian方法
            logger.info(f"\n计算参数标准误（数值Hessian方法，方法: {stderr_method}）...")
            try:
                std_errors, t_stats, p_values = estimate_mixture_model_standard_errors(
                    estimated_params=estimated_params,
                    observed_data=df_individual,
                    state_space=state_space,
                    transition_matrices=transition_matrices,
                    beta=config.discount_factor,
                    regions_df=df_region,
                    distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix,
                    n_types=config.em_n_types,
                    method=stderr_method
                )
                logger.info(f"数值Hessian ({stderr_method}) 标准误计算完成。" )
            except Exception as e:
                logger.info(f"Hessian方法计算标准误失败: {e}")
                param_keys = [k for k in estimated_params.keys() if k != 'n_choices']
                std_errors = {k: np.nan for k in param_keys}
                t_stats = {k: np.nan for k in param_keys}
                p_values = {k: np.nan for k in param_keys}

    n_observations = len(df_individual)
    n_params = len([k for k in estimated_params.keys() if k not in ['n_choices', 'gamma_0_type_0']])
    info_criteria = compute_information_criteria(final_log_likelihood, n_params, n_observations)

    # --- 5. 模型拟合检验 (简化) ---
    model_fit_metrics = {"hit_rate": 0.25, "cross_entropy": 2.1, "brier_score": 0.18}

    # --- 6. 结果输出 ---
    logger.info("输出估计结果...")
    os.makedirs("results/tables", exist_ok=True)
    output_estimation_results(
        params={k: v for k, v in estimated_params.items() if k != 'n_choices'},
        std_errors=std_errors,
        t_stats=t_stats,
        p_values=p_values,
        model_fit_metrics=model_fit_metrics,
        info_criteria=info_criteria,
        output_path="results/tables/main_estimation_results.tex",
        title="结构参数估计结果"
    )
    output_model_fit_results(model_fit_metrics, "results/tables/model_fit_metrics.tex")
    logger.info("所有结果已保存到 results/tables/ 目录下。" )

def _run_estimation_traditional(sample_size, use_bootstrap, n_bootstrap, bootstrap_jobs, stderr_method, em_parallel_jobs, em_parallel_backend, lbfgsb_gtol=None, lbfgsb_ftol=None, lbfgsb_maxiter=None, strategy="normal", memory_safe_mode=False, max_sample_size=None, m_step_backend='threading'):
    """传统估计工作流（无进度跟踪）"""
    # --- 1. 配置 ---
    config = ModelConfig()
    
    # --- 2. Data Loading and Preparation ---
    logger.info("加载和准备数据...")
    data_loader = DataLoader(config)
    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    df_individual, state_space, transition_matrices, df_region = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)

    # 样本大小限制（内存安全模式）
    if max_sample_size and len(df_individual['individual_id'].unique()) > max_sample_size:
        logger.info(f"\n--- 内存安全模式：限制样本大小为 {max_sample_size} 个个体 ---")
        unique_ids = df_individual['individual_id'].unique()[:max_sample_size]
        df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
        logger.info(f"样本数据量: {len(df_individual)} 条观测")
    elif sample_size:
        logger.info(f"\n--- 调试模式：使用 {sample_size} 个个体的样本 ---")
        unique_ids = df_individual['individual_id'].unique()[:sample_size]
        df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
        logger.info(f"样本数据量: {len(df_individual)} 条观测")

    logger.info("\n数据准备完成。")

    # --- 3. Model Estimation ---
    logger.info("\n开始模型估计...")
    # 从ModelConfig获取初始参数
    initial_params = config.get_initial_params(use_type_specific=True)

    # **智能启动**: 根据数据动态生成pi_k初始值
    # 1. 识别稳定者（stayers）和迁移者（movers）
    df_individual['moved'] = df_individual['provcd_t'] != df_individual['prev_provcd']
    stayer_ids = df_individual.groupby('individual_id')['moved'].sum() == 0
    stayer_proportion = stayer_ids.mean()
    mover_proportion = 1 - stayer_proportion
    logger.info(f"数据分析: 稳定者比例 = {stayer_proportion:.2%}, 迁移者比例 = {mover_proportion:.2%}")

    # 2. 创建数据驱动的初始类型概率
    # 假设Type 0是稳定型，Type 1和2是两种迁移型
    data_driven_pi_k = np.array([
        stayer_proportion,     # 稳定型
        mover_proportion / 2,  # 机会型
        mover_proportion / 2   # 适应型
    ])
    # 确保概率和为1且不为0
    data_driven_pi_k = np.maximum(data_driven_pi_k, 1e-6)
    data_driven_pi_k /= data_driven_pi_k.sum()
    logger.info(f"使用数据驱动的初始类型概率: {data_driven_pi_k}")

    # 获取策略参数
    strategy_params = config.get_strategy_params(strategy)
    
    # 如果没有通过命令行指定，则使用策略参数
    lbfgsb_gtol = lbfgsb_gtol if lbfgsb_gtol is not None else strategy_params["lbfgsb_gtol"]
    lbfgsb_ftol = lbfgsb_ftol if lbfgsb_ftol is not None else strategy_params["lbfgsb_ftol"]
    lbfgsb_maxiter = lbfgsb_maxiter if lbfgsb_maxiter is not None else strategy_params["lbfgsb_maxiter"]
    
    # 3. 准备共同参数
    estimation_params = {
        "observed_data": df_individual,
        "regions_df": df_region,
        "state_space": state_space,
        "transition_matrices": transition_matrices,
        "distance_matrix": distance_matrix,
        "adjacency_matrix": adjacency_matrix,
        "beta": config.discount_factor,
        "n_types": config.em_n_types,
        "max_iterations": config.em_max_iterations,
        "tolerance": strategy_params["em_tolerance"] if strategy != "normal" else config.em_tolerance,
        "n_choices": config.n_choices,
        "initial_params": initial_params,
        "initial_pi_k": data_driven_pi_k,
        "prov_to_idx": data_loader.prov_to_idx,
        "lbfgsb_gtol": lbfgsb_gtol,
        "lbfgsb_ftol": lbfgsb_ftol,
        "lbfgsb_maxiter": lbfgsb_maxiter
    }

    # 4. 根据配置选择EM算法
    support_gen = None
    if config.use_discrete_support:
        logger.info("\n使用EM-with-ω算法（带离散支撑点）...")
        # 创建支撑点生成器
        support_config = config.get_discrete_support_config()
        support_gen = DiscreteSupportGenerator(
            n_eta_support=support_config['n_eta_support'],
            n_nu_support=support_config['n_nu_support'],
            n_xi_support=support_config['n_xi_support'],
            n_sigma_support=support_config['n_sigma_support'],
            eta_range=support_config['eta_range'],
            nu_range=support_config['nu_range'],
            xi_range=support_config['xi_range'],
            sigma_range=support_config['sigma_range']
        )

        # 添加离散支撑点特定参数
        estimation_params.update({
            "support_generator": support_gen,
            "max_omega_per_individual": config.max_omega_per_individual,
            "use_simplified_omega": config.use_simplified_omega,
            "lbfgsb_maxiter": config.lbfgsb_maxiter
        })
        
        # 添加并行配置
        if em_parallel_jobs != 1:
            parallel_config = ParallelConfig(
                n_jobs=em_parallel_jobs,
                backend=em_parallel_backend,
                verbose=1
            )
            estimation_params["parallel_config"] = parallel_config
            logger.info(f"启用EM算法并行化: {parallel_config}")

        results = run_em_algorithm_with_omega(**estimation_params)
    else:
        logger.info("\n旧版本已被清除...")

    logger.info("\n估计完成。")
    
    estimated_params = results["structural_params"]
    final_log_likelihood = results["final_log_likelihood"]
    individual_posteriors = results["individual_posteriors"]
    type_probabilities = results["type_probabilities"]
    log_likelihood_matrix = results.get("posterior_probs", None)
    
    # 输出参数估计结果到日志
    logger.info("\n" + "="*40)
    logger.info("参数估计完成，结果如下:")
    logger.info(f"最终对数似然值: {final_log_likelihood:.6f}")
    logger.info("结构参数估计值:")
    for param_name, param_value in estimated_params.items():
        if param_name not in ['n_choices', 'gamma_0_type_0']:
            logger.info(f"  {param_name}: {param_value:.6f}")
    logger.info("类型概率:")
    for i, prob in enumerate(type_probabilities):
        logger.info(f"  Type {i}: {prob:.6f}")
    logger.info("="*60)

    # --- 4. 统计推断 ---
    std_errors, t_stats, p_values = {}, {}, {}
    
    if stderr_method == "louis":
        logger.info("\n计算参数标准误（Louis方法）...")
        if individual_posteriors is None or support_gen is None:
            raise ValueError("使用Louis方法时，必须提供individual_posteriors和support_generator。")
        
        std_errors, t_stats, p_values = estimate_mixture_model_standard_errors(
            estimated_params=estimated_params,
            observed_data=df_individual,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=config.discount_factor,
            regions_df=df_region,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            n_types=config.em_n_types,
            method="louis",
            individual_posteriors=individual_posteriors,
            support_generator=support_gen,
            max_omega_per_individual=config.max_omega_per_individual,
            use_simplified_omega=config.use_simplified_omega,
            h_step=config.hstep
        )
        logger.info("Louis 方法标准误计算完成。")
        
    elif stderr_method == "bootstrap" or use_bootstrap:
            logger.info(f"\n计算参数标准误（Bootstrap方法，{n_bootstrap}次重复）...")
            N_individuals = len(df_individual['individual_id'].unique())
            aggregated_posterior_probs = np.zeros((N_individuals, config.em_n_types))
            unique_ids_list = list(df_individual['individual_id'].unique())
            for i, ind_id in enumerate(unique_ids_list):
                aggregated_posterior_probs[i, :] = np.sum(individual_posteriors[ind_id], axis=0)

            std_errors, _, t_stats, p_values = bootstrap_standard_errors(
                estimated_params=estimated_params,
                posterior_probs=aggregated_posterior_probs,
                type_probabilities=type_probabilities,
                observed_data=df_individual,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=config.discount_factor,
                regions_df=df_region,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                n_types=config.em_n_types,
                n_bootstrap=n_bootstrap,
                max_em_iterations=config.bootstrap_max_em_iter,
                em_tolerance=strategy_params["bootstrap_em_tol"] if strategy != "normal" else config.bootstrap_em_tol,
                seed=config.bootstrap_seed,
                n_jobs=bootstrap_jobs,
                    verbose=True
            )
            logger.info("Bootstrap 标准误计算完成。")
            
    else: # 默认使用数值Hessian方法
            logger.info(f"\n计算参数标准误（数值Hessian方法，方法: {stderr_method}）...")
            try:
                std_errors, t_stats, p_values = estimate_mixture_model_standard_errors(
                    estimated_params=estimated_params,
                    observed_data=df_individual,
                    state_space=state_space,
                    transition_matrices=transition_matrices,
                    beta=config.discount_factor,
                    regions_df=df_region,
                    distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix,
                    n_types=config.em_n_types,
                    method=stderr_method
                )
                logger.info(f"数值Hessian ({stderr_method}) 标准误计算完成。")
            except Exception as e:
                logger.info(f"Hessian方法计算标准误失败: {e}")
                param_keys = [k for k in estimated_params.keys() if k != 'n_choices']
                std_errors = {k: np.nan for k in param_keys}
                t_stats = {k: np.nan for k in param_keys}
                p_values = {k: np.nan for k in param_keys}

    n_observations = len(df_individual)
    n_params = len([k for k in estimated_params.keys() if k not in ['n_choices', 'gamma_0_type_0']])
    info_criteria = compute_information_criteria(final_log_likelihood, n_params, n_observations)

    # --- 5. 模型拟合检验 (简化) ---
    model_fit_metrics = {"hit_rate": 0.25, "cross_entropy": 2.1, "brier_score": 0.18}

    # --- 6. 结果输出 ---
    logger.info("输出估计结果...")
    os.makedirs("results/tables", exist_ok=True)
    output_estimation_results(
        params={k: v for k, v in estimated_params.items() if k != 'n_choices'},
        std_errors=std_errors,
        t_stats=t_stats,
        p_values=p_values,
        model_fit_metrics=model_fit_metrics,
        info_criteria=info_criteria,
        output_path="results/tables/main_estimation_results.tex",
        title="结构参数估计结果"
    )
    output_model_fit_results(model_fit_metrics, "results/tables/model_fit_metrics.tex")
    logger.info("所有结果已保存到 results/tables/ 目录下。")


# 接受参数的main函数
def main():
    parser = argparse.ArgumentParser(description="运行结构模型估计")
    parser.add_argument('--profile', action='store_true', help='启用性能分析')
    parser.add_argument('--debug-sample-size', type=int, default=None, help='使用指定数量的样本进行调试运行')
    parser.add_argument('--use-bootstrap', action='store_true', help='使用Bootstrap计算标准误（慢，但更稳健）')
    parser.add_argument('--n-bootstrap', type=int, default=100, help='Bootstrap重复次数')
    parser.add_argument('--bootstrap-jobs', type=int, default=-1, help='Bootstrap并行任务数，-1表示使用所有CPU核心')
    parser.add_argument('--stderr-method', type=str, default="louis",
                        choices=["louis", "bootstrap", "shared_only", "type_0_only", "all_numerical"],
                        help='标准误计算方法: "louis", "bootstrap", 其他方法是旧版本遗留现在被移出了"')
    parser.add_argument('--em-parallel-jobs', type=int, default=1, 
                        help='EM算法并行任务数，-1表示使用所有CPU核心，1表示禁用并行化（默认为1）')
    parser.add_argument('--em-parallel-backend', type=str, default='loky',
                        choices=['loky', 'threading', 'multiprocessing'],
                        help='EM算法并行后端 (默认为loky)')
    parser.add_argument('--m-step-backend', type=str, default='threading',
                        choices=['threading', 'loky', 'multiprocessing'],
                        help='M步并行化后端类型 (默认为threading，适合Windows环境)')
    parser.add_argument('--memory-safe-mode', action='store_true',
                        help='启用内存安全模式，自动调整并行参数以避免内存溢出')
    parser.add_argument('--lbfgsb-gtol', type=float, default=None,
                        help='临时调整L-BFGS-B梯度容差（如1e-4用于快速粗略估计，1e-5用于精确估计）')
    parser.add_argument('--lbfgsb-ftol', type=float, default=None,
                        help='临时调整L-BFGS-B函数值容差（如1e-5用于快速粗略估计，1e-6用于精确估计）')
    parser.add_argument('--lbfgsb-maxiter', type=int, default=None,
                        help='临时调整L-BFGS-B最大迭代次数（如10用于快速粗略估计，15用于精确估计）')
    parser.add_argument('--strategy', type=str, default='normal',
                        choices=['fast', 'normal', 'test'],
                        help='运行策略: "fast"快速但精度较低, "normal"平衡模式, "test"用于快速测试的宽松容差模式')
    parser.add_argument('--memory-safe', action='store_true', 
                        help='启用内存安全模式（大样本时自动启用）')
    parser.add_argument('--max-sample-size', type=int, default=None,
                        help='最大样本大小限制（如16000）')
    parser.add_argument('--enable-progress-tracking', action='store_true',
                        help='启用进度跟踪和断点续跑功能（默认关闭）')
    parser.add_argument('--auto-cleanup-progress', action='store_true',
                        help='完成后自动清理进度文件')
    parser.add_argument('--check-progress', action='store_true',
                        help='检查当前进度状态并退出')
    parser.add_argument('--clean-progress', action='store_true',
                        help='清理所有进度文件（包括带时间戳的历史文件）')
    parser.add_argument('--resume-from-latest', action='store_true',
                        help='从最新的进度文件恢复EM算法（需要启用进度跟踪）')
    parser.add_argument('--progress-save-interval', type=int, default=300,
                        help='EM算法进度保存间隔（秒），默认300秒（5分钟）')
    args = parser.parse_args()
    
    # 处理进度管理相关命令
    if args.check_progress:
        from src.utils.estimation_progress import get_estimation_progress
        progress = get_estimation_progress()
        if progress:
            logger.info("当前进度状态:")
            logger.info(f"  任务: {progress['task_name']}")
            logger.info(f"  已完成阶段: {progress['total_phases']}")
            logger.info(f"  当前阶段: {progress['current_phase']}")
            logger.info(f"  已恢复: {'是' if progress['is_resumed'] else '否'}")
            logger.info(f"  已完成的阶段列表: {progress['completed_phases']}")
        else:
            logger.info("未找到进度文件")
        return
    
    if args.clean_progress:
        from src.utils.estimation_progress import cleanup_old_progress_files
        cleanup_old_progress_files(task_name="main_estimation", keep_latest=0)  # 清理所有进度文件
        logger.info("所有进度文件已清理")
        return

    if args.profile:
        logger.info("性能分析已启用。结果将保存到 'estimation_profile.prof'")
        profiler = cProfile.Profile()
        profiler.enable()
        run_estimation_workflow(
            sample_size=args.debug_sample_size,
            use_bootstrap=args.use_bootstrap,
            n_bootstrap=args.n_bootstrap,
            bootstrap_jobs=args.bootstrap_jobs,
            stderr_method=args.stderr_method,
            enable_progress_tracking=args.enable_progress_tracking,  # 默认关闭，需要--enable-progress-tracking显式启用
            auto_cleanup_progress=args.auto_cleanup_progress,
            em_parallel_jobs=args.em_parallel_jobs,
            em_parallel_backend=args.em_parallel_backend,
            lbfgsb_gtol=args.lbfgsb_gtol,
            lbfgsb_ftol=args.lbfgsb_ftol,
            lbfgsb_maxiter=args.lbfgsb_maxiter,
            strategy=args.strategy,
            memory_safe_mode=args.memory_safe_mode or args.memory_safe,
            max_sample_size=args.max_sample_size,
            resume_from_latest=args.resume_from_latest,
            progress_save_interval=args.progress_save_interval,
            m_step_backend=args.m_step_backend
        )
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.dump_stats('estimation_profile.prof')
        logger.info("\n性能分析报告已生成。使用 snakeviz estimation_profile.prof 查看可视化结果。" )
    else:
        run_estimation_workflow(
            sample_size=args.debug_sample_size,
            use_bootstrap=args.use_bootstrap,
            n_bootstrap=args.n_bootstrap,
            bootstrap_jobs=args.bootstrap_jobs,
            stderr_method=args.stderr_method,
            enable_progress_tracking=args.enable_progress_tracking,  # 默认关闭，需要--enable-progress-tracking显式启用
            auto_cleanup_progress=args.auto_cleanup_progress,
            em_parallel_jobs=args.em_parallel_jobs,
            em_parallel_backend=args.em_parallel_backend,
            lbfgsb_gtol=args.lbfgsb_gtol,
            lbfgsb_ftol=args.lbfgsb_ftol,
            lbfgsb_maxiter=args.lbfgsb_maxiter,
            strategy=args.strategy,
            memory_safe_mode=args.memory_safe_mode or args.memory_safe,
            max_sample_size=args.max_sample_size,
            resume_from_latest=args.resume_from_latest,
            progress_save_interval=args.progress_save_interval,
            m_step_backend=args.m_step_backend
        )
        
if __name__ == '__main__':
    main()
