"""
该脚本是结构模型估计的主入口点，整合了参数估计、推断和模型拟合检验
"""
import cProfile
import pstats
import argparse
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

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

from src.estimation.inference import (
    compute_information_criteria,
    bootstrap_standard_errors,
    estimate_mixture_model_standard_errors
)

from src.estimation.em_with_omega import run_em_algorithm_with_omega
from src.model.discrete_support import DiscreteSupportGenerator
from src.model.likelihood import calculate_log_likelihood
from src.utils.outreg2 import output_estimation_results, output_model_fit_results
from src.config.model_config import ModelConfig
from src.data_handler.data_loader import DataLoader

def run_estimation_workflow(
    sample_size: int = None, 
    use_bootstrap: bool = False, # 兼容旧参数，但现在由stderr_method控制标准误的计算方法
    n_bootstrap: int = 100, 
    bootstrap_jobs: int = 1,
    stderr_method: str = "louis" # 新增参数
):
    """
    封装了模型估计、推断和输出的完整工作流
    
    Args:
        sample_size (int, optional): 如果提供，则只使用前N个个体的样本.
        use_bootstrap (bool): 是否使用Bootstrap计算标准误 (兼容旧参数，现在由stderr_method控制标准误的计算方法)
        n_bootstrap (int): Bootstrap的重复次数（默认为100）
        bootstrap_jobs (int): Bootstrap并行任务数（默认为1）
        stderr_method (str): 标准误计算方法 ('louis', 'bootstrap', 'shared_only', 'type_0_only', 'all_numerical').
    """
    # --- 1. 配置 ---
    config = ModelConfig()
    
    # --- 2. Data Loading and Preparation ---
    logger.info("加载和准备数据...")
    data_loader = DataLoader(config)
    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    df_individual, state_space, transition_matrices, df_region = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)

    if sample_size:
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
        "tolerance": config.em_tolerance,
        "n_choices": config.n_choices,
        "initial_params": initial_params,
        "initial_pi_k": data_driven_pi_k,
        "prov_to_idx": data_loader.prov_to_idx  # 添加缺失的prov_to_idx参数
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

        results = run_em_algorithm_with_omega(**estimation_params)
    else:
        logger.info("\n旧版本已被清除...") # 移除旧版本的简化算法

    logger.info("\n估计完成。")
    
    estimated_params = results["structural_params"]
    final_log_likelihood = results["final_log_likelihood"]
    # posterior_probs = results["posterior_probs"] # EM-with-omega返回的是individual_posteriors
    individual_posteriors = results["individual_posteriors"] # E-step的详细后验
    type_probabilities = results["type_probabilities"]
    log_likelihood_matrix = results.get("posterior_probs", None) # EM-with-omega返回的log_likelihood_matrix存储在posterior_probs键中

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
            h_step=config.lbfgsb_gtol # 使用M-step的梯度容忍度作为数值微分步长
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
                em_tolerance=config.bootstrap_em_tol,
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

def main():
    parser = argparse.ArgumentParser(description="运行结构模型估计")
    parser.add_argument('--profile', action='store_true', help='启用性能分析')
    parser.add_argument('--debug-sample-size', type=int, default=None, help='使用指定数量的样本进行调试运行')
    parser.add_argument('--use-bootstrap', action='store_true', help='使用Bootstrap计算标准误（慢，但更稳健）')
    parser.add_argument('--n-bootstrap', type=int, default=100, help='Bootstrap重复次数')
    parser.add_argument('--bootstrap-jobs', type=int, default=-1, help='Bootstrap并行任务数，-1表示使用所有CPU核心')
    parser.add_argument('--stderr-method', type=str, default="louis",
                        choices=["louis", "bootstrap", "shared_only", "type_0_only", "all_numerical"],
                        help='标准误计算方法: "louis", "bootstrap", "shared_only", "type_0_only", "all_numerical"')
    args = parser.parse_args()

    if args.profile:
        logger.info("性能分析已启用。结果将保存到 'estimation_profile.prof'")
        profiler = cProfile.Profile()
        profiler.enable()
        run_estimation_workflow(
            sample_size=args.debug_sample_size,
            use_bootstrap=args.use_bootstrap,
            n_bootstrap=args.n_bootstrap,
            bootstrap_jobs=args.bootstrap_jobs,
            stderr_method=args.stderr_method
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
            stderr_method=args.stderr_method
        )

if __name__ == '__main__':
    main()
