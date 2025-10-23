"""
调试M-step问题的脚本

用于检查为什么参数不更新
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging

from src.config.model_config import ModelConfig
from src.data_handler.data_loader import DataLoader
from src.model.discrete_support import DiscreteSupportGenerator
from src.estimation.em_with_omega import run_em_algorithm_with_omega

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    主调试流程
    """
    logger.info("\n" + "#"*60)
    logger.info("# DEBUG M-STEP ISSUE")
    logger.info("#"*60)

    # 1. 加载配置
    config = ModelConfig()
    config.use_discrete_support = True
    config.em_max_iterations = 1
    config.lbfgsb_maxiter = 5

    # 2. 加载数据
    logger.info("\n[1] Loading data...")
    data_loader = DataLoader(config)

    try:
        distance_matrix = data_loader.load_distance_matrix()
        adjacency_matrix = data_loader.load_adjacency_matrix()
        df_individual, state_space, transition_matrices, df_region = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    # 3. 抽取极小样本
    SAMPLE_SIZE = 3
    logger.info(f"\n[2] Sampling {SAMPLE_SIZE} individuals...")
    unique_individuals = df_individual['individual_id'].unique()[:SAMPLE_SIZE]
    df_individual_sample = df_individual[df_individual['individual_id'].isin(unique_individuals)]

    logger.info(f"  Sample size: {len(df_individual_sample)} observations from {SAMPLE_SIZE} individuals")

    # 4. 创建支撑点生成器（极简版）
    logger.info("\n[3] Creating discrete support generator...")
    support_gen = DiscreteSupportGenerator(
        n_eta_support=2,
        n_nu_support=2,
        n_xi_support=2,
        n_sigma_support=2,
        eta_range=(-0.5, 0.5),
        nu_range=(-0.3, 0.3),
        xi_range=(-0.3, 0.3),
        sigma_range=(0.8, 1.2)
    )

    # 5. 准备初始参数
    logger.info("\n[4] Preparing initial parameters...")
    initial_params = config.get_initial_params(use_type_specific=True)
    initial_pi_k = np.array([0.4, 0.3, 0.3])

    logger.info(f"  Total parameters: {len(initial_params)}")
    
    # 打印一些关键参数
    logger.info("  Key initial parameters:")
    key_params = ['alpha_w', 'gamma_0_type_1', 'gamma_0_type_2']
    for param in key_params:
        if param in initial_params:
            logger.info(f"    {param}: {initial_params[param]}")

    # 6. 运行EM-with-ω算法
    logger.info("\n[5] Running EM-with-ω algorithm...")

    try:
        results = run_em_algorithm_with_omega(
            observed_data=df_individual_sample,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=config.discount_factor,
            n_types=config.em_n_types,
            regions_df=df_region,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_gen,
            prov_to_idx=data_loader.prov_to_idx,
            max_iterations=config.em_max_iterations,
            tolerance=config.em_tolerance,
            n_choices=config.n_choices,
            initial_params=initial_params,
            initial_pi_k=initial_pi_k,
            max_omega_per_individual=30,
            use_simplified_omega=True,
            lbfgsb_maxiter=config.lbfgsb_maxiter
        )

        # 7. 输出结果
        logger.info("\n" + "#"*60)
        logger.info("# RESULTS")
        logger.info("#"*60)

        logger.info(f"\nConverged: {results['converged']}")
        logger.info(f"Iterations: {results['n_iterations']}")
        logger.info(f"Final log-likelihood: {results['final_log_likelihood']:.4f}")
        logger.info(f"\nFinal type probabilities:")
        for k, prob in enumerate(results['type_probabilities']):
            logger.info(f"  Type {k}: {prob:.4f}")

        logger.info(f"\nKey structural parameters:")
        estimated_params = results['structural_params']
        for param in key_params:
            if param in estimated_params:
                logger.info(f"  {param}: {estimated_params[param]:.4f}")

        logger.info("\n" + "#"*60)
        logger.info("# ✓ DEBUG COMPLETED")
        logger.info("#"*60)

    except Exception as e:
        logger.error(f"\n" + "!"*60)
        logger.error(f"! DEBUG FAILED: {e}")
        logger.error("!"*60, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()