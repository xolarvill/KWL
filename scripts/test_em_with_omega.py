"""
EM-with-ω小样本测试脚本

测试完整的EM-NFXP算法集成离散支撑点的流程
使用极小样本（10个个体）快速验证
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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    主测试流程
    """
    logger.info("\n" + "#"*80)
    logger.info("# EM-with-ω SMALL SAMPLE TEST")
    logger.info("#"*80)

    # 1. 加载配置
    config = ModelConfig()
    config.use_discrete_support = True  # 确保使用离散支撑点
    config.em_max_iterations = 5  # 测试时减少迭代次数
    config.lbfgsb_maxiter = 5     # 减少M-step优化迭代

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
    SAMPLE_SIZE = 10
    logger.info(f"\n[2] Sampling {SAMPLE_SIZE} individuals...")
    unique_individuals = df_individual['individual_id'].unique()[:SAMPLE_SIZE]
    df_individual_sample = df_individual[df_individual['individual_id'].isin(unique_individuals)]

    logger.info(f"  Sample size: {len(df_individual_sample)} observations from {SAMPLE_SIZE} individuals")
    logger.info(f"  State space size: {len(state_space)}")
    logger.info(f"  Regions: {len(df_region)}")

    # 4. 创建支撑点生成器（简化版）
    logger.info("\n[3] Creating discrete support generator...")
    support_gen = DiscreteSupportGenerator(
        n_eta_support=3,   # 极简化：3个支撑点
        n_nu_support=3,
        n_xi_support=3,
        n_sigma_support=2,
        eta_range=(-1.0, 1.0),
        nu_range=(-0.5, 0.5),
        xi_range=(-0.5, 0.5),
        sigma_range=(0.5, 1.0)
    )

    support_info = support_gen.get_support_info()
    for var_name, info in support_info.items():
        logger.info(f"  {var_name}: {info['n_points']} points")

    # 5. 准备初始参数
    logger.info("\n[4] Preparing initial parameters...")
    initial_params = config.get_initial_params(use_type_specific=True)
    initial_pi_k = np.array([0.4, 0.3, 0.3])  # 初始类型概率

    logger.info(f"  Total parameters: {len(initial_params)}")
    logger.info(f"  Initial type probabilities: {initial_pi_k}")

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
            max_omega_per_individual=500,  # 限制ω组合数
            use_simplified_omega=True,
            lbfgsb_maxiter=config.lbfgsb_maxiter
        )

        # 7. 输出结果
        logger.info("\n" + "#"*80)
        logger.info("# RESULTS")
        logger.info("#"*80)

        logger.info(f"\nConverged: {results['converged']}")
        logger.info(f"Iterations: {results['n_iterations']}")
        logger.info(f"Final log-likelihood: {results['final_log_likelihood']:.4f}")
        logger.info(f"\nFinal type probabilities:")
        for k, prob in enumerate(results['type_probabilities']):
            logger.info(f"  Type {k}: {prob:.4f}")

        logger.info(f"\nSample structural parameters:")
        estimated_params = results['structural_params']
        param_samples = [
            'alpha_w', 'rho_base_tier_1', 'rho_base_tier_2', 'rho_base_tier_3',
            'gamma_0_type_1', 'gamma_0_type_2',
            'alpha_home_type_0', 'alpha_home_type_1', 'alpha_home_type_2'
        ]
        for param_name in param_samples:
            if param_name in estimated_params:
                logger.info(f"  {param_name}: {estimated_params[param_name]:.4f}")

        logger.info("\n" + "#"*80)
        logger.info("# ✓ TEST COMPLETED SUCCESSFULLY")
        logger.info("#"*80)

    except Exception as e:
        logger.error(f"\n" + "!"*80)
        logger.error(f"! TEST FAILED: {e}")
        logger.error("!"*80, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
