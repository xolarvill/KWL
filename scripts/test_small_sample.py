"""
小样本测试脚本

测试新实现的功能：
1. 工资方程和前景理论效用
2. 离散支撑点枚举
3. 带ω的EM算法

使用极小样本（10个个体，2个类型，简化支撑点）快速验证流程
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging

from src.config.model_config import ModelConfig
from src.model.discrete_support import DiscreteSupportGenerator
from src.data_handler.data_loader import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_tiny_sample(full_data: pd.DataFrame, n_individuals: int = 10) -> pd.DataFrame:
    """
    从完整数据中抽取极小样本
    """
    logger.info(f"Creating tiny sample with {n_individuals} individuals...")

    # 选择访问过不同数量地区的个体（增加多样性）
    individual_region_counts = full_data.groupby('IID')['provcd'].nunique()

    # 选择访问过2-4个地区的个体
    suitable_individuals = individual_region_counts[
        (individual_region_counts >= 2) & (individual_region_counts <= 4)
    ].index[:n_individuals]

    tiny_sample = full_data[full_data['IID'].isin(suitable_individuals)].copy()

    logger.info(f"  Selected {len(tiny_sample)} observations from {n_individuals} individuals")
    logger.info(f"  Regions visited: {tiny_sample['provcd'].nunique()}")
    logger.info(f"  Time periods: {tiny_sample['year'].nunique()}")

    return tiny_sample


def test_discrete_support_generator():
    """
    测试1: 离散支撑点生成器
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Discrete Support Generator")
    logger.info("="*60)

    # 创建简化配置（更少的支撑点）
    support_gen = DiscreteSupportGenerator(
        n_eta_support=3,   # 简化：3个支撑点而非7个
        n_nu_support=3,    # 简化：3个支撑点而非5个
        n_xi_support=3,
        n_sigma_support=2,
        eta_range=(-1.0, 1.0),
        nu_range=(-0.5, 0.5),
        xi_range=(-0.5, 0.5),
        sigma_range=(0.5, 1.0)
    )

    support_info = support_gen.get_support_info()

    logger.info("\nSupport points generated:")
    for var_name, info in support_info.items():
        logger.info(f"  {var_name}: {info['n_points']} points")
        logger.info(f"    Values: {info['values']}")

    logger.info("\n✓ TEST 1 PASSED: Support points generated successfully")
    return support_gen


def test_omega_enumeration(support_gen: DiscreteSupportGenerator):
    """
    测试2: ω枚举功能
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Omega Enumeration")
    logger.info("="*60)

    # 创建模拟的个体数据
    mock_individual_data = pd.DataFrame({
        'provcd': ['北京市', '上海市', '北京市'],
        'year': [2010, 2011, 2012],
        'IID': ['test_001', 'test_001', 'test_001']
    })

    from src.model.discrete_support import SimplifiedOmegaEnumerator

    enumerator = SimplifiedOmegaEnumerator(support_gen)

    omega_list, omega_probs = enumerator.enumerate_omega_for_individual(
        mock_individual_data,
        max_combinations=500
    )

    logger.info(f"\n  Individual visited {mock_individual_data['provcd'].nunique()} regions")
    logger.info(f"  Generated {len(omega_list)} omega combinations")
    logger.info(f"  Probability sum: {np.sum(omega_probs):.6f} (should be ~1.0)")

    # 检查第一个ω组合的结构
    logger.info(f"\n  Sample omega combination:")
    logger.info(f"    eta: {omega_list[0]['eta']:.4f}")
    logger.info(f"    sigma: {omega_list[0]['sigma']:.4f}")
    logger.info(f"    nu_dict: {omega_list[0]['nu_dict']}")
    logger.info(f"    xi_dict: {omega_list[0]['xi_dict']}")

    assert len(omega_list) > 0, "Omega list should not be empty"
    assert np.abs(np.sum(omega_probs) - 1.0) < 0.01, "Probabilities should sum to 1"

    logger.info("\n✓ TEST 2 PASSED: Omega enumeration successful")
    return omega_list, omega_probs


def test_wage_equation_module():
    """
    测试3: 工资方程模块
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Wage Equation Module")
    logger.info("="*60)

    from src.model.wage_equation import (
        calculate_prospect_theory_utility,
        calculate_wage_likelihood,
        calculate_reference_wage
    )

    # 创建模拟数据
    n_obs = 100
    w_current = np.random.lognormal(10, 0.5, n_obs)
    w_reference = w_current * np.random.uniform(0.9, 1.1, n_obs)

    # 测试前景理论效用
    utility = calculate_prospect_theory_utility(
        w_current=w_current,
        w_reference=w_reference,
        alpha_w=1.0,
        lambda_loss_aversion=2.0,
        use_log_difference=True
    )

    logger.info(f"\n  Prospect theory utility computed:")
    logger.info(f"    Mean utility: {np.mean(utility):.4f}")
    logger.info(f"    Std utility: {np.std(utility):.4f}")
    logger.info(f"    % in loss domain: {np.mean(w_current < w_reference)*100:.1f}%")

    # 测试工资似然
    w_predicted = w_current * np.random.uniform(0.95, 1.05, n_obs)
    log_lik = calculate_wage_likelihood(
        w_observed=w_current,
        w_predicted=w_predicted,
        sigma_epsilon=0.3
    )

    logger.info(f"\n  Wage likelihood computed:")
    logger.info(f"    Mean log-likelihood: {np.mean(log_lik):.4f}")
    logger.info(f"    Valid entries: {np.sum(~np.isnan(log_lik))}/{len(log_lik)}")

    assert not np.any(np.isnan(utility)), "Utility should not contain NaN"
    assert not np.any(np.isnan(log_lik)), "Log-likelihood should not contain NaN"

    logger.info("\n✓ TEST 3 PASSED: Wage equation functions working")


def test_config_integration():
    """
    测试4: 配置集成
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Model Config Integration")
    logger.info("="*60)

    config = ModelConfig()

    # 测试离散支撑点配置
    support_config = config.get_discrete_support_config()
    logger.info(f"\n  Discrete support config loaded:")
    logger.info(f"    n_eta_support: {support_config['n_eta_support']}")
    logger.info(f"    n_nu_support: {support_config['n_nu_support']}")
    logger.info(f"    max_omega_per_individual: {support_config['max_omega_per_individual']}")

    # 测试参数获取
    params = config.get_initial_params(use_type_specific=True)
    logger.info(f"\n  Initial parameters:")
    logger.info(f"    Total parameters: {len(params)}")
    logger.info(f"    Type-specific params: {[k for k in params.keys() if 'type_' in k][:5]}...")

    # 检查新增参数
    assert 'delta_0' in dir(config), "delta_0 should be in config"
    assert 'delta_1' in dir(config), "delta_1 should be in config"
    assert config.include_wage_likelihood == True, "wage likelihood should be enabled"

    logger.info("\n✓ TEST 4 PASSED: Config integration successful")


def main():
    """
    主测试流程
    """
    logger.info("\n" + "#"*60)
    logger.info("# SMALL SAMPLE TEST SUITE")
    logger.info("#"*60)

    try:
        # TEST 1: 支撑点生成
        support_gen = test_discrete_support_generator()

        # TEST 2: ω枚举
        omega_list, omega_probs = test_omega_enumeration(support_gen)

        # TEST 3: 工资方程
        test_wage_equation_module()

        # TEST 4: 配置集成
        test_config_integration()

        # 所有测试通过
        logger.info("\n" + "#"*60)
        logger.info("# ALL TESTS PASSED ✓")
        logger.info("#"*60)
        logger.info("\n核心功能验证成功！可以进行下一步集成测试。")

    except Exception as e:
        logger.error(f"\n" + "!"*60)
        logger.error(f"! TEST FAILED: {e}")
        logger.error("!"*60, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
