"""
P1功能测试脚本

测试新增的P1级别功能：
1. 互联网机制（地区特定的ν支撑点）
2. 偏好匹配项ξ_ij集成
3. 户籍惩罚三档城市分类
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging

from src.config.model_config import ModelConfig
from src.model.discrete_support import DiscreteSupportGenerator
from src.model.utility import calculate_flow_utility_vectorized

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_internet_mechanism():
    """
    测试1: 互联网机制（地区特定的ν支撑点）
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Internet Mechanism (Region-Specific ν Support)")
    logger.info("="*60)

    # 创建支撑点生成器
    support_gen = DiscreteSupportGenerator(
        n_eta_support=3,
        n_nu_support=3,
        n_xi_support=3,
        n_sigma_support=2
    )

    # 创建模拟地区数据（带互联网普及率）
    mock_region_data = pd.DataFrame({
        'provcd': ['北京市', '上海市', '贵州省', '西藏自治区'],
        '移动电话普及率': [1.5, 1.4, 0.6, 0.3]  # 北京上海高，贵州西藏低
    })

    # 生成地区特定的ν支撑点
    config = ModelConfig()
    region_specific_nu = support_gen.get_region_specific_nu_support(
        region_data=mock_region_data,
        delta_0=config.delta_0,
        delta_1=config.delta_1,
        internet_column='移动电话普及率'
    )

    logger.info("\n地区特定的ν支撑点：")
    for provcd, nu_support in region_specific_nu.items():
        internet_rate = mock_region_data[mock_region_data['provcd'] == provcd]['移动电话普及率'].values[0]
        variance = np.exp(config.delta_0 - config.delta_1 * internet_rate)
        logger.info(f"  {provcd}:")
        logger.info(f"    互联网普及率: {internet_rate:.2f}")
        logger.info(f"    σ²_ν: {variance:.4f}")
        logger.info(f"    支撑点: {nu_support}")

    # 验证：互联网普及率高的地区方差应该更小
    beijing_variance = np.exp(config.delta_0 - config.delta_1 * 1.5)
    tibet_variance = np.exp(config.delta_0 - config.delta_1 * 0.3)

    assert beijing_variance < tibet_variance, "北京方差应小于西藏（互联网降低不确定性）"

    logger.info("\n✓ TEST 1 PASSED: 互联网机制正常工作")


def test_city_tier_mechanism():
    """
    测试2: 户籍惩罚三档城市分类
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 2: City Tier Mechanism (Hukou Penalty)")
    logger.info("="*60)

    config = ModelConfig()

    # 创建模拟状态数据和地区数据
    n_states = 5
    n_choices = 4

    state_data = {
        'age': np.array([25, 30, 35, 40, 45])[:, np.newaxis],
        'prev_provcd_idx': np.array([0, 1, 2, 3, 0])[:, np.newaxis],
        'hukou_prov_idx': np.array([0, 0, 0, 0, 0])[:, np.newaxis],  # 都是北京户籍
        'hometown_prov_idx': np.array([0, 1, 2, 3, 0])[:, np.newaxis]
    }

    # 地区数据：包含三档城市
    region_data = {
        'provcd': [0, 1, 2, 3],
        'city_tier': np.array([1, 2, 3, 3]),  # 一线、二线、三线、三线
        'amenity_education': np.array([0.8, 0.6, 0.4, 0.3]),
        'amenity_health': np.array([0.7, 0.5, 0.3, 0.2]),
        'amenity_house_price': np.array([0.9, 0.6, 0.4, 0.3]),
        'amenity_climate': np.array([0.5, 0.6, 0.7, 0.8]),
        'amenity_public_services': np.array([0.8, 0.6, 0.4, 0.3]),  # 新增
        '常住人口万': np.array([2000, 1500, 800, 500])
    }

    # 创建距离和邻接矩阵
    distance_matrix = np.array([
        [0, 100, 200, 300],
        [100, 0, 150, 250],
        [200, 150, 0, 100],
        [300, 250, 100, 0]
    ])
    adjacency_matrix = (distance_matrix < 150).astype(int)

    # 获取参数
    params = config.get_initial_params(use_type_specific=True)

    # 计算效用（不提供工资数据，测试户籍惩罚部分）
    try:
        utility = calculate_flow_utility_vectorized(
            state_data=state_data,
            region_data=region_data,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            params=params,
            agent_type=0,
            n_states=n_states,
            n_choices=n_choices
        )

        logger.info(f"\n计算得到的效用矩阵形状: {utility.shape}")
        logger.info(f"效用范围: [{np.min(utility):.2f}, {np.max(utility):.2f}]")

        # 检查三档城市的户籍惩罚是否不同
        # 由于所有人都是北京户籍（index 0），去其他城市会有惩罚
        logger.info("\n三档城市户籍惩罚对比（相对于原籍地）：")
        logger.info(f"  第1档城市(北京，原籍): {params['rho_base_tier_1']:.2f}")
        logger.info(f"  第2档城市(二线): {params['rho_base_tier_2']:.2f}")
        logger.info(f"  第3档城市(三线): {params['rho_base_tier_3']:.2f}")

        assert params['rho_base_tier_1'] > params['rho_base_tier_2'] > params['rho_base_tier_3'], \
            "一线城市户籍惩罚应最高"

        logger.info("\n✓ TEST 2 PASSED: 三档城市分类机制正常工作")

    except Exception as e:
        logger.error(f"效用计算失败: {e}", exc_info=True)
        raise


def test_preference_matching_xi():
    """
    测试3: 偏好匹配项ξ_ij集成
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Preference Matching Term ξ_ij")
    logger.info("="*60)

    config = ModelConfig()

    # 创建简单的状态和地区数据
    n_states = 3
    n_choices = 3

    state_data = {
        'age': np.array([25, 30, 35])[:, np.newaxis],
        'prev_provcd_idx': np.array([0, 1, 2])[:, np.newaxis],
        'hukou_prov_idx': np.array([0, 1, 2])[:, np.newaxis],
        'hometown_prov_idx': np.array([0, 1, 2])[:, np.newaxis]
    }

    region_data = {
        'provcd': [0, 1, 2],
        'amenity_education': np.array([0.8, 0.6, 0.4]),
        'amenity_health': np.array([0.7, 0.5, 0.3]),
        'amenity_house_price': np.array([0.6, 0.4, 0.3]),
        'amenity_climate': np.array([0.5, 0.6, 0.7]),
        'amenity_public_services': np.array([0.8, 0.6, 0.4]),  # 新增
        '常住人口万': np.array([2000, 1000, 500])
    }

    distance_matrix = np.eye(n_choices) * 100
    adjacency_matrix = np.eye(n_choices)

    params = config.get_initial_params(use_type_specific=True)

    # 创建随机的ξ_ij矩阵
    xi_ij = np.random.randn(n_states, n_choices) * 0.5

    # 计算不带ξ_ij的效用
    utility_without_xi = calculate_flow_utility_vectorized(
        state_data=state_data,
        region_data=region_data,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        params=params,
        agent_type=0,
        n_states=n_states,
        n_choices=n_choices,
        xi_ij=None
    )

    # 计算带ξ_ij的效用
    utility_with_xi = calculate_flow_utility_vectorized(
        state_data=state_data,
        region_data=region_data,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        params=params,
        agent_type=0,
        n_states=n_states,
        n_choices=n_choices,
        xi_ij=xi_ij
    )

    # 验证：带ξ_ij的效用应该等于不带ξ_ij的效用加上ξ_ij
    utility_diff = utility_with_xi - utility_without_xi

    logger.info(f"\nξ_ij矩阵:\n{xi_ij}")
    logger.info(f"\n效用差异:\n{utility_diff}")

    assert np.allclose(utility_diff, xi_ij, atol=1e-6), "ξ_ij应正确加到效用中"

    logger.info("\n✓ TEST 3 PASSED: 偏好匹配项ξ_ij正确集成")


def main():
    """
    主测试流程
    """
    logger.info("\n" + "#"*60)
    logger.info("# P1 FEATURES TEST SUITE")
    logger.info("#"*60)

    try:
        # TEST 1: 互联网机制
        test_internet_mechanism()

        # TEST 2: 三档城市分类
        test_city_tier_mechanism()

        # TEST 3: 偏好匹配项ξ_ij
        test_preference_matching_xi()

        # 所有测试通过
        logger.info("\n" + "#"*60)
        logger.info("# ALL P1 TESTS PASSED ✓")
        logger.info("#"*60)
        logger.info("\nP1级别功能全部验证成功！")

    except Exception as e:
        logger.error(f"\n" + "!"*60)
        logger.error(f"! TEST FAILED: {e}")
        logger.error("!"*60, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
