#!/usr/bin/env python3
"""
测试流式Louis方法是否正确工作
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.estimation.louis_method import louis_method_standard_errors_streaming
from src.model.discrete_support import DiscreteSupportGenerator
from src.config.model_config import ModelConfig
from src.data_handler.data_loader import DataLoader

def test_streaming_louis():
    """测试流式Louis方法"""
    logger.info("开始测试流式Louis方法...")
    
    # 创建最小测试数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    # 加载数据
    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    df_individual, state_space, transition_matrices, df_region = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    
    # 只取10个个体进行测试
    unique_ids = df_individual['individual_id'].unique()[:10]
    df_test = df_individual[df_individual['individual_id'].isin(unique_ids)].copy()
    
    logger.info(f"测试数据: {len(unique_ids)} 个个体, {len(df_test)} 条观测")
    
    # 创建支撑点生成器
    support_config = config.get_discrete_support_config()
    # 移除不适用于DiscreteSupportGenerator的参数
    support_config_clean = {k: v for k, v in support_config.items() 
                           if k not in ['max_omega_per_individual', 'use_simplified_omega']}
    support_gen = DiscreteSupportGenerator(**support_config_clean)
    
    # 模拟EM结果
    from src.estimation.em_with_omega import run_em_algorithm_with_omega
    
    logger.info("运行EM算法获取参数估计...")
    em_results = run_em_algorithm_with_omega(
        observed_data=df_test,
        state_space=state_space,
        transition_matrices=transition_matrices,
        beta=config.discount_factor,
        n_types=config.em_n_types,
        regions_df=df_region,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        max_iterations=2,  # 快速测试，只运行2轮
        tolerance=config.em_tolerance,
        n_choices=config.n_choices,
        support_generator=support_gen,
        max_omega_per_individual=config.max_omega_per_individual,
        use_simplified_omega=config.use_simplified_omega,
        prov_to_idx=data_loader.prov_to_idx
    )
    
    logger.info("EM算法完成，开始测试流式Louis方法...")
    
    # 预处理regions_df为NumPy格式
    prov_to_idx = {prov_id: idx for idx, prov_id in enumerate(df_region['provcd'].unique())}
    from src.estimation.em_with_omega import _prepare_numpy_region_data
    regions_df_np = _prepare_numpy_region_data(df_region, prov_to_idx)
    
    # 测试流式Louis方法
    logger.info("开始流式Louis标准误计算...")
    
    try:
        std_errors, t_stats, p_values = louis_method_standard_errors_streaming(
            estimated_params=em_results["structural_params"],
            type_probabilities=em_results["type_probabilities"],
            individual_posteriors=em_results["individual_posteriors"],
            observed_data=df_test,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=config.discount_factor,
            regions_df=regions_df_np,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_gen,
            n_types=config.em_n_types,
            prov_to_idx=prov_to_idx,
            max_omega_per_individual=config.max_omega_per_individual,  # 保持完整数量！
            use_simplified_omega=config.use_simplified_omega,
            h_step=1e-4
        )
        
        logger.info("流式Louis方法测试成功！")
        logger.info(f"计算了 {len(std_errors)} 个参数的标准误")
        
        # 显示前几个参数的结果
        param_names = list(std_errors.keys())[:5]
        for name in param_names:
            logger.info(f"  {name}: 估计值={em_results['structural_params'][name]:.4f}, "
                       f"标准误={std_errors[name]:.4f}, t值={t_stats[name]:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"流式Louis方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_louis()
    if success:
        logger.info("✅ 流式Louis方法测试通过！")
        sys.exit(0)
    else:
        logger.info("❌ 流式Louis方法测试失败！")
        sys.exit(1)