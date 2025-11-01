#!/usr/bin/env python3
"""
专门测试大样本下流式Louis方法是否解决了future_v_for_age问题
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 捕获所有警告
warnings.filterwarnings('default')
logging.captureWarnings(True)

# 设置日志 - 专门捕获future_v_for_age警告
class FutureVFilter(logging.Filter):
    def filter(self, record):
        return "future_v_for_age" in record.getMessage()

# 设置详细的日志捕获
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建专门捕获future_v_for_age警告的处理器
future_v_handler = logging.StreamHandler()
future_v_handler.addFilter(FutureVFilter())
future_v_handler.setFormatter(logging.Formatter('%(asctime)s - FUTURE_V_WARNING - %(message)s'))
logger.addHandler(future_v_handler)

# 普通日志处理器
normal_handler = logging.StreamHandler()
normal_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(normal_handler)

from src.estimation.louis_method_streaming import louis_method_standard_errors_streaming
from src.model.discrete_support import DiscreteSupportGenerator
from src.config.model_config import ModelConfig
from src.data_handler.data_loader import DataLoader

def test_large_sample_streaming():
    """测试大样本流式Louis方法"""
    print("开始测试大样本流式Louis方法...")
    print("专门检查是否还有future_v_for_age警告")
    
    # 创建最小测试数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    
    # 加载数据
    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    df_individual, state_space, transition_matrices, df_region = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    
    # 取100个个体进行测试 - 足够触发大样本逻辑
    unique_ids = df_individual['individual_id'].unique()[:100]
    df_test = df_individual[df_individual['individual_id'].isin(unique_ids)].copy()
    
    print(f"测试数据: {len(unique_ids)} 个个体, {len(df_test)} 条观测")
    
    # 创建支撑点生成器
    support_config = config.get_discrete_support_config()
    support_config_clean = {k: v for k, v in support_config.items() 
                           if k not in ['max_omega_per_individual', 'use_simplified_omega']}
    support_gen = DiscreteSupportGenerator(**support_config_clean)
    
    # 模拟EM结果
    from src.estimation.em_with_omega import run_em_algorithm_with_omega
    
    print("运行EM算法获取参数估计...")
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
    
    print("EM算法完成，开始测试流式Louis方法...")
    
    # 预处理regions_df为NumPy格式
    prov_to_idx = {prov_id: idx for idx, prov_id in enumerate(df_region['provcd'].unique())}
    from src.estimation.em_with_omega import _prepare_numpy_region_data
    regions_df_np = _prepare_numpy_region_data(df_region, prov_to_idx)
    
    # 测试流式Louis方法
    print("开始流式Louis标准误计算...")
    print(f"配置: 每个个体 {config.max_omega_per_individual} 个omega（完整数量）")
    
    try:
        # 重定向日志以捕获警告
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        
        louis_logger = logging.getLogger('src.estimation.louis_method_streaming')
        louis_logger.addHandler(handler)
        louis_logger.setLevel(logging.WARNING)
        
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
        
        # 检查捕获的日志
        log_content = log_capture.getvalue()
        future_v_warnings = [line for line in log_content.split('\n') if 'future_v_for_age' in line]
        
        print(f"\n=== 测试结果 ===")
        print(f"计算了 {len(std_errors)} 个参数的标准误")
        print(f"捕获到的future_v_for_age警告数量: {len(future_v_warnings)}")
        
        if future_v_warnings:
            print("❌ 警告: 仍然存在future_v_for_age问题！")
            for warning in future_v_warnings[:5]:  # 显示前5个
                print(f"  {warning}")
        else:
            print("✅ 成功: 没有future_v_for_age警告！")
        
        # 显示前几个参数的结果
        param_names = list(std_errors.keys())[:3]
        print(f"\n前几个参数的结果:")
        for name in param_names:
            print(f"  {name}: 估计值={em_results['structural_params'][name]:.4f}, "
                  f"标准误={std_errors[name]:.4f}, t值={t_stats[name]:.2f}")
        
        return len(future_v_warnings) == 0
        
    except Exception as e:
        print(f"❌ 流式Louis方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_large_sample_streaming()
    if success:
        print("\n✅ 大样本流式Louis方法测试通过！")
        print("future_v_for_age问题已解决！")
        sys.exit(0)
    else:
        print("\n❌ 大样本流式Louis方法测试失败！")
        print("future_v_for_age问题仍然存在！")
        sys.exit(1)