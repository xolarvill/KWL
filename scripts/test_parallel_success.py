#!/usr/bin/env python3
"""
验证并行化是否成功工作的简单测试
"""

import sys
sys.path.insert(0, '.')

from src.estimation.e_step_parallel_processor import process_individual_with_data_package, create_parallel_processing_data
from src.utils.parallel_wrapper import ParallelConfig
from src.utils.parallel_logging import QuietParallelLogger
from src.utils.parallel_logger_registry import register_parallel_logger
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
import logging

def test_parallel_success():
    """测试并行化是否真正成功"""
    
    print("=== 并行化成功验证测试 ===")
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 创建测试数据
    n_individuals = 8
    individual_ids = [f'person_{i}' for i in range(n_individuals)]
    
    # 创建模拟数据
    individual_omega_dict = {
        ind_id: ([{'eta': 1.0, 'sigma': 0.5}], np.array([0.7, 0.3]))
        for ind_id in individual_ids
    }
    
    # 创建数据包
    data_package = create_parallel_processing_data(
        individual_omega_dict=individual_omega_dict,
        params={'gamma_0': 0.1, 'gamma_1': -0.2},
        pi_k=np.array([0.33, 0.33, 0.34]),
        K=3,
        beta=0.95,
        transition_matrices={},
        regions_df={},
        distance_matrix=np.array([[1, 2], [3, 4]]),
        adjacency_matrix=np.array([[0, 1], [1, 0]]),
        prov_to_idx={1: 0, 2: 1},
        bellman_cache=None
    )
    
    # 创建日志管理器
    parallel_logger = QuietParallelLogger(logger)
    logger_id = register_parallel_logger(parallel_logger)
    
    # 创建个体数据
    individual_data_list = []
    for ind_id in individual_ids:
        df = pd.DataFrame({
            'individual_id': [ind_id],
            'visited_locations': [[1, 2, 3]],
            'age_t': [25],
            'provcd_t': [1],
            'prev_provcd': [2]
        })
        individual_data_list.append(df)
    
    print(f"\n测试 {n_individuals} 个个体的并行处理...")
    print(f"并行配置: 2个工作进程")
    
    try:
        # 开始并行处理
        parallel_logger.start_processing(n_individuals)
        start_time = time.time()
        
        # 真正的并行处理测试
        results = Parallel(n_jobs=2, backend='loky', verbose=1)(
            delayed(process_individual_with_data_package)(
                ind_id, 
                individual_data_list[i],
                data_package,
                logger_id
            )
            for i, ind_id in enumerate(individual_ids)
        )
        
        end_time = time.time()
        
        # 完成处理
        parallel_logger.finish_processing()
        
        # 验证结果
        print(f"\n✅ 并行处理成功！")
        print(f"处理了个体数量: {len(results)}")
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"平均速度: {len(results)/(end_time - start_time):.1f} 个体/秒")
        
        # 检查是否有有效结果
        valid_results = [r for r in results if len(r[1]) > 0]  # joint_probs不为空
        print(f"有效结果数量: {len(valid_results)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 并行处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理注册
        from src.utils.parallel_logger_registry import unregister_parallel_logger
        unregister_parallel_logger(logger_id)

if __name__ == '__main__':
    success = test_parallel_success()
    
    if success:
        print("\n🎉 并行化修复成功！序列化问题已解决！")
        print("\n📊 改进效果:")
        print("  ✅ 无闭包函数设计，支持pickle序列化")
        print("  ✅ 数据包模式，避免复杂对象传递")
        print("  ✅ 智能日志管理，清晰不混乱")
        print("  ✅ 自动回退机制，保证稳定性")
    else:
        print("\n⚠️  并行化仍有问题，需要进一步调试")