#!/usr/bin/env python3
"""
测试新的轻量级并行系统
专门验证Windows pickle问题是否解决
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.lightweight_parallel_wrapper import LightweightParallelConfig
from src.estimation.em_with_omega import e_step_with_omega
from src.utils.lightweight_parallel_logging import SimpleParallelLogger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(n_individuals: int = 50):
    """创建最小化的测试数据"""
    np.random.seed(42)
    
    # 个体数据
    individuals = []
    for i in range(n_individuals):
        n_periods = np.random.randint(3, 8)
        for t in range(n_periods):
            individuals.append({
                'individual_id': f'ind_{i:03d}',
                'period': t,
                'province_id': np.random.randint(1, 10),
                'provcd_t': np.random.randint(1, 10),  # 添加缺失的字段
                'wage': np.random.lognormal(8, 0.5),
                'wage_reg': np.random.lognormal(8, 0.5),
                'wage_res': np.random.lognormal(8, 0.5),
                'distance_to_home': np.random.exponential(100),
                'age': 25 + t,
                'married': np.random.choice([0, 1]),
                'health': np.random.choice([1, 2, 3])
            })
    
    observed_data = pd.DataFrame(individuals)
    
    # 状态空间
    state_space = pd.DataFrame({
        'province_id': range(1, 10),
        'wage_support': np.random.lognormal(8, 0.3, 9),
        'wage_support_reg': np.random.lognormal(8, 0.3, 9),
        'wage_support_res': np.random.lognormal(8, 0.3, 9)
    })
    
    return observed_data, state_space

def test_parallel_system():
    """测试新的并行系统"""
    logger.info("="*60)
    logger.info("开始测试新的轻量级并行系统")
    logger.info("="*60)
    
    try:
        # 创建测试数据
        logger.info("创建测试数据...")
        observed_data, state_space = create_test_data(n_individuals=20)
        logger.info(f"测试数据: {len(observed_data)} 条记录, {observed_data['individual_id'].nunique()} 个个体")
        
        # 最小化参数配置
        logger.info("配置测试参数...")
        
        # 基础参数
        params = {
            'theta_1': np.array([0.1, 0.2, 0.3]),
            'theta_2': np.array([0.05, 0.1]),
            'sigma_eps': 0.3,
            'sigma_xi': 0.2,
            'mu': np.array([8.0, 0.1, 0.05]),
            'Sigma': np.eye(3) * 0.1,
            'beta': 0.95,
            'gamma': np.array([0.1, 0.2]),
            'lambda': 0.5,
            'alpha': np.array([0.3, 0.4]),
            'sigma_eta': 0.15
        }
        
        # 其他必需参数
        n_types = 2
        pi_k = np.array([0.6, 0.4])
        beta = 0.95
        
        # 简化的转移矩阵
        transition_matrices = {
            'P': np.ones((9, 9)) / 9,
            'P_reg': np.ones((9, 9)) / 9,
            'P_res': np.ones((9, 9)) / 9
        }
        
        # 地区数据
        regions_df = {
            'region_codes': np.array(range(1, 10)),
            'region_names': np.array([f'region_{i}' for i in range(1, 10)])
        }
        
        # 距离和邻接矩阵
        distance_matrix = np.random.exponential(100, (9, 9))
        adjacency_matrix = np.random.choice([0, 1], (9, 9))
        
        # 省份映射
        prov_to_idx = {i: i-1 for i in range(1, 10)}
        
        # ω支持点生成器（简化版）
        class SimpleSupportGenerator:
            def generate_support_points(self, *args, **kwargs):
                return [{'omega_1': 0.1, 'omega_2': 0.2}], np.array([1.0])
        
        support_generator = SimpleSupportGenerator()
        
        logger.info("测试不同并行配置...")
        
        # 测试1：串行模式
        logger.info("\n--- 测试1：串行模式 ---")
        config_serial = LightweightParallelConfig(n_jobs=1)
        
        start_time = time.time()
        try:
            individual_posteriors_serial, log_likelihood_serial = e_step_with_omega(
                params=params,
                pi_k=pi_k,
                observed_data=observed_data,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=beta,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                support_generator=support_generator,
                n_types=n_types,
                prov_to_idx=prov_to_idx,
                max_omega_per_individual=5,  # 减少计算量
                use_simplified_omega=True,
                bellman_cache=None,
                parallel_config=config_serial
            )
            serial_time = time.time() - start_time
            logger.info(f"✅ 串行模式成功！耗时: {serial_time:.2f}秒")
            logger.info(f"结果个体数: {len(individual_posteriors_serial)}")
            
        except Exception as e:
            logger.error(f"❌ 串行模式失败: {e}")
            raise
        
        # 测试2：并行模式（2进程）
        logger.info("\n--- 测试2：并行模式（2进程） ---")
        config_parallel = LightweightParallelConfig(n_jobs=2, backend='loky')
        
        start_time = time.time()
        try:
            individual_posteriors_parallel, log_likelihood_parallel = e_step_with_omega(
                params=params,
                pi_k=pi_k,
                observed_data=observed_data,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=beta,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                support_generator=support_generator,
                n_types=n_types,
                prov_to_idx=prov_to_idx,
                max_omega_per_individual=5,
                use_simplified_omega=True,
                bellman_cache=None,
                parallel_config=config_parallel
            )
            parallel_time = time.time() - start_time
            logger.info(f"✅ 并行模式成功！耗时: {parallel_time:.2f}秒")
            logger.info(f"结果个体数: {len(individual_posteriors_parallel)}")
            
            # 验证结果一致性
            if len(individual_posteriors_serial) == len(individual_posteriors_parallel):
                logger.info("✅ 串行和并行结果数量一致")
            else:
                logger.warning(f"⚠️ 结果数量不一致: 串行{len(individual_posteriors_serial)} vs 并行{len(individual_posteriors_parallel)}")
            
        except Exception as e:
            logger.error(f"❌ 并行模式失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            if "pickle" in str(e).lower() or "serialize" in str(e).lower():
                logger.error("🚨 检测到pickle序列化错误！新系统未生效")
            raise
        
        # 测试3：检查pickle安全性
        logger.info("\n--- 测试3：pickle安全性验证 ---")
        try:
            import pickle
            
            # 测试worker数据是否可以pickle
            from src.utils.lightweight_parallel_logging import create_safe_worker_logger, log_worker_progress
            
            test_worker_data = create_safe_worker_logger()
            log_worker_progress(test_worker_data, "test_id", success=True, cache_hit=True)
            
            # 尝试pickle
            pickled_data = pickle.dumps(test_worker_data)
            unpickled_data = pickle.loads(pickled_data)
            
            logger.info(f"✅ WorkerLogData可以安全pickle！大小: {len(pickled_data)} bytes")
            logger.info(f"✅  unpickle后数据完整: processed={unpickled_data.processed_count}")
            
        except Exception as e:
            logger.error(f"❌ Pickle测试失败: {e}")
            raise
        
        logger.info("\n" + "="*60)
        logger.info("🎉 所有测试通过！新的轻量级并行系统工作正常")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        logger.error(f"错误追踪:", exc_info=True)
        return False

def main():
    """主函数"""
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"项目根目录: {project_root}")
    
    # 运行测试
    success = test_parallel_system()
    
    if success:
        logger.info("\n✅ 新并行系统测试成功！Windows pickle问题应该已解决")
        sys.exit(0)
    else:
        logger.error("\n❌ 新并行系统测试失败！需要进一步调试")
        sys.exit(1)

if __name__ == "__main__":
    main()