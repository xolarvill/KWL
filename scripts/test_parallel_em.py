#!/usr/bin/env python3
"""
测试EM算法并行化功能的脚本
"""

import argparse
import logging
import time
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.parallel_wrapper import ParallelConfig
from scripts import run_estimation

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_parallel_config():
    """测试并行配置"""
    logger.info("测试并行配置...")
    
    # 测试不同的配置
    configs = [
        ParallelConfig(n_jobs=1),  # 串行
        ParallelConfig(n_jobs=2),  # 2个核心
        ParallelConfig(n_jobs=-1), # 所有核心
        ParallelConfig(n_jobs=-2), # cpu_count - 1
    ]
    
    for config in configs:
        logger.info(f"配置: {config}")
        logger.info(f"并行化启用: {config.is_parallel_enabled()}")

def test_small_sample():
    """使用小样本测试并行化"""
    logger.info("使用小样本测试并行化...")
    
    # 测试串行模式
    logger.info("=== 测试串行模式 ===")
    start_time = time.time()
    try:
        run_estimation.run_estimation_workflow(
            sample_size=50,  # 很小的样本
            stderr_method="louis",
            em_parallel_jobs=1,  # 串行
            em_parallel_backend='loky'
        )
        serial_time = time.time() - start_time
        logger.info(f"串行模式耗时: {serial_time:.2f}秒")
    except Exception as e:
        logger.error(f"串行模式测试失败: {e}")
        serial_time = None
    
    # 测试并行模式
    logger.info("=== 测试并行模式 ===")
    start_time = time.time()
    try:
        run_estimation.run_estimation_workflow(
            sample_size=50,  # 很小的样本
            stderr_method="louis",
            em_parallel_jobs=2,  # 2个核心并行
            em_parallel_backend='loky'
        )
        parallel_time = time.time() - start_time
        logger.info(f"并行模式耗时: {parallel_time:.2f}秒")
    except Exception as e:
        logger.error(f"并行模式测试失败: {e}")
        parallel_time = None
    
    # 比较结果
    if serial_time and parallel_time:
        speedup = serial_time / parallel_time
        logger.info(f"加速比: {speedup:.2f}x")
        if speedup > 1.0:
            logger.info("✓ 并行化提供了性能提升")
        else:
            logger.warning("⚠ 并行化未提供明显性能提升（小样本正常）")
    else:
        logger.warning("无法比较性能，部分测试失败")

def main():
    parser = argparse.ArgumentParser(description="测试EM算法并行化功能")
    parser.add_argument('--test-config', action='store_true', help='仅测试并行配置')
    parser.add_argument('--test-sample-size', type=int, default=50, help='测试样本大小')
    parser.add_argument('--parallel-jobs', type=int, default=2, help='并行任务数')
    parser.add_argument('--backend', type=str, default='loky', choices=['loky', 'threading', 'multiprocessing'],
                        help='并行后端')
    
    args = parser.parse_args()
    
    logger.info("开始EM算法并行化测试")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"CPU核心数: {os.cpu_count()}")
    
    if args.test_config:
        test_parallel_config()
    else:
        test_small_sample()
    
    logger.info("测试完成")

if __name__ == '__main__':
    main()