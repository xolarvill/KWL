#!/usr/bin/env python3
"""
测试进度跟踪功能的简单脚本
"""
import time
import numpy as np
from src.utils.estimation_progress import estimation_progress, resume_estimation_phase
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_work(name, duration=2, fail=False):
    """模拟工作任务"""
    logger.info(f"开始工作: {name}")
    time.sleep(duration)
    
    if fail:
        raise ValueError(f"模拟 {name} 失败")
    
    result = f"{name} 完成于 {time.strftime('%H:%M:%S')}"
    logger.info(result)
    return result


def test_progress_tracking():
    """测试进度跟踪功能"""
    logger.info("=== 开始测试进度跟踪功能 ===")
    
    # 使用进度跟踪
    with estimation_progress(
        task_name="test_task",
        progress_dir="progress",
        save_interval=1,  # 每1秒保存一次，用于测试
        auto_cleanup=False
    ) as tracker:
        
        logger.info(f"是否恢复模式: {tracker.state['is_resumed']}")
        
        # 阶段1: 数据准备
        result1 = resume_estimation_phase(
            tracker, "data_preparation", 
            simulate_work, "数据准备", duration=2
        )
        
        # 阶段2: 模型估计
        result2 = resume_estimation_phase(
            tracker, "model_estimation",
            simulate_work, "模型估计", duration=3
        )
        
        # 阶段3: 标准误计算
        result3 = resume_estimation_phase(
            tracker, "standard_errors",
            simulate_work, "标准误计算", duration=2
        )
        
        # 阶段4: 结果输出
        result4 = resume_estimation_phase(
            tracker, "output_results",
            simulate_work, "结果输出", duration=1
        )
        
        logger.info("=== 所有阶段完成 ===")
        logger.info(f"结果摘要: {result1}, {result2}, {result3}, {result4}")


def test_resume_capability():
    """测试断点续跑功能"""
    logger.info("\n=== 测试断点续跑功能 ===")
    
    # 第一次运行（部分完成）
    logger.info("第一次运行: 模拟中断在模型估计阶段")
    
    try:
        with estimation_progress(
            task_name="resume_test",
            progress_dir="progress",
            save_interval=1,
            auto_cleanup=False
        ) as tracker:
            
            # 阶段1: 数据准备
            resume_estimation_phase(
                tracker, "data_preparation",
                simulate_work, "数据准备", duration=1
            )
            
            # 阶段2: 模型估计（模拟失败）
            resume_estimation_phase(
                tracker, "model_estimation",
                simulate_work, "模型估计", duration=1, fail=True
            )
            
    except ValueError as e:
        logger.info(f"预期中的失败: {e}")
    
    # 第二次运行（应该跳过已完成的阶段）
    logger.info("\n第二次运行: 应该跳过已完成的阶段")
    
    with estimation_progress(
        task_name="resume_test",
        progress_dir="progress",
        save_interval=1,
        auto_cleanup=True  # 测试完成后清理
    ) as tracker:
        
        logger.info(f"恢复模式: {tracker.state['is_resumed']}")
        logger.info(f"已完成阶段: {tracker.state['completed_phases']}")
        
        # 阶段1: 数据准备（应该跳过）
        resume_estimation_phase(
            tracker, "data_preparation",
            simulate_work, "数据准备", duration=1
        )
        
        # 阶段2: 模型估计（重新执行）
        resume_estimation_phase(
            tracker, "model_estimation",
            simulate_work, "模型估计", duration=1, fail=False
        )
        
        # 阶段3: 标准误计算
        resume_estimation_phase(
            tracker, "standard_errors",
            simulate_work, "标准误计算", duration=1
        )
        
        logger.info("=== 断点续跑测试完成 ===")


if __name__ == "__main__":
    # 测试进度跟踪
    test_progress_tracking()
    
    # 测试断点续跑
    test_resume_capability()
    
    logger.info("\n=== 所有测试完成 ===")