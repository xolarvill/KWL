#!/usr/bin/env python3
"""
进度跟踪功能演示脚本
演示如何在长时间运行的估计任务中使用断点续跑功能
"""
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.estimation_progress import estimation_progress, resume_estimation_phase, get_estimation_progress
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_heavy_computation(name, duration=5, memory_mb=10):
    """模拟耗时的计算任务"""
    logger.info(f"🚀 开始 {name}")
    logger.info(f"   预计耗时: {duration}秒, 内存使用: {memory_mb}MB")
    
    # 模拟内存使用
    data = []
    for i in range(memory_mb * 1000):  # 粗略模拟
        data.append([i] * 100)
    
    # 模拟耗时计算
    for i in range(duration):
        time.sleep(1)
        if i > 0 and i % 2 == 0:
            logger.info(f"   ⏰ {name} 进度: {i}/{duration}秒")
    
    result = {
        'task': name,
        'duration': duration,
        'memory_used': memory_mb,
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'accuracy': 0.95,
            'convergence': True,
            'iterations': 150
        }
    }
    
    logger.info(f"✅ {name} 完成")
    return result


def simulate_data_preparation():
    """模拟数据准备阶段"""
    return simulate_heavy_computation("数据准备", duration=3, memory_mb=50)


def simulate_model_estimation():
    """模拟模型估计阶段"""
    return simulate_heavy_computation("模型估计", duration=8, memory_mb=200)


def simulate_standard_errors():
    """模拟标准误计算阶段"""
    return simulate_heavy_computation("标准误计算", duration=6, memory_mb=100)


def simulate_result_output():
    """模拟结果输出阶段"""
    return simulate_heavy_computation("结果输出", duration=2, memory_mb=20)


def demo_normal_execution():
    """演示正常执行流程"""
    logger.info("\n" + "="*60)
    logger.info("演示1: 正常执行流程")
    logger.info("="*60)
    
    with estimation_progress(
        task_name="demo_estimation",
        progress_dir="progress",
        save_interval=2,  # 每2秒保存一次
        auto_cleanup=True
    ) as tracker:
        
        logger.info(f"恢复模式: {tracker.state['is_resumed']}")
        
        # 执行各个阶段
        data_result = resume_estimation_phase(
            tracker, "data_preparation", simulate_data_preparation
        )
        
        model_result = resume_estimation_phase(
            tracker, "model_estimation", simulate_model_estimation
        )
        
        stderr_result = resume_estimation_phase(
            tracker, "standard_errors", simulate_standard_errors
        )
        
        output_result = resume_estimation_phase(
            tracker, "result_output", simulate_result_output
        )
        
        logger.info("\n🎉 所有阶段完成!")
        logger.info(f"数据准备结果: {data_result['metrics']}")
        logger.info(f"模型估计结果: {model_result['metrics']}")
        logger.info(f"标准误计算结果: {stderr_result['metrics']}")


def demo_interrupted_execution():
    """演示中断后恢复执行"""
    logger.info("\n" + "="*60)
    logger.info("演示2: 中断后恢复执行")
    logger.info("="*60)
    
    # 第一次运行（模拟中断）
    logger.info("第一次运行: 将在模型估计阶段中断...")
    
    try:
        with estimation_progress(
            task_name="demo_interrupted",
            progress_dir="progress",
            save_interval=1,
            auto_cleanup=False
        ) as tracker:
            
            # 阶段1: 数据准备
            resume_estimation_phase(
                tracker, "data_preparation", simulate_data_preparation
            )
            
            # 阶段2: 模型估计（模拟长时间运行）
            logger.info("模拟模型估计阶段（按Ctrl+C中断）...")
            logger.info("提示: 可以按Ctrl+C模拟异常中断")
            
            for i in range(10):  # 长时间循环，容易被中断
                time.sleep(1)
                logger.info(f"模型估计进行中... {i+1}/10")
            
            # 如果正常完成到这里
            resume_estimation_phase(
                tracker, "standard_errors", simulate_standard_errors
            )
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  用户中断执行")
        logger.info("进度已自动保存")
    except Exception as e:
        logger.info(f"\n❌ 异常中断: {e}")
    
    # 检查当前进度
    logger.info("\n检查当前进度状态:")
    progress = get_estimation_progress(task_name="demo_interrupted")
    if progress:
        logger.info(f"已完成阶段: {progress['completed_phases']}")
        logger.info(f"当前阶段: {progress['current_phase']}")
    
    # 第二次运行（恢复执行）
    logger.info("\n第二次运行: 将从断点恢复执行...")
    
    with estimation_progress(
        task_name="demo_interrupted",
        progress_dir="progress",
        save_interval=2,
        auto_cleanup=True  # 演示完成后清理
    ) as tracker:
        
        logger.info(f"恢复模式: {tracker.state['is_resumed']}")
        logger.info(f"已完成阶段: {tracker.state['completed_phases']}")
        
        # 数据准备阶段应该被跳过
        data_result = resume_estimation_phase(
            tracker, "data_preparation", simulate_data_preparation
        )
        
        # 模型估计阶段重新执行
        model_result = resume_estimation_phase(
            tracker, "model_estimation", simulate_model_estimation
        )
        
        # 继续执行剩余阶段
        stderr_result = resume_estimation_phase(
            tracker, "standard_errors", simulate_standard_errors
        )
        
        output_result = resume_estimation_phase(
            tracker, "result_output", simulate_result_output
        )
        
        logger.info("\n🎉 恢复执行完成!")


def demo_command_line_tools():
    """演示命令行工具的使用"""
    logger.info("\n" + "="*60)
    logger.info("演示3: 命令行工具使用")
    logger.info("="*60)
    
    logger.info("可以使用以下命令管理进度:")
    logger.info("  python scripts/manage_progress.py check   - 检查进度")
    logger.info("  python scripts/manage_progress.py list    - 列出所有进度")
    logger.info("  python scripts/manage_progress.py clean   - 清理进度")
    logger.info("")
    logger.info("主脚本也提供进度管理选项:")
    logger.info("  python scripts/02_run_estimation.py --check-progress   - 检查并退出")
    logger.info("  python scripts/02_run_estimation.py --clean-progress   - 清理进度")
    logger.info("  python scripts/02_run_estimation.py --no-progress-tracking - 禁用进度跟踪")


def main():
    """主演示函数"""
    logger.info("🚀 开始进度跟踪功能演示")
    logger.info("这个演示将展示如何使用断点续跑功能")
    
    # 演示1: 正常执行
    demo_normal_execution()
    
    # 演示2: 中断后恢复（可选）
    logger.info("\n" + "?"*60)
    response = input("是否演示中断恢复功能? (y/N): ").strip().lower()
    if response == 'y':
        demo_interrupted_execution()
    
    # 演示3: 命令行工具
    demo_command_line_tools()
    
    logger.info("\n✨ 演示完成!")
    logger.info("进度文件已自动清理")


if __name__ == "__main__":
    main()