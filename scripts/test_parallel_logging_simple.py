#!/usr/bin/env python3
"""
简化版并行化日志效果演示
"""

import logging
import time
import threading
from src.utils.parallel_logging import ParallelLogger, QuietParallelLogger

def setup_logging():
    """设置日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

def simulate_worker_task(logger: ParallelLogger, worker_id: str, start_idx: int, end_idx: int):
    """模拟工作进程任务"""
    individual_ids = list(range(start_idx, end_idx))
    logger.log_worker_start(worker_id, len(individual_ids))
    
    for individual_id in individual_ids:
        # 模拟处理时间
        processing_time = 0.1
        time.sleep(processing_time)
        
        # 模拟随机错误（10%概率）
        if individual_id % 10 == 2:  # 让某些ID出错
            logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=False,
                error_msg=f"处理个体 {individual_id} 时出错"
            )
        else:
            # 记录成功处理
            logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=True,
                cache_hit=individual_id % 3 == 0,  # 模拟缓存命中
                processing_time=processing_time
            )

def demonstrate_improvement():
    """演示改进效果"""
    logger = setup_logging()
    
    print("\n" + "="*70)
    print("并行化日志系统改进效果演示")
    print("="*70)
    
    # 测试参数
    total_individuals = 20
    n_workers = 4
    individuals_per_worker = total_individuals // n_workers
    
    # === 1. 传统方式（混乱的日志）===
    print("\n1. 传统并行日志（消息交错混乱）:")
    print("-" * 50)
    
    def traditional_worker(worker_id, start_idx, end_idx):
        for individual_id in range(start_idx, end_idx):
            print(f"[{worker_id}] 开始处理个体 {individual_id}")
            time.sleep(0.1)
            print(f"[{worker_id}] 个体 {individual_id} 处理完成")
            if individual_id % 10 == 2:  # 模拟错误
                print(f"[{worker_id}] ⚠️  个体 {individual_id} 处理出错！")
    
    # 启动传统工作线程
    threads = []
    for i in range(n_workers):
        worker_id = f"Worker-{i}"
        start_idx = i * individuals_per_worker
        end_idx = start_idx + individuals_per_worker
        
        thread = threading.Thread(target=traditional_worker, args=(worker_id, start_idx, end_idx))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    time.sleep(0.5)  # 短暂暂停
    
    # === 2. 改进方式（有序的日志）===
    print("\n\n2. 改进并行日志（清晰有序）:")
    print("-" * 50)
    
    # 创建并行日志管理器
    parallel_logger = QuietParallelLogger(logger)
    
    # 开始处理
    parallel_logger.start_processing(total_individuals)
    
    # 启动改进的工作线程
    threads = []
    for i in range(n_workers):
        worker_id = f"Worker-{i}"
        start_idx = i * individuals_per_worker
        end_idx = start_idx + individuals_per_worker
        
        thread = threading.Thread(
            target=simulate_worker_task,
            args=(parallel_logger, worker_id, start_idx, end_idx)
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 输出最终统计
    parallel_logger.finish_processing()

def show_key_improvements():
    """展示关键改进点"""
    print("\n" + "="*70)
    print("关键改进效果总结")
    print("="*70)
    
    print("\n✅ 改进前的问题：")
    print("  • 日志消息交错混乱，难以阅读")
    print("  • 无法追踪整体进度")
    print("  • 统计信息重复或丢失")
    print("  • 错误信息难以定位")
    
    print("\n✅ 改进后的优势：")
    print("  • 🎯 统一进度显示：清晰显示处理进度和速度")
    print("  • 📊 聚合统计信息：准确的缓存命中率和错误统计")
    print("  • 🔍 精确定位错误：具体到工作进程和个体ID")
    print("  • ⚡ 智能更新频率：避免过度频繁的日志输出")
    print("  • 🛡️  自动错误处理：并行失败时优雅回退")
    
    print("\n✅ 技术实现：")
    print("  • 线程安全的日志管理器")
    print("  • 进程隔离的统计收集")
    print("  • 智能的进度更新策略")
    print("  • 详细的调试信息缓存（可选）")

def main():
    # 演示改进效果
    demonstrate_improvement()
    
    # 展示关键改进
    show_key_improvements()
    
    print("\n" + "="*70)
    print("演示完成！并行化日志系统已显著改善！")
    print("="*70)

if __name__ == '__main__':
    main()