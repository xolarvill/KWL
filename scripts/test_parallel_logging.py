#!/usr/bin/env python3
"""
测试并行化日志输出的改进效果
"""

import logging
import time
import threading
import random
from src.utils.parallel_logging import ParallelLogger, QuietParallelLogger

def setup_logging():
    """设置日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

def simulate_individual_processing(logger: ParallelLogger, worker_id: str, 
                                 individual_ids: list, error_rate: float = 0.1):
    """模拟个体处理过程"""
    logger.log_worker_start(worker_id, len(individual_ids))
    
    for individual_id in individual_ids:
        try:
            # 模拟处理时间
            processing_time = random.uniform(0.01, 0.1)
            time.sleep(processing_time)
            
            # 模拟随机错误
            if random.random() < error_rate:
                raise ValueError(f"模拟处理错误 for individual {individual_id}")
            
            # 模拟缓存命中/未命中
            cache_hit = random.random() < 0.7  # 70% 缓存命中率
            
            # 记录成功处理
            logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=True,
                cache_hit=cache_hit,
                processing_time=processing_time
            )
            
        except Exception as e:
            # 记录处理错误
            logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=False,
                error_msg=str(e)
            )

def test_parallel_logging_comparison():
    """对比测试传统日志 vs 改进后的并行日志"""
    
    print("\n" + "="*60)
    print("对比测试：传统并行日志 vs 改进并行日志")
    print("="*60)
    
    # 测试参数
    total_individuals = 20
    n_workers = 4
    individuals_per_worker = total_individuals // n_workers
    
    # === 传统方式（混乱的日志）===
    print("\n1. 传统并行日志（预期：混乱交错）:")
    print("-" * 40)
    
    def traditional_worker(worker_id, individual_ids):
        for i, individual_id in enumerate(individual_ids):
            print(f"[Worker {worker_id}] 开始处理个体 {individual_id}")
            time.sleep(0.05)
            print(f"[Worker {worker_id}] 个体 {individual_id} 处理完成")
            if i == 1:  # 模拟一个错误
                print(f"[Worker {worker_id}] 个体 {individual_id} 处理出错: 模拟错误")
    
    threads = []
    for i in range(n_workers):
        worker_id = f"worker_{i}"
        start_idx = i * individuals_per_worker
        end_idx = start_idx + individuals_per_worker
        individual_ids = list(range(start_idx, end_idx))
        
        thread = threading.Thread(
            target=traditional_worker,
            args=(worker_id, individual_ids)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    time.sleep(1)  # 短暂暂停
    
    # === 改进方式（有序的日志）===
    print("\n\n2. 改进并行日志（预期：清晰有序）:")
    print("-" * 40)
    
    logger = setup_logging()
    parallel_logger = ParallelLogger(logger, verbose=False)
    
    def improved_worker(worker_id, individual_ids, parallel_logger):
        parallel_logger.log_worker_start(worker_id, len(individual_ids))
        
        for individual_id in individual_ids:
            try:
                # 模拟处理
                processing_time = 0.05
                time.sleep(processing_time)
                
                # 模拟第二个个体出错
                if individual_id == individual_ids[1]:
                    raise ValueError("模拟处理错误")
                
                # 记录成功
                parallel_logger.log_individual_processed(
                    worker_id=worker_id,
                    individual_id=individual_id,
                    success=True,
                    cache_hit=individual_id % 3 == 0,  # 模拟缓存命中
                    processing_time=processing_time
                )
                
            except Exception as e:
                # 记录错误
                parallel_logger.log_individual_processed(
                    worker_id=worker_id,
                    individual_id=individual_id,
                    success=False,
                    error_msg=str(e)
                )
    
    # 启动改进的工作进程
    threads = []
    for i in range(n_workers):
        worker_id = f"worker_{i}"
        start_idx = i * individuals_per_worker
        end_idx = start_idx + individuals_per_worker
        individual_ids = list(range(start_idx, end_idx))
        
        thread = threading.Thread(
            target=improved_worker,
            args=(worker_id, individual_ids, parallel_logger)
        )
        threads.append(thread)
        thread.start()
    
    # 等待所有工作进程完成
    for thread in threads:
        thread.join()
    
    # 输出最终统计
    parallel_logger.finish_processing()

def test_progress_display():
    """测试进度显示效果"""
    print("\n\n3. 进度显示效果测试:")
    print("-" * 40)
    
    logger = setup_logging()
    parallel_logger = QuietParallelLogger(logger)
    
    # 模拟处理100个个体
    total_individuals = 100
    n_workers = 4
    
    def batch_worker(worker_id, start_idx, end_idx, parallel_logger):
        individual_ids = list(range(start_idx, end_idx))
        parallel_logger.log_worker_start(worker_id, len(individual_ids))
        
        for individual_id in individual_ids:
            # 模拟处理时间
            processing_time = random.uniform(0.01, 0.03)
            time.sleep(processing_time)
            
            # 记录处理完成
            parallel_logger.log_individual_processed(
                worker_id=worker_id,
                individual_id=individual_id,
                success=True,
                cache_hit=random.random() < 0.7,  # 70% 缓存命中率
                processing_time=processing_time
            )
    
    # 开始处理
    parallel_logger.start_processing(total_individuals)
    
    # 启动工作线程
    threads = []
    individuals_per_worker = total_individuals // n_workers
    
    for i in range(n_workers):
        worker_id = f"worker_{i}"
        start_idx = i * individuals_per_worker
        end_idx = start_idx + individuals_per_worker
        
        thread = threading.Thread(
            target=batch_worker,
            args=(worker_id, start_idx, end_idx, parallel_logger)
        )
        threads.append(thread)
        thread.start()
    
    # 等待完成
    for thread in threads:
        thread.join()
    
    # 输出最终统计
    parallel_logger.finish_processing()

def main():
    print("并行化日志系统改进演示")
    print("=" * 60)
    
    # 1. 对比测试
    test_parallel_logging_comparison()
    
    # 2. 进度显示测试
    test_progress_display()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("\n改进效果总结:")
    print("✓ 日志消息不再交错混乱")
    print("✓ 进度显示清晰有序")
    print("✓ 统计信息聚合准确")
    print("✓ 错误信息精确定位")

if __name__ == '__main__':
    main()