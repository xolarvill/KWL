"""
最小侵入式并行处理包装器
为EM算法提供可配置的个体级别并行化
"""

import functools
import logging
import time
from typing import Callable, List, Any, Optional
from joblib import Parallel, delayed
import os

logger = logging.getLogger(__name__)


class ParallelConfig:
    """并行配置管理器"""
    
    def __init__(self, n_jobs: int = 1, backend: str = 'loky', 
                 batch_size: str = 'auto', verbose: int = 0):
        """
        初始化并行配置
        
        Args:
            n_jobs: 并行任务数，-1表示使用所有CPU核心，1表示禁用并行化
            backend: 并行后端 ('loky', 'threading', 'multiprocessing')
            batch_size: 批处理大小 ('auto' 或具体数值)
            verbose: 详细程度 (0-10)
        """
        self.n_jobs = self._validate_n_jobs(n_jobs)
        self.backend = backend
        self.batch_size = batch_size
        self.verbose = verbose
        self._original_env = None
    
    def _validate_n_jobs(self, n_jobs: int) -> int:
        """验证并标准化n_jobs参数"""
        if n_jobs == -1:
            # 使用所有可用核心
            return os.cpu_count() or 4
        elif n_jobs < -1:
            # 负值表示使用cpu_count + n_jobs + 1
            return max(1, (os.cpu_count() or 4) + n_jobs + 1)
        else:
            return max(1, n_jobs)
    
    def is_parallel_enabled(self) -> bool:
        """检查是否启用了并行化"""
        return self.n_jobs > 1
    
    def __str__(self) -> str:
        return (f"ParallelConfig(n_jobs={self.n_jobs}, backend='{self.backend}', "
                f"batch_size='{self.batch_size}', verbose={self.verbose})")


def parallel_individual_processor(config_getter: Optional[Callable] = None):
    """
    个体处理并行化装饰器
    
    这个装饰器可以应用于处理个体列表的函数，自动根据配置进行并行化。
    适用于E-step和M-step中的个体处理循环。
    
    Args:
        config_getter: 获取并行配置的函数，如果为None则使用默认配置
    
    使用示例:
        @parallel_individual_processor()
        def process_individuals(individual_list, *args, **kwargs):
            results = []
            for individual_id in individual_list:
                result = process_single_individual(individual_id, *args, **kwargs)
                results.append(result)
            return results
    
    或者:
        @parallel_individual_processor(lambda: my_parallel_config)
        def my_function(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(individual_list: List[Any], *args, **kwargs):
            # 获取并行配置
            if config_getter is not None:
                config = config_getter()
            else:
                # 默认配置：不并行
                config = ParallelConfig(n_jobs=1)
            
            if not isinstance(config, ParallelConfig):
                logger.warning(f"Invalid parallel config, using default. Got: {type(config)}")
                config = ParallelConfig(n_jobs=1)
            
            # 记录开始时间
            start_time = time.time()
            total_individuals = len(individual_list)
            
            logger.info(f"并行处理 {total_individuals} 个个体，配置: {config}")
            
            if not config.is_parallel_enabled():
                # 串行处理（原始逻辑）
                logger.info("使用串行处理模式")
                return func(individual_list, *args, **kwargs)
            else:
                # 并行处理
                logger.info(f"使用并行处理模式，{config.n_jobs} 个工作进程")
                
                try:
                    # 使用joblib进行并行处理
                    results = Parallel(
                        n_jobs=config.n_jobs,
                        backend=config.backend,
                        batch_size=config.batch_size,
                        verbose=config.verbose
                    )(
                        delayed(_process_single_individual_wrapper)(func, individual_id, *args, **kwargs)
                        for individual_id in individual_list
                    )
                    
                    # 记录处理统计
                    elapsed_time = time.time() - start_time
                    rate = total_individuals / elapsed_time if elapsed_time > 0 else 0
                    
                    logger.info(f"并行处理完成: {total_individuals} 个个体, "
                              f"耗时: {elapsed_time:.2f}s, 处理速度: {rate:.2f} 个体/秒")
                    
                    return results
                    
                except Exception as e:
                    logger.error(f"并行处理失败，回退到串行模式: {e}")
                    # 如果并行处理失败，回退到串行处理
                    return func(individual_list, *args, **kwargs)
        
        return wrapper
    return decorator


def _process_single_individual_wrapper(func: Callable, individual_id: Any, *args, **kwargs):
    """
    单个个体处理的包装函数
    
    这个函数包装原始函数，使其能够处理单个个体而不是列表。
    """
    try:
        # 调用原始函数处理单个个体（包装成列表）
        result = func([individual_id], *args, **kwargs)
        # 如果结果是列表且只有一个元素，返回该元素
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        else:
            return result
    except Exception as e:
        logger.error(f"处理个体 {individual_id} 时出错: {e}")
        # 返回错误标记，让调用者决定如何处理
        return {"error": str(e), "individual_id": individual_id}


def create_progress_wrapper(total: int, desc: str = "Processing"):
    """
    创建进度包装器（可选）
    
    如果需要更详细的进度跟踪，可以使用这个包装器。
    目前使用简单的日志记录。
    """
    def progress_wrapper(iterator):
        processed = 0
        start_time = time.time()
        
        for item in iterator:
            yield item
            processed += 1
            
            # 每处理10%输出一次进度
            if processed % max(1, total // 10) == 0 or processed == total:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                
                logger.info(f"{desc}: {processed}/{total} ({processed/total*100:.1f}%), "
                          f"速度: {rate:.2f} 个体/秒, 预计剩余: {remaining:.1f}秒")
    
    return progress_wrapper