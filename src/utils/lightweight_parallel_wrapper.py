"""
轻量级并行处理包装器
专门解决Windows下的序列化问题
"""

import functools
import logging
import time
import threading
import psutil
import os
from typing import Callable, List, Any, Optional, Dict
from joblib import Parallel, delayed
from .lightweight_parallel_logging import (
    SimpleParallelLogger, WorkerLogData, 
    create_safe_worker_logger, log_worker_progress
)
from .memory_monitor import MemoryMonitor, log_memory_usage

logger = logging.getLogger(__name__)


def check_memory_usage():
    """检查当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = psutil.virtual_memory().percent
    return {
        'process_memory_mb': memory_info.rss / 1024 / 1024,
        'system_memory_percent': memory_percent
    }


class LightweightParallelConfig:
    """轻量级并行配置"""
    
    def __init__(self, n_jobs: int = 1, backend: str = 'loky', batch_size: str = 'auto'):
        """
        初始化配置
        
        Args:
            n_jobs: 并行任务数，-1表示使用所有CPU核心
            backend: 并行后端 ('loky', 'threading')
            batch_size: 批处理大小
        """
        self.n_jobs = self._validate_n_jobs(n_jobs)
        self.backend = backend if backend in ['loky', 'threading'] else 'loky'
        self.batch_size = batch_size
    
    def _validate_n_jobs(self, n_jobs: int) -> int:
        """验证并标准化n_jobs参数"""
        import os
        if n_jobs == -1:
            return os.cpu_count() or 4
        elif n_jobs < -1:
            return max(1, (os.cpu_count() or 4) + n_jobs + 1)
        else:
            return max(1, n_jobs)
    
    def is_parallel_enabled(self) -> bool:
        """检查是否启用了并行化"""
        return self.n_jobs > 1


def lightweight_parallel_processor(config_getter: Optional[Callable] = None,
                                 logger: Optional[logging.Logger] = None,
                                 quiet_mode: bool = True):
    """
    轻量级个体处理并行化装饰器
    
    核心特点：
    1. 子进程只返回简单数据，不直接操作日志对象
    2. 所有日志聚合在主进程中完成
    3. 完全避免pickle序列化问题
    
    Args:
        config_getter: 获取并行配置的函数
        logger: 主进程日志记录器
        quiet_mode: 是否使用安静模式
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(individual_list: List[Any], *args, **kwargs):
            # 获取并行配置
            if config_getter is not None:
                config = config_getter()
            else:
                config = LightweightParallelConfig(n_jobs=1)
            
            # 特别处理n_jobs=-1的情况
            if hasattr(config, 'n_jobs') and config.n_jobs == -1:
                import os
                config.n_jobs = os.cpu_count() or 4
            
            if not isinstance(config, LightweightParallelConfig):
                logger.warning(f"Invalid config type: {type(config)}, using serial mode")
                config = LightweightParallelConfig(n_jobs=1)
            
            # 设置日志记录器
            current_logger = logger or logging.getLogger(__name__)
            
            total_individuals = len(individual_list)
            current_logger.info(f"处理 {total_individuals} 个个体，配置: n_jobs={config.n_jobs}")
            
            # 内存监控
            memory_monitor = None
            if enable_memory_monitoring:
                memory_monitor = MemoryMonitor()
                memory_monitor.start_monitoring(interval=10.0)  # 每10秒检查一次
                log_memory_usage("并行处理开始前")
            
            try:
                # 并行处理 - 子进程只返回简单数据
                results_with_logs = Parallel(
                    n_jobs=config.n_jobs,
                    backend=config.backend,
                    batch_size=config.batch_size,
                    verbose=0,  # 禁用joblib日志
                    max_nbytes=50*1024*1024,  # 限制在worker之间传输的数据大小为50MB（减少内存使用）
                    mmap_mode='r',  # 使用内存映射模式读取数据
                    timeout=None,  # 不设置超时限制
                    temp_folder=None  # 使用系统默认临时文件夹
                )(
                    delayed(_safe_worker_function)(func, individual_id, *args, **kwargs)
                    for individual_id in individual_list
                )
                
                # 记录处理后的内存使用情况
                if enable_memory_monitoring:
                    log_memory_usage("并行处理完成后")
                    memory_info = memory_monitor.get_memory_info()
                    current_logger.info(f"内存使用峰值: 系统内存={memory_info['peak_memory_percent']:.1f}%, "
                                      f"进程内存={memory_info['peak_process_memory_mb']:.1f}MB")
                
            finally:
                # 停止内存监控
                if memory_monitor:
                    memory_monitor.stop_monitoring()
            
            if not config.is_parallel_enabled():
                # 串行处理
                current_logger.info("使用串行处理模式")
                return func(individual_list, *args, **kwargs)
            else:
                # 并行处理 - 使用新的轻量级方案
                current_logger.info(f"使用并行处理模式，{config.n_jobs} 个工作进程")
                
                # 创建日志管理器（主进程专用）
                parallel_logger = SimpleParallelLogger(current_logger, quiet_mode=quiet_mode)
                parallel_logger.start_processing(total_individuals)
                
                try:
                    # 并行处理 - 子进程只返回简单数据
                    results_with_logs = Parallel(
                        n_jobs=config.n_jobs,
                        backend=config.backend,
                        batch_size=config.batch_size,
                        verbose=0,  # 禁用joblib日志
                        max_nbytes=50*1024*1024,  # 限制在worker之间传输的数据大小为50MB（减少内存使用）
                        mmap_mode='r',  # 使用内存映射模式读取数据
                        timeout=None,  # 不设置超时限制
                        temp_folder=None  # 使用系统默认临时文件夹
                    )(
                        delayed(_safe_worker_function)(func, individual_id, *args, **kwargs)
                        for individual_id in individual_list
                    )
                    
                    # 分离结果和日志数据
                    results = []
                    all_worker_data = []
                    
                    for result_data in results_with_logs:
                        if isinstance(result_data, dict) and 'worker_log_data' in result_data:
                            # 包含日志数据的新格式
                            results.append(result_data.get('result'))
                            all_worker_data.append(result_data['worker_log_data'])
                        else:
                            # 旧格式兼容
                            results.append(result_data)
                    
                    # 在主进程中聚合日志数据
                    if all_worker_data:
                        parallel_logger.aggregate_worker_data(all_worker_data)
                    
                    # 完成处理
                    parallel_logger.finish_processing()
                    
                    # 记录总体统计
                    elapsed_time = time.time() - parallel_logger.start_time
                    if elapsed_time > 0:
                        rate = total_individuals / elapsed_time
                        current_logger.info(f"处理完成: {total_individuals} 个体, "
                                          f"耗时: {elapsed_time:.2f}s, 速度: {rate:.2f} 个体/秒")
                    
                    return results
                    
                except Exception as e:
                    current_logger.error(f"并行处理失败，回退到串行模式: {e}")
                    # 回退到串行处理
                    return func(individual_list, *args, **kwargs)
        
        return wrapper
    return decorator


def _safe_worker_function(func: Callable, individual_id: Any, *args, **kwargs):
    """
    安全的单个个体处理函数
    
    核心特点：
    1. 在子进程中创建可pickle的日志数据
    2. 捕获异常并记录到简单数据结构中
    3. 返回结果和日志数据
    """
    # 创建工作进程日志数据（可pickle）
    worker_data = create_safe_worker_logger()
    
    start_time = time.time()
    
    try:
        # 调用原始函数处理单个个体
        result = func([individual_id], *args, **kwargs)
        
        # 处理结果格式
        if isinstance(result, list) and len(result) == 1:
            final_result = result[0]
        else:
            final_result = result
        
        processing_time = time.time() - start_time
        
        # 尝试从结果中提取缓存信息
        cache_hit = None
        if isinstance(final_result, dict):
            cache_hit = final_result.get('cache_hit')
        
        # 记录成功处理
        log_worker_progress(
            worker_data=worker_data,
            individual_id=individual_id,
            success=True,
            cache_hit=cache_hit,
            processing_time=processing_time
        )
        
        # 返回结果和日志数据
        return {
            'result': final_result,
            'worker_log_data': worker_data
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        # 记录错误
        log_worker_progress(
            worker_data=worker_data,
            individual_id=individual_id,
            success=False,
            error_msg=error_msg,
            processing_time=processing_time
        )
        
        # 返回错误标记和日志数据
        return {
            'result': {"error": error_msg, "individual_id": individual_id},
            'worker_log_data': worker_data
        }


def create_simple_parallel_config(n_jobs: int = 1) -> LightweightParallelConfig:
    """创建简单的并行配置"""
    return LightweightParallelConfig(n_jobs=n_jobs)