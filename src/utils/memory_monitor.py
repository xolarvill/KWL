"""
内存监控和管理工具
用于在并行处理过程中监控内存使用情况并自动调整参数
"""

import psutil
import logging
import os
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控器，用于实时监控内存使用情况"""
    
    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.95):
        """
        初始化内存监控器
        
        Args:
            warning_threshold: 警告阈值（内存使用百分比）
            critical_threshold: 严重阈值（内存使用百分比）
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.peak_memory_percent = 0.0
        self.peak_process_memory_mb = 0.0
        
    def start_monitoring(self, interval: float = 5.0):
        """开始监控内存使用"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"内存监控已启动，警告阈值: {self.warning_threshold:.1%}, 严重阈值: {self.critical_threshold:.1%}")
    
    def stop_monitoring(self):
        """停止监控内存使用"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("内存监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                memory_info = self.get_memory_info()
                current_percent = memory_info['memory_percent']
                current_process_mb = memory_info['process_memory_mb']
                
                # 更新峰值
                self.peak_memory_percent = max(self.peak_memory_percent, current_percent)
                self.peak_process_memory_mb = max(self.peak_process_memory_mb, current_process_mb)
                
                # 检查阈值
                if current_percent > self.critical_threshold:
                    logger.warning(f"严重内存警告：系统内存使用率达到 {current_percent:.1%}!")
                elif current_percent > self.warning_threshold:
                    logger.info(f"内存使用警告：系统内存使用率达到 {current_percent:.1%}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"内存监控出错: {e}")
                time.sleep(interval)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取当前内存使用信息"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': memory_info.rss / virtual_memory.total * 100,
            'memory_percent': virtual_memory.percent,
            'available_memory_mb': virtual_memory.available / 1024 / 1024,
            'total_memory_mb': virtual_memory.total / 1024 / 1024,
            'peak_memory_percent': self.peak_memory_percent,
            'peak_process_memory_mb': self.peak_process_memory_mb
        }
    
    def is_memory_safe(self) -> bool:
        """检查当前内存使用是否安全"""
        memory_info = self.get_memory_info()
        return memory_info['memory_percent'] < self.warning_threshold
    
    def get_recommended_parallel_jobs(self, base_jobs: int, n_individuals: int) -> int:
        """
        根据内存使用情况推荐并行工作进程数
        
        Args:
            base_jobs: 基础工作进程数
            n_individuals: 个体数量
            
        Returns:
            推荐的工作进程数
        """
        memory_info = self.get_memory_info()
        available_memory_gb = memory_info['available_memory_mb'] / 1024
        
        # 根据可用内存计算安全的工作进程数
        if available_memory_gb < 4:  # 少于4GB可用内存
            safe_jobs = 1
        elif available_memory_gb < 8:  # 4-8GB可用内存
            safe_jobs = min(2, base_jobs)
        elif available_memory_gb < 16:  # 8-16GB可用内存
            safe_jobs = min(4, base_jobs)
        else:  # 16GB以上可用内存
            safe_jobs = min(8, base_jobs)
        
        # 根据个体数量进一步调整
        if n_individuals > 10000:
            safe_jobs = min(safe_jobs, max(1, n_individuals // 2000))
        elif n_individuals > 5000:
            safe_jobs = min(safe_jobs, max(1, n_individuals // 1500))
        
        return safe_jobs
    
    def get_recommended_batch_size(self, n_individuals: int) -> int:
        """
        根据内存使用情况推荐批次大小
        
        Args:
            n_individuals: 个体数量
            
        Returns:
            推荐的批次大小
        """
        memory_info = self.get_memory_info()
        available_memory_gb = memory_info['available_memory_mb'] / 1024
        
        if available_memory_gb < 4:
            base_batch_size = 500
        elif available_memory_gb < 8:
            base_batch_size = 1000
        elif available_memory_gb < 16:
            base_batch_size = 2000
        else:
            base_batch_size = 5000
        
        # 大样本时减小批次大小
        if n_individuals > 20000:
            base_batch_size = min(base_batch_size, 1000)
        elif n_individuals > 10000:
            base_batch_size = min(base_batch_size, 1500)
        
        return base_batch_size


def create_memory_safe_config(base_parallel_config, n_individuals: int, 
                            memory_safe_mode: bool = True) -> Optional[Dict[str, Any]]:
    """
    创建内存安全的配置
    
    Args:
        base_parallel_config: 基础并行配置
        n_individuals: 个体数量
        memory_safe_mode: 是否启用内存安全模式
        
    Returns:
        内存安全配置字典，如果不需要调整则返回None
    """
    if not memory_safe_mode:
        return None
    
    monitor = MemoryMonitor()
    
    # 获取推荐配置
    recommended_jobs = monitor.get_recommended_parallel_jobs(
        base_parallel_config.n_jobs if base_parallel_config else 1, 
        n_individuals
    )
    recommended_batch_size = monitor.get_recommended_batch_size(n_individuals)
    
    config = {
        'n_jobs': recommended_jobs,
        'batch_size': recommended_batch_size,
        'memory_limit_mb': min(1000, 500 if n_individuals > 10000 else 1000)
    }
    
    memory_info = monitor.get_memory_info()
    logger.info(f"内存安全模式配置: 工作进程={recommended_jobs}, 批次大小={recommended_batch_size}, "
                f"可用内存={memory_info['available_memory_mb']:.1f}MB")
    
    return config


def log_memory_usage(stage: str = ""):
    """记录当前内存使用情况"""
    try:
        monitor = MemoryMonitor()
        memory_info = monitor.get_memory_info()
        stage_info = f" [{stage}]" if stage else ""
        logger.info(f"内存使用{stage_info}: 进程内存={memory_info['process_memory_mb']:.1f}MB, "
                    f"系统内存={memory_info['memory_percent']:.1f}%, "
                    f"可用内存={memory_info['available_memory_mb']:.1f}MB")
    except Exception as e:
        logger.debug(f"无法获取内存信息: {e}")