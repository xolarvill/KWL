"""
轻量级并行日志管理模块
专门解决Windows下的pickle序列化问题
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class WorkerLogData:
    """可序列化的工作进程日志数据 - 可在子进程中安全创建"""
    worker_id: str
    processed_count: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_error(self, individual_id: str, error_msg: str):
        """添加错误信息"""
        self.errors.append({
            'individual_id': individual_id,
            'error': error_msg,
            'timestamp': time.time()
        })


class SimpleParallelLogger:
    """
    极简并行日志管理器
    
    核心设计原则：
    1. 子进程只收集数据，不直接操作日志对象
    2. 所有复杂日志操作都在主进程中统一处理
    3. 使用可pickle的简单数据结构
    """
    
    def __init__(self, main_logger: logging.Logger, quiet_mode: bool = True, disable_stats_output: bool = False):
        self.main_logger = main_logger
        self.quiet_mode = quiet_mode
        self.disable_stats_output = disable_stats_output
        self.start_time = None
        self.total_items = 0
        self.worker_data = {}  # worker_id -> WorkerLogData
        
        # 进度跟踪
        self.last_progress_time = 0
        self.progress_interval = 10.0 if quiet_mode else 2.0  # 安静模式10秒更新一次
        
    def start_processing(self, total_items: int):
        """开始处理批次"""
        self.start_time = time.time()
        self.total_items = total_items
        self.main_logger.info(f"开始处理 {total_items} 个个体")
        self._output_progress(force=True)
        
    def create_worker_data(self, worker_id: str) -> WorkerLogData:
        """为工作进程创建日志数据对象（在子进程中调用）"""
        return WorkerLogData(
            worker_id=worker_id,
            start_time=time.time()
        )
        
    def _output_progress(self, force: bool = False):
        """输出进度信息（仅在主进程中调用）"""
        if not self.main_logger:
            return
            
        current_time = time.time()
        if not force and (current_time - self.last_progress_time) < self.progress_interval:
            return
            
        self.last_progress_time = current_time
        
        # 聚合数据
        total_processed = sum(data.processed_count for data in self.worker_data.values())
        total_errors = sum(data.error_count for data in self.worker_data.values())
        
        if self.total_items > 0:
            progress_percent = (total_processed / self.total_items) * 100
            elapsed_time = current_time - self.start_time if self.start_time else 0
            
            if elapsed_time > 0 and total_processed > 0:
                processing_rate = total_processed / elapsed_time
                remaining_items = self.total_items - total_processed
                estimated_remaining = remaining_items / processing_rate if processing_rate > 0 else 0
                
                if not self.quiet_mode or force or progress_percent >= 100:
                    self.main_logger.info(
                        f"进度: {total_processed}/{self.total_items} ({progress_percent:.1f}%), "
                        f"速度: {processing_rate:.1f} 个体/秒, "
                        f"预计剩余: {estimated_remaining:.1f}秒"
                    )
            else:
                if not self.quiet_mode or force:
                    self.main_logger.info(f"进度: {total_processed}/{self.total_items} ({progress_percent:.1f}%), 计算速率中...")
        
    def aggregate_worker_data(self, all_worker_data: List[WorkerLogData]):
        """聚合所有工作进程的数据（在主进程中调用）"""
        self.worker_data = {data.worker_id: data for data in all_worker_data}
        
    def finish_processing(self):
        """完成处理，输出最终统计"""
        self._output_progress(force=True)
        
        # 【修复】如果禁用统计输出，直接返回，避免与E-step/M-step统计重复
        if self.disable_stats_output:
            return
            
        # 聚合统计
        total_processed = sum(data.processed_count for data in self.worker_data.values())
        total_errors = sum(data.error_count for data in self.worker_data.values())
        total_cache_hits = sum(data.cache_hits for data in self.worker_data.values())
        total_cache_misses = sum(data.cache_misses for data in self.worker_data.values())
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # 输出最终统计
        self.main_logger.info(f"处理完成: {total_processed} 个个体")
        
        if total_time > 0:
            self.main_logger.info(f"总耗时: {total_time:.1f}秒")
            if total_processed > 0:
                self.main_logger.info(f"平均速度: {total_processed/total_time:.1f} 个体/秒")
        
        if total_errors > 0:
            error_rate = total_errors / total_processed if total_processed > 0 else 0
            self.main_logger.warning(f"处理错误: {total_errors} 个 ({error_rate:.1%})")
        
        if total_cache_hits + total_cache_misses > 0:
            cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)
            self.main_logger.info(f"缓存命中率: {cache_hit_rate:.1%}")
        
        # 输出错误汇总（限制数量避免日志过多）
        all_errors = []
        for worker_data in self.worker_data.values():
            all_errors.extend(worker_data.errors)
        
        if all_errors:
            self.main_logger.warning(f"错误汇总 ({len(all_errors)} 个错误):")
            for error in all_errors[:3]:  # 只显示前3个错误
                self.main_logger.warning(f"  工作进程 {error['worker_id']}, 个体 {error['individual_id']}: {error['error']}")
            if len(all_errors) > 3:
                self.main_logger.warning(f"  ... 还有 {len(all_errors) - 3} 个错误")


def create_safe_worker_logger() -> WorkerLogData:
    """
    在子进程中创建安全的日志数据对象
    
    Returns:
        WorkerLogData: 可安全pickle的日志数据对象
    """
    worker_id = f"worker_{threading.current_thread().ident}"
    return WorkerLogData(worker_id=worker_id, start_time=time.time())


def log_worker_progress(worker_data: WorkerLogData, individual_id: str, 
                       success: bool = True, error_msg: str = None,
                       cache_hit: bool = None, processing_time: float = None):
    """
    在子进程中记录处理进度
    
    Args:
        worker_data: 工作进程日志数据对象
        individual_id: 个体ID
        success: 是否成功处理
        error_msg: 错误信息（如果失败）
        cache_hit: 缓存命中情况
        processing_time: 处理耗时
    """
    worker_data.processed_count += 1
    
    if not success:
        worker_data.error_count += 1
        worker_data.add_error(individual_id, error_msg or "Unknown error")
    
    if cache_hit is not None:
        if cache_hit:
            worker_data.cache_hits += 1
        else:
            worker_data.cache_misses += 1
    
    worker_data.end_time = time.time()