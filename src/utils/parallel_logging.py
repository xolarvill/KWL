"""
并行化日志管理模块
确保并行处理时的日志输出清晰有序
"""

import logging
import threading
import time
import queue
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import numpy as np
import weakref


class ParallelLogger:
    """
    并行化日志管理器
    
    功能：
    1. 进程/线程隔离的日志收集
    2. 统一的进度显示
    3. 统计信息聚合
    4. 错误信息隔离
    """
    
    def __init__(self, main_logger: logging.Logger, verbose: bool = False):
        self.main_logger = main_logger
        self.verbose = verbose
        
        # 线程本地存储
        self._local = threading.local()
        
        # 统计信息
        self.stats_lock = threading.Lock()
        self.worker_stats = defaultdict(lambda: {
            'processed': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': None,
            'end_time': None
        })
        
        # 进度信息
        self.progress_info = {
            'total': 0,
            'completed': 0,
            'start_time': None,
            'last_update': None
        }
        
        # 错误队列
        self.error_queue = queue.Queue()
        
        # 详细日志缓冲区（可选）
        self.debug_logs = defaultdict(list) if verbose else None
        
    def start_processing(self, total_items: int, worker_id: Optional[str] = None):
        """开始处理批次"""
        if worker_id is None:
            worker_id = self._get_worker_id()
            
        with self.stats_lock:
            self.progress_info['total'] = total_items
            self.progress_info['start_time'] = time.time()
            self.progress_info['last_update'] = time.time()
            
            self.worker_stats[worker_id]['start_time'] = time.time()
            
        self.main_logger.info(f"开始并行处理 {total_items} 个个体")
        
    def log_worker_start(self, worker_id: str, item_count: int):
        """记录工作进程开始"""
        if self.verbose:
            self.main_logger.debug(f"工作进程 {worker_id}: 开始处理 {item_count} 个个体")
            
    def log_individual_processed(self, worker_id: str, individual_id: Any, 
                                success: bool = True, error_msg: str = None,
                                cache_hit: bool = None, processing_time: float = None):
        """记录个体处理完成"""
        with self.stats_lock:
            self.progress_info['completed'] += 1
            self.worker_stats[worker_id]['processed'] += 1
            
            if not success:
                self.worker_stats[worker_id]['errors'] += 1
                self.error_queue.put({
                    'worker_id': worker_id,
                    'individual_id': individual_id,
                    'error': error_msg,
                    'timestamp': time.time()
                })
            
            if cache_hit is not None:
                if cache_hit:
                    self.worker_stats[worker_id]['cache_hits'] += 1
                else:
                    self.worker_stats[worker_id]['cache_misses'] += 1
        
        # 记录详细日志（如果启用）
        if self.verbose and self.debug_logs is not None:
            self.debug_logs[worker_id].append({
                'individual_id': individual_id,
                'success': success,
                'cache_hit': cache_hit,
                'processing_time': processing_time,
                'timestamp': time.time()
            })
        
        # 定期输出进度
        self._check_progress_update()
        
    def _check_progress_update(self, force: bool = False):
        """检查是否需要更新进度显示"""
        current_time = time.time()
        
        with self.stats_lock:
            # 检查是否已经初始化
            if self.progress_info['start_time'] is None:
                return
                
            # 检查更新频率限制
            last_update = self.progress_info.get('last_update')
            if not force and last_update is not None and (current_time - last_update) < 2.0:
                return  # 每2秒最多更新一次
                
            completed = self.progress_info['completed']
            total = self.progress_info['total']
            start_time = self.progress_info['start_time']
            
            if total == 0 or start_time is None:
                return
                
            progress_percent = (completed / total) * 100
            elapsed_time = current_time - start_time
            
            if completed > 0 and elapsed_time > 0:
                processing_rate = completed / elapsed_time
                remaining_items = total - completed
                estimated_remaining = remaining_items / processing_rate if processing_rate > 0 else 0
            else:
                processing_rate = 0
                estimated_remaining = 0
            
            self.progress_info['last_update'] = current_time
        
        # 输出进度信息
        if completed < total or force:
            self.main_logger.info(
                f"进度: {completed}/{total} ({progress_percent:.1f}%), "
                f"速度: {processing_rate:.1f} 个体/秒, "
                f"预计剩余: {estimated_remaining:.1f}秒"
            )
            
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        with self.stats_lock:
            total_processed = sum(stats['processed'] for stats in self.worker_stats.values())
            total_errors = sum(stats['errors'] for stats in self.worker_stats.values())
            total_cache_hits = sum(stats['cache_hits'] for stats in self.worker_stats.values())
            total_cache_misses = sum(stats['cache_misses'] for stats in self.worker_stats.values())
            
            total_cache_requests = total_cache_hits + total_cache_misses
            cache_hit_rate = total_cache_hits / total_cache_requests if total_cache_requests > 0 else 0
            
            # 计算各worker的详细统计
            worker_details = {}
            for worker_id, stats in self.worker_stats.items():
                worker_cache_rate = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])) \
                                  if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                
                worker_details[worker_id] = {
                    'processed': stats['processed'],
                    'errors': stats['errors'],
                    'error_rate': stats['errors'] / stats['processed'] if stats['processed'] > 0 else 0,
                    'cache_hit_rate': worker_cache_rate
                }
            
            return {
                'total_processed': total_processed,
                'total_errors': total_errors,
                'overall_error_rate': total_errors / total_processed if total_processed > 0 else 0,
                'cache_hit_rate': cache_hit_rate,
                'worker_details': worker_details,
                'processing_time': time.time() - self.progress_info['start_time'] \
                                  if self.progress_info['start_time'] else 0
            }
    
    def finish_processing(self):
        """完成处理，输出最终统计"""
        # 强制最终进度更新
        self._check_progress_update(force=True)
        
        # 输出聚合统计
        stats = self.get_aggregated_stats()
        
        self.main_logger.info(f"并行处理完成: {stats['total_processed']} 个个体")
        
        # 避免除零错误
        processing_time = stats['processing_time']
        if processing_time > 0:
            self.main_logger.info(f"总耗时: {processing_time:.1f}秒")
            self.main_logger.info(f"平均速度: {stats['total_processed']/processing_time:.1f} 个体/秒")
        else:
            self.main_logger.info("处理时间过短，无法准确统计")
        
        if stats['total_errors'] > 0:
            self.main_logger.warning(f"处理错误: {stats['total_errors']} 个 ({stats['overall_error_rate']:.1%})")
        
        self.main_logger.info(f"缓存命中率: {stats['cache_hit_rate']:.1%}")
        
        # 输出worker详细信息（如果启用详细模式）
        if self.verbose:
            self.main_logger.info("各工作进程详细统计:")
            for worker_id, worker_stats in stats['worker_details'].items():
                self.main_logger.info(
                    f"  {worker_id}: 处理{worker_stats['processed']}个, "
                    f"错误率{worker_stats['error_rate']:.1%}, "
                    f"缓存命中率{worker_stats['cache_hit_rate']:.1%}"
                )
        
        # 输出错误汇总
        errors = []
        while not self.error_queue.empty():
            try:
                error_info = self.error_queue.get_nowait()
                errors.append(error_info)
            except queue.Empty:
                break
        
        if errors:
            self.main_logger.warning(f"错误汇总 ({len(errors)} 个错误):")
            for error in errors[:5]:  # 只显示前5个错误
                self.main_logger.warning(
                    f"  工作进程 {error['worker_id']}, 个体 {error['individual_id']}: {error['error']}"
                )
            if len(errors) > 5:
                self.main_logger.warning(f"  ... 还有 {len(errors) - 5} 个错误")
    
    def _get_worker_id(self) -> str:
        """获取当前工作进程ID"""
        thread_id = threading.current_thread().ident
        return f"worker_{thread_id}"
    
    def get_debug_logs(self, worker_id: str = None) -> List[Dict[str, Any]]:
        """获取详细调试日志"""
        if not self.verbose or self.debug_logs is None:
            return []
        
        if worker_id:
            return self.debug_logs.get(worker_id, [])
        else:
            # 返回所有worker的日志，按时间排序
            all_logs = []
            for logs in self.debug_logs.values():
                all_logs.extend(logs)
            return sorted(all_logs, key=lambda x: x['timestamp'])


class QuietParallelLogger(ParallelLogger):
    """
    安静模式的并行日志管理器
    只输出重要信息，适合生产环境
    """
    
    def __init__(self, main_logger: logging.Logger):
        super().__init__(main_logger, verbose=False)
        
    def _check_progress_update(self, force: bool = False):
        """减少进度更新频率"""
        current_time = time.time()
        
        with self.stats_lock:
            # 每10秒或重要里程碑更新一次
            if not force and current_time - self.progress_info['last_update'] < 10.0:
                return
                
            completed = self.progress_info['completed']
            total = self.progress_info['total']
            
            # 只在重要里程碑更新
            milestones = [0.25, 0.5, 0.75, 1.0]
            current_progress = completed / total if total > 0 else 0
            
            should_update = False
            for milestone in milestones:
                if current_progress >= milestone and \
                   (self.progress_info.get('last_milestone', 0) < milestone):
                    self.progress_info['last_milestone'] = milestone
                    should_update = True
                    break
            
            if not (should_update or force):
                return
                
            self.progress_info['last_update'] = current_time
        
        # 调用父类的进度更新
        super()._check_progress_update(force)