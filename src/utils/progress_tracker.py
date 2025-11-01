"""
轻量级进度跟踪和断点续跑工具
"""
import json
import os
import pickle
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressTracker:
    """轻量级进度跟踪器，支持断点续跑"""
    
    def __init__(self, 
                 progress_dir: str = "progress",
                 task_name: str = "estimation",
                 save_interval: int = 10,
                 use_json: bool = True):
        """
        初始化进度跟踪器
        
        Args:
            progress_dir: 进度文件保存目录
            task_name: 任务名称，用于区分不同任务的进度文件
            save_interval: 每多少步保存一次进度
            use_json: 是否使用JSON格式保存（否则使用pickle）
        """
        self.progress_dir = Path(progress_dir)
        self.task_name = task_name
        self.save_interval = save_interval
        self.use_json = use_json
        self.progress_file = self.progress_dir / f"{task_name}_progress.json" if use_json else self.progress_dir / f"{task_name}_progress.pkl"
        
        # 确保进度目录存在
        self.progress_dir.mkdir(exist_ok=True)
        
        # 当前状态
        self.step_count = 0
        self.start_time = None
        self.current_step = ""
        self.completed_steps = []
        self.step_results = {}
        self.is_resumed = False
        self.last_save_time = 0
        
    def load_progress(self) -> bool:
        """
        加载已有的进度
        
        Returns:
            bool: 是否成功加载了进度
        """
        if not self.progress_file.exists():
            logger.info(f"未找到进度文件: {self.progress_file}")
            return False
            
        try:
            if self.use_json:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(self.progress_file, 'rb') as f:
                    data = pickle.load(f)
            
            # 恢复状态
            self.step_count = data.get('step_count', 0)
            self.start_time = data.get('start_time', time.time())
            self.current_step = data.get('current_step', '')
            self.completed_steps = data.get('completed_steps', [])
            self.step_results = data.get('step_results', {})
            self.is_resumed = True
            
            logger.info(f"成功加载进度: 已完成 {len(self.completed_steps)} 步")
            logger.info(f"已完成的步骤: {self.completed_steps}")
            return True
            
        except Exception as e:
            logger.warning(f"加载进度文件失败: {e}")
            return False
    
    def save_progress(self, force: bool = False):
        """
        保存当前进度
        
        Args:
            force: 是否强制保存（忽略保存间隔）
        """
        current_time = time.time()
        
        # 检查保存间隔
        if not force and (current_time - self.last_save_time) < self.save_interval:
            return
            
        try:
            data = {
                'step_count': self.step_count,
                'start_time': self.start_time or current_time,
                'current_step': self.current_step,
                'completed_steps': self.completed_steps,
                'step_results': self.step_results,
                'last_update': current_time,
                'task_name': self.task_name
            }
            
            if self.use_json:
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(self.progress_file, 'wb') as f:
                    pickle.dump(data, f)
                    
            self.last_save_time = current_time
            logger.debug(f"进度已保存: {self.current_step}")
            
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def start_step(self, step_name: str, step_info: str = ""):
        """
        开始一个新步骤
        
        Args:
            step_name: 步骤名称
            step_info: 步骤的额外信息
        """
        self.current_step = step_name
        self.step_count += 1
        
        if self.start_time is None:
            self.start_time = time.time()
            
        logger.info(f"[{self.step_count}] 开始步骤: {step_name}")
        if step_info:
            logger.info(f"  └─ {step_info}")
            
        # 定期保存
        if self.step_count % self.save_interval == 0:
            self.save_progress()
    
    def complete_step(self, step_name: str, result: Any = None):
        """
        完成一个步骤
        
        Args:
            step_name: 步骤名称
            result: 步骤的结果（可选）
        """
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
            
        if result is not None:
            self.step_results[step_name] = result
            
        logger.info(f"✓ 完成步骤: {step_name}")
        self.save_progress()
    
    def is_step_completed(self, step_name: str) -> bool:
        """
        检查步骤是否已完成
        
        Args:
            step_name: 步骤名称
            
        Returns:
            bool: 步骤是否已完成
        """
        return step_name in self.completed_steps
    
    def get_step_result(self, step_name: str) -> Any:
        """
        获取步骤的结果
        
        Args:
            step_name: 步骤名称
            
        Returns:
            步骤的结果，如果不存在则返回None
        """
        return self.step_results.get(step_name)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        获取进度摘要
        
        Returns:
            进度摘要信息
        """
        elapsed_time = time.time() - (self.start_time or time.time())
        return {
            'task_name': self.task_name,
            'total_steps': self.step_count,
            'completed_steps': len(self.completed_steps),
            'current_step': self.current_step,
            'elapsed_time': elapsed_time,
            'is_resumed': self.is_resumed
        }
    
    def cleanup(self):
        """清理进度文件"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                logger.info(f"进度文件已清理: {self.progress_file}")
        except Exception as e:
            logger.warning(f"清理进度文件失败: {e}")


def checkpoint(step_name: str, 
               save_interval: int = 10,
               progress_dir: str = "progress",
               task_name: str = "estimation"):
    """
    装饰器：为函数添加进度跟踪和断点续跑功能
    
    Args:
        step_name: 步骤名称
        save_interval: 保存间隔
        progress_dir: 进度文件目录
        task_name: 任务名称
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建进度跟踪器
            tracker = ProgressTracker(
                progress_dir=progress_dir,
                task_name=task_name,
                save_interval=save_interval
            )
            
            # 尝试加载已有进度
            has_progress = tracker.load_progress()
            
            # 检查步骤是否已完成
            if has_progress and tracker.is_step_completed(step_name):
                logger.info(f"步骤 '{step_name}' 已存在且已完成，跳过执行")
                # 返回已保存的结果（如果存在）
                saved_result = tracker.get_step_result(step_name)
                if saved_result is not None:
                    return saved_result
                else:
                    logger.warning(f"步骤 '{step_name}' 已完成但没有保存结果，重新执行")
            
            # 开始新步骤
            tracker.start_step(step_name)
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 完成步骤并保存结果
                tracker.complete_step(step_name, result)
                
                return result
                
            except Exception as e:
                # 异常时保存当前状态
                logger.error(f"步骤 '{step_name}' 执行失败: {e}")
                tracker.save_progress(force=True)
                raise
                
        return wrapper
    return decorator


class ProgressContext:
    """上下文管理器：管理进度跟踪的上下文"""
    
    def __init__(self, 
                 task_name: str = "estimation",
                 progress_dir: str = "progress",
                 save_interval: int = 10,
                 auto_cleanup: bool = False):
        """
        初始化进度上下文
        
        Args:
            task_name: 任务名称
            progress_dir: 进度文件目录
            save_interval: 保存间隔
            auto_cleanup: 完成后是否自动清理进度文件
        """
        self.tracker = ProgressTracker(
            progress_dir=progress_dir,
            task_name=task_name,
            save_interval=save_interval
        )
        self.auto_cleanup = auto_cleanup
        
    def __enter__(self):
        """进入上下文"""
        self.tracker.load_progress()
        return self.tracker
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if exc_type is not None:
            # 异常退出时强制保存
            logger.error(f"任务异常退出: {exc_val}")
            self.tracker.save_progress(force=True)
        
        if self.auto_cleanup:
            self.tracker.cleanup()
        
        return False  # 不抑制异常


def get_progress_summary(task_name: str = "estimation", progress_dir: str = "progress") -> Optional[Dict[str, Any]]:
    """
    获取指定任务的进度摘要
    
    Args:
        task_name: 任务名称
        progress_dir: 进度文件目录
        
    Returns:
        进度摘要，如果不存在则返回None
    """
    tracker = ProgressTracker(progress_dir=progress_dir, task_name=task_name)
    if tracker.load_progress():
        return tracker.get_progress_summary()
    return None