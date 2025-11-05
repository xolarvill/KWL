"""
并行日志管理器注册表
解决日志管理器无法序列化的问题
"""

import threading
import weakref
from typing import Optional, Dict, Any


class ParallelLoggerRegistry:
    """
    全局并行日志管理器注册表
    
    使用线程ID作为键，避免直接传递日志管理器对象，
    从而解决pickle序列化问题。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loggers = {}
        return cls._instance
    
    def register_logger(self, logger: Any) -> str:
        """
        注册日志管理器，返回注册ID
        
        Args:
            logger: 并行日志管理器实例
            
        Returns:
            注册ID（基于线程ID）
        """
        import threading
        thread_id = threading.current_thread().ident
        logger_id = f"logger_{thread_id}"
        
        with self._lock:
            self._loggers[logger_id] = weakref.ref(logger)
        
        return logger_id
    
    def get_logger(self, logger_id: str) -> Optional[Any]:
        """
        根据ID获取日志管理器
        
        Args:
            logger_id: 日志管理器ID
            
        Returns:
            日志管理器实例，如果不存在或已被回收则返回None
        """
        with self._lock:
            logger_ref = self._loggers.get(logger_id)
            if logger_ref is None:
                return None
            
            logger = logger_ref()
            if logger is None:
                # 清理已回收的引用
                del self._loggers[logger_id]
                return None
            
            return logger
    
    def unregister_logger(self, logger_id: str):
        """注销日志管理器"""
        with self._lock:
            self._loggers.pop(logger_id, None)
    
    def clear_all(self):
        """清理所有注册"""
        with self._lock:
            self._loggers.clear()


# 全局注册表实例
_logger_registry = ParallelLoggerRegistry()


def get_parallel_logger_registry() -> ParallelLoggerRegistry:
    """获取全局并行日志注册表"""
    return _logger_registry


def register_parallel_logger(logger: Any) -> str:
    """便捷函数：注册并行日志管理器"""
    return _logger_registry.register_logger(logger)


def get_parallel_logger(logger_id: str) -> Optional[Any]:
    """便捷函数：获取并行日志管理器"""
    return _logger_registry.get_logger(logger_id)


def unregister_parallel_logger(logger_id: str):
    """便捷函数：注销并行日志管理器"""
    _logger_registry.unregister_logger(logger_id)