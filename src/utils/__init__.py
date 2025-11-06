"""
Utility modules for the project
"""

from .lightweight_parallel_logging import (
    SimpleParallelLogger, WorkerLogData, 
    create_safe_worker_logger, log_worker_progress
)

from .lightweight_parallel_wrapper import (
    lightweight_parallel_processor, LightweightParallelConfig,
    create_simple_parallel_config, _safe_worker_function
)

from .parallel_migration import (
    safe_parallel_wrapper, create_compatible_parallel_config,
    configure_parallel_system, migrate_to_new_system,
    get_recommended_parallel_system, test_parallel_system
)

# 为了保持向后兼容，也导出旧的接口
# 但推荐使用新的轻量级系统
try:
    from .parallel_logging import ParallelLogger, QuietParallelLogger
    from .parallel_wrapper import parallel_individual_processor, ParallelConfig
    from .parallel_logger_registry import (
        register_parallel_logger, get_parallel_logger, 
        unregister_parallel_logger
    )
    OLD_SYSTEM_AVAILABLE = True
except ImportError:
    OLD_SYSTEM_AVAILABLE = False

# 版本信息
__version__ = "2.0.0"

# 推荐使用的新系统标志
RECOMMENDED_SYSTEM = "lightweight"

# 便捷的入口点
def get_parallel_processor(prefer_new_system: bool = True):
    """
    获取推荐的并行处理器
    
    Args:
        prefer_new_system: 是否优先使用新系统
        
    Returns:
        并行处理器装饰器
    """
    if prefer_new_system:
        return safe_parallel_wrapper()
    elif OLD_SYSTEM_AVAILABLE:
        return parallel_individual_processor()
    else:
        raise ImportError("没有可用的并行处理器")

# 自动配置推荐系统
def auto_configure():
    """自动配置最优的并行系统"""
    from .parallel_migration import get_recommended_parallel_system
    
    if get_recommended_parallel_system():
        # 使用新系统
        configure_parallel_system(use_new_system=True)
        return "lightweight"
    else:
        # 使用旧系统（不推荐）
        if OLD_SYSTEM_AVAILABLE:
            return "legacy"
        else:
            raise RuntimeError("没有可用的并行系统")

# 模块级别的便捷函数
def parallel_processor(func=None, *, prefer_new: bool = True, **kwargs):
    """
    通用的并行处理器装饰器
    
    用法：
        @parallel_processor
        def my_function(items):
            ...
            
        @parallel_processor(prefer_new=True, quiet_mode=False)
        def my_verbose_function(items):
            ...
    """
    if func is None:
        # 带参数的装饰器
        def decorator(f):
            if prefer_new:
                return safe_parallel_wrapper(**kwargs)(f)
            else:
                if OLD_SYSTEM_AVAILABLE:
                    return parallel_individual_processor(**kwargs)(f)
                else:
                    raise ImportError("旧系统不可用")
        return decorator
    else:
        # 无参数的装饰器
        if prefer_new:
            return safe_parallel_wrapper()(func)
        else:
            if OLD_SYSTEM_AVAILABLE:
                return parallel_individual_processor()(func)
            else:
                raise ImportError("旧系统不可用")