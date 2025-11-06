"""
并行日志系统迁移模块
提供从旧系统到新系统的平滑过渡
"""

import logging
import warnings
from typing import Callable, Optional, Any
from .parallel_wrapper import parallel_individual_processor as old_parallel_processor
from .lightweight_parallel_wrapper import lightweight_parallel_processor, create_simple_parallel_config
from .parallel_logging import QuietParallelLogger, ParallelLogger
from .lightweight_parallel_logging import SimpleParallelLogger


logger = logging.getLogger(__name__)


class CompatibilityWrapper:
    """
    兼容性包装器
    提供旧接口但内部使用新实现
    """
    
    def __init__(self, use_new_system: bool = True, quiet_mode: bool = True):
        self.use_new_system = use_new_system
        self.quiet_mode = quiet_mode
        
    def parallel_processor(self, config_getter: Optional[Callable] = None,
                          logger: Optional[logging.Logger] = None,
                          use_quiet_mode: bool = True):
        """
        兼容性并行处理器装饰器
        
        Args:
            config_getter: 获取配置的函数
            logger: 日志记录器
            use_quiet_mode: 是否使用安静模式
        """
        if self.use_new_system:
            # 使用新系统，但保持接口兼容
            return lightweight_parallel_processor(
                config_getter=config_getter,
                logger=logger,
                quiet_mode=use_quiet_mode
            )
        else:
            # 回退到旧系统
            warnings.warn("使用旧的并行处理器，可能存在Windows兼容性问题", DeprecationWarning)
            return old_parallel_processor(
                config_getter=config_getter,
                logger=logger,
                use_quiet_mode=use_quiet_mode
            )


def create_compatible_parallel_config(n_jobs: int = 1, backend: str = 'loky', 
                                    use_new_system: bool = True, **kwargs):
    """
    创建兼容的并行配置
    
    Args:
        n_jobs: 并行任务数
        backend: 并行后端
        use_new_system: 是否使用新系统
        **kwargs: 其他参数
        
    Returns:
        并行配置对象
    """
    if use_new_system:
        # 新系统配置
        from .lightweight_parallel_wrapper import LightweightParallelConfig
        return LightweightParallelConfig(n_jobs=n_jobs, backend=backend)
    else:
        # 旧系统配置
        from .parallel_wrapper import ParallelConfig
        return ParallelConfig(n_jobs=n_jobs, backend=backend, **kwargs)


def safe_parallel_wrapper(use_new_system: bool = True, quiet_mode: bool = True):
    """
    安全的并行包装器工厂函数
    
    推荐使用这个函数来创建并行处理器，它会根据系统选择最优方案
    
    Args:
        use_new_system: 是否强制使用新系统（推荐True）
        quiet_mode: 是否使用安静模式
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        if use_new_system:
            # 使用新的轻量级系统
            wrapped = lightweight_parallel_processor(
                config_getter=lambda: create_simple_parallel_config(),
                quiet_mode=quiet_mode
            )(func)
            
            # 添加标识以便调试
            wrapped._parallel_system = 'new'
            return wrapped
        else:
            # 使用旧的系统（不推荐）
            warnings.warn(
                "使用旧的并行系统，在Windows上可能存在pickle序列化问题。"
                "建议设置 use_new_system=True", 
                UserWarning
            )
            wrapped = old_parallel_processor(
                config_getter=lambda: create_compatible_parallel_config(use_new_system=False),
                use_quiet_mode=quiet_mode
            )(func)
            wrapped._parallel_system = 'old'
            return wrapped
    
    return decorator


# 全局配置
_global_config = {
    'use_new_system': True,  # 默认使用新系统
    'quiet_mode': True,
    'auto_detect_windows': True
}


def configure_parallel_system(use_new_system: bool = None, 
                            quiet_mode: bool = None,
                            auto_detect_windows: bool = None):
    """
    配置并行系统行为
    
    Args:
        use_new_system: 是否使用新系统
        quiet_mode: 默认安静模式
        auto_detect_windows: 是否自动检测Windows并使用新系统
    """
    global _global_config
    
    if use_new_system is not None:
        _global_config['use_new_system'] = use_new_system
    
    if quiet_mode is not None:
        _global_config['quiet_mode'] = quiet_mode
        
    if auto_detect_windows is not None:
        _global_config['auto_detect_windows'] = auto_detect_windows
    
    logger.info(f"并行系统配置更新: {_global_config}")


def get_recommended_parallel_system():
    """
    获取推荐的并行系统
    
    自动检测运行环境并推荐最适合的并行系统
    """
    import platform
    
    # 如果用户明确指定了，使用用户的设置
    if not _global_config['auto_detect_windows']:
        return _global_config['use_new_system']
    
    # 自动检测Windows系统
    if platform.system() == 'Windows':
        logger.info("检测到Windows系统，推荐使用新的轻量级并行系统")
        return True
    
    # 其他系统也推荐使用新系统（更稳定）
    return True


def create_migration_report():
    """
    创建迁移报告，帮助用户了解系统变化
    """
    report = """
并行日志系统迁移报告
========================

问题背景：
- 旧系统在Windows上存在pickle序列化问题，导致并行处理失败
- 复杂的日志对象无法在Windows的spawn模式下传递
- 回退到串行模式后出现ZeroDivisionError

解决方案：
- 新的轻量级系统使用可pickle的简单数据结构
- 子进程只收集数据，日志聚合在主进程完成
- 完全避免序列化问题，跨平台兼容性更好

迁移建议：
1. 推荐使用 safe_parallel_wrapper() 替代旧的装饰器
2. 新系统默认启用安静模式，减少日志输出
3. 在Windows系统上自动使用新系统
4. 旧系统仍然可用，但会有警告提示

兼容性：
- 接口基本保持不变，可以平滑迁移
- 新增了配置选项和更好的错误处理
- 性能影响极小，稳定性显著提升
"""
    return report


# 便捷的迁移函数
def migrate_to_new_system():
    """
    一键迁移到新系统
    """
    configure_parallel_system(use_new_system=True, auto_detect_windows=True)
    logger.info("已切换到新的轻量级并行系统")
    logger.info("使用 safe_parallel_wrapper() 装饰器来获得最佳兼容性")


def test_parallel_system():
    """
    测试并行系统是否正常工作
    """
    import time
    
    def dummy_function(items):
        """测试函数"""
        results = []
        for item in items:
            time.sleep(0.001)  # 模拟处理时间
            results.append(f"processed_{item}")
        return results
    
    # 使用新系统测试
    new_wrapper = safe_parallel_wrapper(use_new_system=True)
    wrapped_function = new_wrapper(dummy_function)
    
    try:
        test_items = list(range(10))
        results = wrapped_function(test_items)
        logger.info(f"新系统测试成功，处理 {len(results)} 个items")
        return True
    except Exception as e:
        logger.error(f"新系统测试失败: {e}")
        return False