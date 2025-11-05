#!/usr/bin/env python3
"""
序列化问题调试脚本
"""

import pickle
import sys
import traceback
sys.path.insert(0, '.')

def test_serialization(obj, name):
    """测试对象的序列化能力"""
    try:
        pickle.dumps(obj)
        print(f"✅ {name} 可以序列化")
        return True
    except Exception as e:
        print(f"❌ {name} 序列化失败: {type(e).__name__}: {str(e)[:200]}...")
        return False

def debug_individual_processing():
    """调试个体处理函数的序列化问题"""
    print("=== 个体处理序列化调试 ===")
    
    # 模拟实际场景
    from src.estimation.e_step_individual_processor import process_single_individual_e_step
    
    # 创建测试数据
    import pandas as pd
    import numpy as np
    from src.model.smart_cache import create_enhanced_cache
    from src.utils.parallel_logger_registry import ParallelLoggerRegistry
    
    # 1. 测试基础函数
    print("\n1. 测试基础函数:")
    test_serialization(process_single_individual_e_step, "process_single_individual_e_step")
    
    # 2. 测试各个参数
    print("\n2. 测试各个参数类型:")
    
    # DataFrame
    df = pd.DataFrame({'individual_id': [1], 'data': [2]})
    test_serialization(df, "DataFrame")
    
    # 数组
    test_serialization(np.array([1, 2, 3]), "numpy array")
    
    # 列表
    omega_list = [{'eta': 1.0, 'sigma': 0.5}]
    test_serialization(omega_list, "omega_list")
    
    # 参数字典
    params = {'gamma_0': 0.1}
    test_serialization(params, "params dict")
    
    # 3. 测试缓存对象
    print("\n3. 测试缓存对象:")
    bellman_cache = create_enhanced_cache()
    test_serialization(bellman_cache, "bellman_cache")
    
    # 4. 测试函数调用组合
    print("\n4. 测试函数调用组合:")
    try:
        # 模拟实际的函数调用参数
        args = (
            'individual_123',  # individual_id
            df,                # individual_data
            omega_list,        # omega_list
            np.array([0.7]),   # omega_probs
            params,            # params
            np.array([0.33, 0.33, 0.34]),  # pi_k
            3,                 # K
            0.95,              # beta
            {},                # transition_matrices
            {},                # regions_df
            np.array([[1, 2], [3, 4]]),     # distance_matrix
            np.array([[0, 1], [1, 0]]),     # adjacency_matrix
            {1: 0, 2: 1},      # prov_to_idx
            bellman_cache,     # bellman_cache
            {'cache_hits': 0, 'cache_misses': 0}  # cache_stats
        )
        
        pickle.dumps(args)
        print("✅ 函数参数组合可以序列化")
    except Exception as e:
        print(f"❌ 函数参数组合序列化失败: {type(e).__name__}: {str(e)[:200]}...")
        
        # 找出具体哪个参数有问题
        print("\n   逐个测试参数:")
        for i, arg in enumerate(args):
            arg_type = type(arg).__name__
            success = test_serialization(arg, f"   参数{i} ({arg_type})")
            if not success:
                print(f"   🎯 问题参数: 参数{i} ({arg_type})")

def test_parallel_logger_system():
    """测试并行日志系统"""
    print("\n\n=== 并行日志系统调试 ===")
    
    import logging
    from src.utils.parallel_logging import QuietParallelLogger
    from src.utils.parallel_logger_registry import register_parallel_logger, get_parallel_logger
    
    # 测试日志管理器本身
    logger = logging.getLogger('test')
    parallel_logger = QuietParallelLogger(logger)
    
    print("1. 测试日志管理器:")
    test_serialization(parallel_logger, "parallel_logger")
    
    # 测试注册表系统
    print("\n2. 测试注册表系统:")
    registry = register_parallel_logger(parallel_logger)
    test_serialization(registry, "logger_registry_id")
    
    # 测试通过ID获取
    retrieved_logger = get_parallel_logger(registry)
    print(f"3. 检索到的日志管理器: {retrieved_logger is not None}")

def test_actual_wrapper_function():
    """测试实际的包装函数"""
    print("\n\n=== 实际包装函数调试 ===")
    
    from src.utils.parallel_wrapper import _process_single_individual_wrapper
    
    # 创建一个简单的测试函数
    def test_func(individual_list, *args, **kwargs):
        return [f"processed {ind}" for ind in individual_list]
    
    # 测试包装函数本身
    print("1. 测试包装函数:")
    test_serialization(_process_single_individual_wrapper, "_process_single_individual_wrapper")
    
    # 测试包装函数的调用
    print("\n2. 测试包装函数调用:")
    try:
        # 模拟实际的调用
        call_args = (test_func, 'individual_123')
        call_kwargs = {'parallel_logger_id': 'logger_12345'}
        
        pickle.dumps((_process_single_individual_wrapper, call_args, call_kwargs))
        print("✅ 包装函数调用可以序列化")
    except Exception as e:
        print(f"❌ 包装函数调用序列化失败: {type(e).__name__}: {str(e)[:200]}...")
        
        # 详细分析
        print("\n   详细分析:")
        test_serialization(test_func, "   test_func")
        test_serialization(call_args, "   call_args")
        test_serialization(call_kwargs, "   call_kwargs")

if __name__ == '__main__':
    debug_individual_processing()
    test_parallel_logger_system()
    test_actual_wrapper_function()