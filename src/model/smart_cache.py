"""
智能缓存系统 - 解决缓存效率问题

吸取之前bug的教训：
1. 避免重复缓存检查逻辑
2. 正确使用缓存键（包含individual_id）
3. 正确使用缓存属性遍历
4. 保持向后兼容性
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
from src.model.likelihood import LRUCache

logger = logging.getLogger(__name__)


class SmartCacheKey:
    """智能缓存键生成器 - 避免过度敏感"""
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        # 关键参数定义 - 这些参数对Bellman方程解影响最大
        self.critical_params = [
            'gamma_0', 'gamma_1', 'gamma_2', 'sigma_epsilon', 
            'cost_distance', 'cost_adjacent', 'beta'
        ]
        
    def create_key(self, individual_id: str, params: Dict[str, Any], 
                   agent_type: int) -> Tuple:
        """创建智能缓存键 - 记住之前的教训，保持individual_id"""
        
        # 1. 提取关键参数（避免使用所有参数）
        critical_values = {}
        for param in self.critical_params:
            if param in params:
                value = params[param]
                # 参数离散化 - 减少敏感度
                discrete_value = self._discretize_value(value)
                critical_values[param] = discrete_value
        
        # 2. 创建参数摘要（减少键长度）
        param_summary = self._create_param_hash(critical_values)
        
        # 3. 构建缓存键（保持individual_id - 吸取之前的教训）
        cache_key = (individual_id, param_summary, agent_type)
        
        logger.debug(f"智能缓存键创建: {individual_id}, 参数摘要={param_summary}, 类型={agent_type}")
        return cache_key
    
    def _discretize_value(self, value: float) -> float:
        """将连续参数值离散化"""
        if abs(value) < 1e-10:
            return 0.0
        
        # 使用相对容差进行离散化
        if abs(value) > 1.0:
            return round(value / self.tolerance) * self.tolerance
        else:
            # 对小值使用更精细的离散化
            precision = max(1, int(-np.log10(self.tolerance)))
            return round(value, precision)
    
    def _create_param_hash(self, critical_values: Dict[str, float]) -> int:
        """创建参数摘要 - 减少键长度同时保持区分度"""
        # 排序确保一致性
        sorted_items = tuple(sorted(critical_values.items()))
        # 使用哈希但限制范围避免过大数值
        return hash(sorted_items) % 100000  # 限制在0-99999


class SimilarityMatcher:
    """相似性匹配器 - 用于L2缓存"""
    
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        
    def find_similar_solution(self, target_params: Dict[str, Any], 
                            candidates: Dict, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """在候选解中寻找最相似的解决方案"""
        
        if not candidates:
            return None
            
        best_match = None
        best_similarity = -1
        
        for cached_key, (cached_params, cached_solution) in candidates.items():
            
            # 检查形状匹配（吸取之前的教训 - 确保正确检查）
            if not (isinstance(cached_solution, np.ndarray) and 
                   cached_solution.shape[0] == target_shape[0]):
                continue
                
            # 计算参数相似性
            similarity = self._calculate_similarity(target_params, cached_params)
            
            if similarity > best_similarity and similarity >= self.threshold:
                best_similarity = similarity
                best_match = cached_solution
                
        if best_match is not None:
            logger.debug(f"找到相似解，相似度={best_similarity:.3f}")
            
        return best_match
    
    def _calculate_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """计算两个参数字典的相似性（0-1）"""
        
        # 找到共同参数
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            val1, val2 = params1[key], params2[key]
            
            # 处理数值参数
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1) > 1e-10 and abs(val2) > 1e-10:
                    # 相对差异
                    rel_diff = abs(val1 - val2) / max(abs(val1), abs(val2))
                    similarity = max(0.0, 1.0 - rel_diff)
                else:
                    # 绝对差异
                    abs_diff = abs(val1 - val2)
                    similarity = max(0.0, 1.0 - abs_diff * 10)  # 缩放因子
                    
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0


class EnhancedBellmanCache:
    """增强版Bellman缓存系统 - 吸取之前所有教训"""
    
    def __init__(self, capacity: int = 2000, memory_limit_mb: int = 2000):
        # 大幅增加缓存容量 - 从500提升到2000
        self.l1_cache = LRUCache(capacity=capacity, max_memory_mb=memory_limit_mb)
        
        # L2相似性缓存
        self.l2_cache = {}
        self.l2_capacity = capacity // 2  # L2容量为L1的一半
        
        # 相似性匹配器
        self.similarity_matcher = SimilarityMatcher(threshold=0.15)
        
        # 智能键生成器
        self.key_generator = SmartCacheKey(tolerance=0.01)
        
        # 性能统计
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'similarity_hits': 0,
            'total_requests': 0
        }
        
        # 只在主进程初始化时记录日志，避免并行处理时日志刷屏
        # 使用环境变量来控制是否输出日志
        import os
        if os.environ.get('CACHE_QUIET_MODE', '').lower() != 'true':
            logger.info(f"增强缓存系统初始化: L1容量={capacity}, L2容量={self.l2_capacity}, 内存限制={memory_limit_mb}MB")
    
    def get(self, individual_id: str, params: Dict[str, Any], 
            agent_type: int, solution_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """获取缓存解 - 三层查询策略"""
        
        self.stats['total_requests'] += 1
        
        # 第1层：精确匹配（智能键）
        smart_key = self.key_generator.create_key(individual_id, params, agent_type)
        result = self.l1_cache.get(smart_key)
        
        if result is not None:
            self.stats['l1_hits'] += 1
            logger.debug(f"L1缓存命中: {individual_id}, 类型={agent_type}")
            return result
        
        self.stats['l1_misses'] += 1
        
        # 第2层：相似性匹配（吸取之前的教训 - 正确使用缓存属性）
        if self.l2_cache:
            result = self.similarity_matcher.find_similar_solution(
                params, self.l2_cache, solution_shape
            )
            
            if result is not None:
                self.stats['l2_hits'] += 1
                self.stats['similarity_hits'] += 1
                logger.debug(f"L2相似性命中: {individual_id}, 类型={agent_type}")
                
                # 升级到L1缓存
                self.l1_cache.put(smart_key, result)
                return result
        
        self.stats['l2_misses'] += 1
        logger.debug(f"缓存未命中: {individual_id}, 类型={agent_type}")
        return None
    
    def put(self, individual_id: str, params: Dict[str, Any], 
            agent_type: int, solution: np.ndarray) -> None:
        """存储解到缓存"""
        
        # 存储到L1（精确缓存）
        smart_key = self.key_generator.create_key(individual_id, params, agent_type)
        self.l1_cache.put(smart_key, solution)
        
        # 存储到L2（相似性缓存）- 控制数量避免内存溢出
        if len(self.l2_cache) < self.l2_capacity:
            self.l2_cache[smart_key] = (params.copy(), solution.copy())
        else:
            # LRU策略：移除最旧的（简单实现）
            oldest_key = next(iter(self.l2_cache.keys()))
            del self.l2_cache[oldest_key]
            self.l2_cache[smart_key] = (params.copy(), solution.copy())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取详细的缓存统计"""
        total = self.stats['total_requests']
        l1_total = self.stats['l1_hits'] + self.stats['l1_misses']
        l2_total = self.stats['l2_hits'] + self.stats['l2_misses']
        
        l1_hit_rate = self.stats['l1_hits'] / l1_total if l1_total > 0 else 0
        l2_hit_rate = self.stats['l2_hits'] / l2_total if l2_total > 0 else 0
        total_hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits']) / total if total > 0 else 0
        
        # 获取L1缓存的详细统计
        l1_stats = self.l1_cache.get_stats() if hasattr(self.l1_cache, 'get_stats') else {}
        
        return {
            'total_requests': total,
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'total_hit_rate': total_hit_rate,
            'similarity_hits': self.stats['similarity_hits'],
            'l1_stats': l1_stats,
            'l2_cache_size': len(self.l2_cache),
            'l2_capacity': self.l2_capacity
        }
    
    def clear_stats(self) -> None:
        """清除统计信息"""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("缓存统计已重置")


# 向后兼容：创建函数用于替换原有的LRUCache
def create_enhanced_cache(capacity: int = 2000, memory_limit_mb: int = 2000) -> EnhancedBellmanCache:
    """创建增强版缓存系统 - 保持API兼容"""
    # 记录主进程PID，用于避免并行处理时日志刷屏
    import os
    if '_main_pid' not in globals():
        globals()['_main_pid'] = os.getpid()
    
    return EnhancedBellmanCache(capacity=capacity, memory_limit_mb=memory_limit_mb)