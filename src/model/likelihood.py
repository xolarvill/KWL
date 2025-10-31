"""
This module constructs the log-likelihood function for the structural model.
This version is refactored for performance, decoupling Bellman solution from likelihood calculation.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple, Optional
from numba import jit, float64, int64, prange

from src.model.utility import calculate_flow_utility_vectorized, calculate_flow_utility_individual_vectorized
from src.model.wage_equation import calculate_wage_likelihood

@jit(nopython=True)
def _compute_ccp_jit(choice_specific_values):
    """JIT编译的条件选择概率计算"""
    n_states, n_choices = choice_specific_values.shape
    ccps = np.zeros((n_states, n_choices))
    
    for i in range(n_states):
        max_v = np.max(choice_specific_values[i, :])
        exp_v = np.exp(choice_specific_values[i, :] - max_v)
        sum_exp_v = np.sum(exp_v)
        for j in range(n_choices):
            ccps[i, j] = exp_v[j] / max(sum_exp_v, 1e-300)
    
    return ccps

# Global cache for Bellman solutions
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity=100, max_memory_mb: int = 500):
        self.cache = OrderedDict()
        self.frequency = {}  # 添加频率统计
        self.capacity = capacity
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.logger = logging.getLogger(__name__)

    def _estimate_memory_usage(self, key, value) -> int:
        """估算键值对的内存使用"""
        import sys
        # 键的内存使用（元组 + 内容）
        key_memory = sys.getsizeof(key)
        if isinstance(key, tuple):
            for item in key:
                key_memory += sys.getsizeof(item)
                if isinstance(item, tuple):
                    for sub_item in item:
                        key_memory += sys.getsizeof(sub_item)
        
        # 值的内存使用（numpy数组或普通值）
        if isinstance(value, np.ndarray):
            value_memory = value.nbytes
        else:
            value_memory = sys.getsizeof(value)
            
        return key_memory + value_memory

    def get(self, key):
        self.access_count += 1
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        # 移动到末尾（LRU）并更新频率
        self.cache.move_to_end(key)
        self.frequency[key] = self.frequency.get(key, 0) + 1
        self.hit_count += 1
        return self.cache[key]

    def put(self, key, value):
        # 估算新条目的内存使用
        new_memory = self._estimate_memory_usage(key, value)
        
        # 如果键已存在，先删除旧的
        if key in self.cache:
            old_value = self.cache.pop(key)
            old_memory = self._estimate_memory_usage(key, old_value)
            self.current_memory_bytes -= old_memory
            # 保留频率信息
        
        # 检查是否需要淘汰条目（大小或内存限制）
        while (len(self.cache) >= self.capacity or 
               self.current_memory_bytes + new_memory > self.max_memory_bytes):
            if not self.cache:
                break  # 缓存为空，无法继续淘汰
            self._remove_lfu()
        
        # 添加新条目
        self.cache[key] = value
        self.cache.move_to_end(key)  # 移到末尾（最近使用）
        self.frequency[key] = self.frequency.get(key, 0) + 1
        self.current_memory_bytes += new_memory
        
        self.logger.debug(f"缓存添加: {key}, 内存使用: {new_memory / 1024 / 1024:.2f} MB, "
                         f"总内存: {self.current_memory_bytes / 1024 / 1024:.2f} MB")

    def _remove_lfu(self):
        """移除最少使用的项（LFU策略）"""
        if self.frequency:
            # 找到访问频率最低的键
            lfu_key = min(self.frequency.keys(), key=lambda k: self.frequency[k])
            
            # 更新内存统计
            old_memory = self._estimate_memory_usage(lfu_key, self.cache[lfu_key])
            self.current_memory_bytes -= old_memory
            self.eviction_count += 1
            
            # 删除键
            del self.cache[lfu_key]
            del self.frequency[lfu_key]
            
            self.logger.debug(f"LFU淘汰: {lfu_key}, 释放内存: {old_memory / 1024 / 1024:.2f} MB")

    def get_stats(self):
        """获取缓存统计信息"""
        hit_rate = self.hit_count / self.access_count if self.access_count > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.capacity,
            'memory_mb': self.current_memory_bytes / 1024 / 1024,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            'access_count': self.access_count,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'eviction_count': self.eviction_count
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.frequency.clear()
        self.current_memory_bytes = 0
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.logger.info("缓存已清空")

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache

_BELLMAN_CACHE = LRUCache(capacity=100, max_memory_mb=500)  # 保留最近100组参数，最大500MB内存

def _make_cache_key(params: Dict[str, Any], agent_type: int) -> Tuple:
    """
    Creates a hashable cache key from parameters and agent type.

    Uses rounded parameter values to avoid cache misses from numerical noise
    during optimization (e.g., L-BFGS-B's finite difference gradient calculation).
    """
    # Only include structural parameters, exclude n_choices
    # Round to 6 decimal places to avoid excessive cache misses from tiny perturbations
    # Use sorted items for consistent ordering
    param_items = tuple((k, round(v, 6)) for k, v in sorted(params.items()) if k != 'n_choices')
    return (agent_type, param_items)

def clear_bellman_cache():
    """
    Clears the global Bellman solution cache.
    """
    global _BELLMAN_CACHE
    _BELLMAN_CACHE.clear()

def calculate_likelihood_from_v(
    converged_v: np.ndarray,
    params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    wage_predicted: Optional[np.ndarray] = None,
    include_wage_likelihood: bool = True
) -> np.ndarray:
    """
    Calculates the log-likelihood for each observation given a converged value function.

    根据论文公式(1162-1165行)，联合似然 = 选择概率 × 工资密度:
        p(j_t, w_t | x_t) = P(j_t | x_t) × Ψ(w_t | j_t, x_t)

    新增参数:
    --------
    wage_predicted : np.ndarray, optional
        预测工资数组 (shape: n_obs,)。如果提供且include_wage_likelihood=True,
        将计算工资似然并加到总似然中
    include_wage_likelihood : bool, default=True
        是否包含工资似然。设为False可向后兼容（仅选择似然）

    Returns:
        np.ndarray: A vector of log-likelihoods for each observation.
    """
    logger = logging.getLogger()
    n_states, n_choices = len(state_space), params["n_choices"]

    # This part is now a self-contained CCP calculator
    state_space_np = {col: state_space[col].to_numpy() for col in state_space.columns}
    numeric_cols = regions_df.select_dtypes(include=np.number).columns.tolist()
    regions_df_indexed = regions_df.set_index('provcd')
    unique_provcds = sorted(regions_df['provcd'].unique())
    regions_df_np = {
        col: regions_df_indexed.loc[unique_provcds][col].to_numpy()
        for col in numeric_cols if col != 'provcd'
    }
    regions_df_np['provcd'] = unique_provcds

    flow_utility_matrix = calculate_flow_utility_vectorized(
        state_data=state_space_np, region_data=regions_df_np,
        distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
        params=params, agent_type=agent_type, n_states=n_states, n_choices=n_choices
    )

    expected_future_value_matrix = np.zeros((n_states, n_choices))
    for j_idx in range(n_choices):
        P_j = transition_matrices[j_idx]
        expected_future_value_matrix[:, j_idx] = P_j @ converged_v
        
    choice_specific_values = flow_utility_matrix + beta * expected_future_value_matrix

    # 使用JIT优化的CCP计算
    ccps = _compute_ccp_jit(choice_specific_values)
    ccps = np.maximum(ccps, 1e-15)

    state_indices = observed_data['state_index'].values
    choice_indices = observed_data['choice_index'].values

    if np.any(state_indices >= ccps.shape[0]) or np.any(choice_indices >= ccps.shape[1]):
        logger.error("Error: Invalid state or choice index detected.")
        return np.full(len(observed_data), -100.0)

    # 1. 计算选择概率的对数似然
    choice_probs = ccps[state_indices, choice_indices]
    log_choice_likelihood = np.log(choice_probs)

    # 2. 计算工资似然（如果提供了工资数据）
    if include_wage_likelihood and wage_predicted is not None and 'income' in observed_data.columns:
        # 提取观测工资
        wage_observed = observed_data['income'].values

        # 获取暂态冲击的标准差参数
        sigma_epsilon = params.get('sigma_epsilon', 0.5)  # 默认值，应该被估计

        # 计算工资对数似然密度 (论文746-750行)
        log_wage_likelihood = calculate_wage_likelihood(
            w_observed=wage_observed,
            w_predicted=wage_predicted,
            sigma_epsilon=sigma_epsilon
        )

        # 联合似然 = 选择似然 + 工资似然 (在对数空间中相加)
        total_log_likelihood = log_choice_likelihood + log_wage_likelihood

        # 记录诊断信息
        if np.random.rand() < 0.001:  # 偶尔记录，避免日志过多
            logger.debug(f"Mean log_choice_lik: {np.mean(log_choice_likelihood):.4f}, "
                        f"Mean log_wage_lik: {np.mean(log_wage_likelihood):.4f}")

    else:
        # 向后兼容：仅使用选择似然
        total_log_likelihood = log_choice_likelihood

    return total_log_likelihood

def calculate_likelihood_from_v_individual(
    converged_v_individual: np.ndarray,
    params: Dict[str, Any],
    individual_data: pd.DataFrame,
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    prov_to_idx: Dict[int, int] = None
) -> np.ndarray:
    """
    Calculates the log-likelihood for a single individual's observations
    using a pre-solved, compact value function.
    
    Args:
        converged_v_individual: Converged value function for this individual
        params: Model parameters
        individual_data: Data for this individual
        agent_type: Agent type
        beta: Discount factor
        transition_matrices: Transition matrices
        regions_df: Regional data
        distance_matrix: Distance matrix
        adjacency_matrix: Adjacency matrix
        prov_to_idx: Province code to index mapping (optional)
    """
    logger = logging.getLogger()
    
    # --- 1. Reconstruct individual's state space and V-function structure ---
    visited_locations = individual_data['visited_locations'].iloc[0]
    location_map = individual_data['location_map'].iloc[0]
    n_visited_locations = len(visited_locations)
    ages = sorted(individual_data['age_t'].unique())
    age_map = {age: i for i, age in enumerate(ages)}
    n_ages = len(ages)

    # --- 2. Calculate Choice Probabilities (CCPs) for all states of this individual ---
    # The state is (age, prev_loc_compact_idx)
    
    # Create the state_data dict for all states this individual could be in
    state_ages = np.repeat(ages, n_visited_locations)
    state_prev_locs = np.tile(np.arange(n_visited_locations), n_ages)
    
    state_data_individual = {
        'age': state_ages,
        'prev_provcd_idx': state_prev_locs, # Compact indices
        'hukou_prov_idx': np.full(len(state_ages), individual_data['hukou_prov_idx'].iloc[0]),
        'hometown_prov_idx': np.full(len(state_ages), individual_data['hometown_prov_idx'].iloc[0])
    }

    # a. Calculate flow utility
    flow_utility_matrix = calculate_flow_utility_individual_vectorized(
        state_data=state_data_individual,
        region_data=regions_df, # Passed as numpy dict
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        params=params,
        agent_type=agent_type,
        n_states=(n_ages * n_visited_locations),
        n_choices=params['n_choices'],
        visited_locations=visited_locations,
        prov_to_idx=prov_to_idx
    )

    # b. Calculate expected future value
    future_v_global = np.zeros(params['n_choices'])
    for global_loc_id, compact_idx in location_map.items():
        # This mapping is tricky. V is indexed by (age_idx, loc_idx)
        # We need V for age t+1.
        pass # This logic is now inside the Bellman solve, not here.

    # We already have the converged V, so we use it directly.
    # The CCP calculation needs EV, which is P @ V_next.
    # This logic is duplicated from the Bellman solve, which is correct.
    
    expected_future_value_matrix = np.zeros_like(flow_utility_matrix)
    for age_idx, age in enumerate(ages):
        if age == max(ages):
            future_v_for_age = np.zeros(n_visited_locations)
        else:
            next_age_idx = age_map[age + 1]
            future_v_for_age = converged_v_individual[next_age_idx * n_visited_locations : (next_age_idx + 1) * n_visited_locations]
        
        future_v_global = np.zeros(params['n_choices'])
        for global_loc_id, compact_idx in location_map.items():
            # 安全检查：确保future_v_for_age不为空且索引有效
            if len(future_v_for_age) == 0:
                logger.warning(f"future_v_for_age is empty for age {age}, using zero future value")
                continue
            if int(compact_idx) >= len(future_v_for_age):
                logger.warning(f"Compact index {compact_idx} out of bounds for future_v_for_age with length {len(future_v_for_age)}")
                continue
                
            if prov_to_idx is not None:
                # Convert global_loc_id to the correct index using prov_to_idx
                global_idx = prov_to_idx.get(int(global_loc_id))
                if global_idx is not None:
                    future_v_global[global_idx] = future_v_for_age[int(compact_idx)]
                else:
                    logger.warning(f"Province code {global_loc_id} not found in prov_to_idx mapping")
            else:
                # Fallback for old behavior or tests that don't provide the map
                logger.warning("prov_to_idx not provided. Assuming global_loc_id is a valid index.")
                future_v_global[int(global_loc_id)] = future_v_for_age[int(compact_idx)]
            
        start_idx = age_idx * n_visited_locations
        end_idx = start_idx + n_visited_locations
        expected_future_value_matrix[start_idx:end_idx, :] = np.tile(future_v_global, (n_visited_locations, 1))

    # c. Calculate CCPs
    choice_specific_values = flow_utility_matrix + beta * expected_future_value_matrix
    max_v = np.max(choice_specific_values, axis=1, keepdims=True)
    exp_v = np.exp(choice_specific_values - max_v)
    sum_exp_v = np.sum(exp_v, axis=1, keepdims=True)
    ccps_individual = exp_v / np.maximum(sum_exp_v, 1e-300)
    ccps_individual = np.maximum(ccps_individual, 1e-15)

    # --- 3. Look up probabilities for observed choices ---
    # Map observed states to the row index of ccps_individual
    obs_ages = individual_data['age_t'].values
    obs_prev_loc_global = individual_data['prev_provcd'].values
    
    # Convert global prev_loc to compact index
    obs_prev_loc_compact = np.array([location_map.get(loc, -1) for loc in obs_prev_loc_global])
    
    # Convert age to age_idx
    obs_age_idx = np.array([age_map.get(age, -1) for age in obs_ages])

    # Calculate state index for the individual's state space
    # state_idx = age_idx * n_visited_locations + prev_loc_compact_idx
    valid_obs = (obs_prev_loc_compact != -1) & (obs_age_idx != -1)
    
    # **新增**: 记录无效观测值的数量
    n_invalid_obs = np.sum(~valid_obs)
    if n_invalid_obs > 0:
        logger.warning(
            f"Found {n_invalid_obs} invalid observations for individual "
            f"(ID hash: {hash(individual_data['individual_id'].iloc[0]) % 10000}). "
            f"Prev_loc or age not in individual's compact state space."
        )

    state_indices = obs_age_idx[valid_obs] * n_visited_locations + obs_prev_loc_compact[valid_obs]
    
    choice_indices = individual_data['choice_index'].values[valid_obs]

    # Look up choice probabilities
    choice_probs = ccps_individual[state_indices, choice_indices]
    log_choice_likelihood = np.log(choice_probs)

    # For now, we only handle choice likelihood
    # A full implementation would also calculate and add wage likelihood here.
    
    # Create a full-length result vector
    # **修改**: 将惩罚值从-1e10改为-100
    total_log_likelihood = np.full(len(individual_data), -100.0) # Default for invalid obs
    total_log_likelihood[valid_obs] = log_choice_likelihood

    return total_log_likelihood

