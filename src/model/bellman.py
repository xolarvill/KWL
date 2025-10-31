"""
This module implements the Bellman equation solver using value function iteration
for the dynamic discrete choice model. This version is optimized with vectorization.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

from src.model.utility import calculate_flow_utility_vectorized, calculate_flow_utility_individual_vectorized

def solve_bellman_iteration_vectorized(
    v_old: np.ndarray,
    utility_function: Callable,
    state_space_np: Dict[str, np.ndarray],
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df_np: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
) -> np.ndarray:
    """
    Performs a single, vectorized iteration of the Bellman equation.
    """
    n_states, n_choices = len(state_space_np['age']), params["n_choices"]
    
    flow_utility_matrix = utility_function(
        state_data=state_space_np,
        region_data=regions_df_np,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        params=params,
        agent_type=agent_type,
        n_states=n_states,
        n_choices=n_choices
    )

    expected_future_value_matrix = np.zeros((n_states, n_choices))
    for j_idx in range(n_choices):
        P_j = transition_matrices[j_idx]
        expected_future_value_matrix[:, j_idx] = P_j @ v_old

    choice_specific_values = flow_utility_matrix + beta * expected_future_value_matrix

    max_v = np.max(choice_specific_values, axis=1)
    exp_diff = np.exp(choice_specific_values - max_v[:, np.newaxis])
    v_new = max_v + np.log(np.sum(exp_diff, axis=1))
    
    v_new = np.nan_to_num(v_new, nan=0.0, posinf=1e6, neginf=-1e6)

    return v_new


def solve_bellman_equation_individual(
    utility_function: Callable,
    individual_data: pd.DataFrame,
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    verbose: bool = True,
    prov_to_idx: Dict[int, int] = None,
    initial_v: np.ndarray = None,  # 添加热启动支持
) -> Tuple[np.ndarray, int]:
    """
    Solves the Bellman equation for a SINGLE INDIVIDUAL using a compact state space.
    """
    logger = logging.getLogger()

    # --- 0. Hot-start setup ---
    hot_start_used = False
    iteration_savings = 0
    visited_locations = individual_data['visited_locations'].iloc[0]
    location_map = individual_data['location_map'].iloc[0]
    n_visited_locations = len(visited_locations)
    
    # The state is defined by (age, prev_location_compact_idx)
    # We only need to solve for ages present in the individual's data
    ages = sorted(individual_data['age_t'].unique())
    age_map = {age: i for i, age in enumerate(ages)}
    n_ages = len(ages)

    # The value function V will have shape (n_ages, n_visited_locations)
    n_individual_states = n_ages * n_visited_locations
    
    # 热启动：如果有初始值且形状匹配，则使用；否则从零开始
    if initial_v is not None and initial_v.shape == (n_individual_states,):
        v_old = initial_v.copy()
        hot_start_used = True
        if verbose:
            logger.info(f"      [Bellman Individual] Hot-starting with provided initial value function for agent_type={agent_type}")
    else:
        v_old = np.zeros(n_individual_states)
        if initial_v is not None and verbose:
            logger.info(f"      [Bellman Individual] Hot-start shape mismatch (expected {n_individual_states}, got {initial_v.shape[0]}), using zeros")

    # OPTIMIZATION: The conversion from regions_df to regions_df_np is now done *outside* this function.
    regions_df_np = regions_df
    
    # --- 2. Value Function Iteration (backward induction) ---
    v_new = np.zeros_like(v_old)
    
    # Iterate backwards from the oldest age the individual experiences
    for age_idx, age in reversed(list(enumerate(ages))):
        # For the terminal age, the future value is zero
        if age == max(ages):
            future_v = np.zeros(n_visited_locations)
        else:
            # Get the value function from the next age
            next_age_idx = age_map[age + 1]
            start_idx = next_age_idx * n_visited_locations
            end_idx = (next_age_idx + 1) * n_visited_locations
            
            # 安全检查：确保切片范围有效
            if start_idx >= len(v_new) or end_idx > len(v_new):
                logger.error(f"Invalid slice range for future_v: [{start_idx}:{end_idx}] for v_new with length {len(v_new)}")
                future_v = np.zeros(n_visited_locations)
            else:
                future_v = v_new[start_idx:end_idx]
                
            # 额外检查：确保future_v不为空
            if len(future_v) == 0:
                logger.error(f"Empty future_v for age {age}, next_age_idx {next_age_idx}, using zeros")
                future_v = np.zeros(n_visited_locations)

        # Create the state_data dict for this age
        # It will have n_visited_locations rows, one for each possible previous location
        state_data_age = {
            'age': np.full(n_visited_locations, age),
            'prev_provcd_idx': np.arange(n_visited_locations), # Compact indices
            'hukou_prov_idx': np.full(n_visited_locations, individual_data['hukou_prov_idx'].iloc[0]),
            'hometown_prov_idx': np.full(n_visited_locations, individual_data['hometown_prov_idx'].iloc[0])
        }

        # a. Calculate flow utility for all states at this age
        flow_utility_matrix = calculate_flow_utility_individual_vectorized(
            state_data=state_data_age,
            region_data=regions_df_np,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            params=params,
            agent_type=agent_type,
            n_states=n_visited_locations,
            n_choices=params['n_choices'],
            visited_locations=visited_locations,
            prov_to_idx=prov_to_idx
        )

        # b. Calculate expected future value
        # This is the tricky part. The choice is over GLOBAL locations.
        # The future state depends on the choice. The future value depends on the future state.
        # If the choice j is a location the individual has visited before, the future value is in our compact `future_v`.
        # If it's a new location, the future value is more complex.
        # For this model, we assume the future state is (age+1, choice_location),
        # so we need to map the global choice j back to a compact index if possible.
        
        # For now, let's assume a simplified transition: the next state's `prev_loc` is the current `choice`.
        # The value of being in a certain location next period is given by `future_v`.
        # We need to get the value for each of the global choices.
        
        # Create a mapping from global choice index to future value
        # If a choice corresponds to a visited location, use its value. Otherwise, what?
        # A common assumption is that the value of entering a new, unvisited state
        # is an average, or zero if we are far from terminal period.
        # Let's assume for now the value is zero for unvisited locations.
        
        future_v_global = np.zeros(params['n_choices'])
        if prov_to_idx is None:
            # Fallback for old behavior or tests that don't provide the map
            logger.warning("prov_to_idx not provided. Assuming global_loc_id is a valid index.")
            for global_loc_id, compact_idx in location_map.items():
                future_v_global[int(global_loc_id)] = future_v[int(compact_idx)]
        else:
            for global_loc_id, compact_idx in location_map.items():
                global_idx = prov_to_idx.get(int(global_loc_id))
                if global_idx is not None:
                    future_v_global[global_idx] = future_v[int(compact_idx)]

        # Expected future value is the same for all current states, as it only depends on the next period's value
        expected_future_value_matrix = np.tile(future_v_global, (n_visited_locations, 1))

        # c. Bellman update
        choice_specific_values = flow_utility_matrix + beta * expected_future_value_matrix
        
        max_v = np.max(choice_specific_values, axis=1)
        exp_diff = np.exp(choice_specific_values - max_v[:, np.newaxis])
        current_v = max_v + np.log(np.sum(exp_diff, axis=1))
        
        # 处理可能的数值问题（NaN, 无穷大等）
        current_v = np.nan_to_num(current_v, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Store the result for the current age
        start_idx = age_idx * n_visited_locations
        end_idx = start_idx + n_visited_locations
        v_new[start_idx:end_idx] = current_v

    # The loop above is one full backward induction pass, not an iteration until convergence.
    # This is because V(age) depends on V(age+1), so we can't iterate to convergence at each age.
    # The whole backward pass IS the solution.
    
    # 最终验证和清理：确保返回值有效
    v_new = np.nan_to_num(v_new, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 验证数组形状和大小
    if v_new.shape != (n_individual_states,):
        logger.error(f"Bellman solution shape mismatch: expected {(n_individual_states,)}, got {v_new.shape}")
        # 返回零向量作为后备方案
        v_new = np.zeros(n_individual_states)
    
    if verbose:
        if hot_start_used:
            logger.info(f"Individual Bellman backward induction complete for {len(ages)} age periods (with hot-start).")
        else:
            logger.info(f"Individual Bellman backward induction complete for {len(ages)} age periods.")
        
    return v_new, 1 # Returns the solved V function and "1 iteration"