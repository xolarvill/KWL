
"""
This module implements the Bellman equation solver using value function iteration
for the dynamic discrete choice model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

def solve_bellman_iteration(
    v_old: np.ndarray,
    utility_function: Callable,
    state_space: pd.DataFrame,
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame = None,  # Additional parameter for region data
) -> np.ndarray:
    """
    Performs a single iteration of the Bellman equation.

    Args:
        v_old (np.ndarray): The value function from the previous iteration.
                           Shape: (n_states,).
        utility_function (Callable): The function to calculate flow utility.
        state_space (pd.DataFrame): DataFrame representing the state space.
        params (Dict[str, Any]): Model parameters.
        agent_type (int): The agent's unobserved type.
        beta (float): The discount factor.
        transition_matrices (Dict[str, np.ndarray]): A dictionary mapping choices
                                                     to their state transition matrices.
        regions_df (pd.DataFrame): DataFrame containing regional data for utility calculation.

    Returns:
        np.ndarray: The updated value function, v_new. Shape: (n_states,).
    """
    n_states, n_choices = len(state_space), params["n_choices"]
    choice_specific_values = np.zeros((n_states, n_choices))

    # 获取地区ID列表用于映射
    if regions_df is not None and 'provcd' in regions_df.columns:
        unique_regions = sorted(regions_df['provcd'].unique())
    else:
        unique_regions = list(range(n_choices))  # 如果没有地区数据，则使用索引

    # Step 1: Calculate choice-specific value function v(x, j)
    for j in range(n_choices):
        if j < len(unique_regions):
            dest_region_id = unique_regions[j]
        else:
            dest_region_id = j  # fallback

        for i in range(n_states):
            # 获取当前状态
            current_state = state_space.iloc[i]
            current_data = current_state.to_dict()
            
            # 获取目的地区域数据
            if regions_df is not None:
                # 为目的地添加区域特征
                dest_region_data = regions_df[regions_df['provcd'] == dest_region_id]
                if len(dest_region_data) > 0:
                    # 取第一个匹配项（如果有多行，通常按年份匹配）
                    dest_row = dest_region_data.iloc[0]  # 或者应该根据年份匹配
                    for col in dest_region_data.columns:
                        if col != 'provcd':  # 避免覆盖provcd列
                            current_data[f"{col}_dest"] = dest_row[col]

            # 添加个体特定参数
            current_data['agent_type'] = agent_type
            
            # Add the preference match xi_ij (for now, using a placeholder)
            xi_ij = 0.0  # Placeholder approach
            
            try:
                # Calculate flow utility for this specific state and choice
                flow_utility = utility_function(
                    data=current_data,
                    params=params,
                    agent_type=agent_type,
                    xi_ij=xi_ij,
                )
                
                # Calculate expected future value E[V(x')]
                # P_j is the transition matrix for choice j: (n_states, n_states)
                j_key = int(j)  # 确保j是整数键
                i_key = int(i)  # 确保i是整数索引
                P_j = transition_matrices[j_key]
                expected_future_value = P_j @ v_old

                choice_specific_values[i_key, j_key] = flow_utility + beta * expected_future_value[i_key]
            except KeyError as e:
                print(f"KeyError in utility calculation: {e}")
                print(f"Available keys in current_data: {list(current_data.keys())}")
                print(f"Error occurred at state {i}, choice {j}")
                # 使用默认效用值
                choice_specific_values[i, j] = 0.0

    # Step 2: Update the expected value function V(x) using log-sum-exp
    # This is the Emax operator for Type I EV shocks
    # Use numerically stable log-sum-exp
    max_v = np.max(choice_specific_values, axis=1, keepdims=True)
    v_new = max_v.squeeze() + np.log(np.sum(np.exp(choice_specific_values - max_v), axis=1))
    
    # Check for invalid values
    if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
        print(f"Warning: Invalid values in value function update")
        v_new = np.nan_to_num(v_new, nan=0.0, posinf=1e10, neginf=-1e10)

    return v_new

def solve_bellman_equation(
    utility_function: Callable,
    state_space: pd.DataFrame,
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    regions_df: pd.DataFrame = None,  # Additional parameter for regional data
) -> 'tuple[np.ndarray, int]':
    """
    Solves the Bellman equation using value function iteration until convergence.

    Args:
        utility_function (Callable): The function to calculate flow utility.
        state_space (pd.DataFrame): DataFrame representing the state space.
        params (Dict[str, Any]): Model parameters.
        agent_type (int): The agent's unobserved type.
        beta (float): The discount factor.
        transition_matrices (Dict[str, np.ndarray]): State transition matrices.
        tolerance (float): Convergence criterion.
        max_iterations (int): Maximum number of iterations.
        regions_df (pd.DataFrame): DataFrame containing regional data for utility calculation.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - The converged value function (np.ndarray).
            - The number of iterations performed (int).
    """
    n_states = len(state_space)
    v_old = np.zeros(n_states)
    
    for i in range(max_iterations):
        v_new = solve_bellman_iteration(
            v_old,
            utility_function,
            state_space,
            params,
            agent_type,
            beta,
            transition_matrices,
            regions_df,
        )

        # Check for convergence
        if np.max(np.abs(v_new - v_old)) < tolerance:
            print(f"Bellman equation converged for type {agent_type} in {i+1} iterations.")
            return v_new, i + 1

        v_old = v_new

    print(f"Warning: Bellman equation did not converge for type {agent_type} after {max_iterations} iterations.")
    return v_old, max_iterations
