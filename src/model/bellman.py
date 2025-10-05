
"""
This module implements the Bellman equation solver using value function iteration
for the dynamic discrete choice model. This version is optimized with vectorization.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

from src.model.utility import calculate_flow_utility_vectorized

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
    
    # --- 1. Calculate Flow Utility for all (state, choice) pairs ---
    # This function is now vectorized and returns a (n_states, n_choices) matrix
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

    # --- 2. Calculate Expected Future Value ---
    # E[V(x')] = sum over j of P(x'|x,j) * V(x')
    # We can compute this for all states and choices simultaneously.
    expected_future_value_matrix = np.zeros((n_states, n_choices))
    for j_idx in range(n_choices):
        P_j = transition_matrices[j_idx]  # Transition matrix for choice j
        # P_j is (n_states, n_states), v_old is (n_states,)
        # The result is the expected future value for each state, given choice j
        expected_future_value_matrix[:, j_idx] = P_j @ v_old

    # --- 3. Combine Flow Utility and Future Value ---
    choice_specific_values = flow_utility_matrix + beta * expected_future_value_matrix

    # --- 4. Update Value Function using Log-Sum-Exp ---
    max_v = np.max(choice_specific_values, axis=1)
    exp_diff = np.exp(choice_specific_values - max_v[:, np.newaxis])
    v_new = max_v + np.log(np.sum(exp_diff, axis=1))
    
    # Handle potential NaN/Inf values
    v_new = np.nan_to_num(v_new, nan=0.0, posinf=1e6, neginf=-1e6)

    return v_new

def solve_bellman_equation(
    utility_function: Callable,
    state_space: pd.DataFrame,
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    tolerance: float = 1e-4,
    max_iterations: int = 50,
    verbose: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Solves the Bellman equation using vectorized value function iteration.
    """
    logger = logging.getLogger()
    n_states = len(state_space)
    if n_states == 0:
        if verbose: logger.warning("Warning: Empty state space, returning zero vector")
        return np.zeros(0), 0

    # Convert pandas DataFrames to NumPy arrays/dicts for performance
    state_space_np = {col: state_space[col].to_numpy() for col in state_space.columns}
    
    # Pre-process region data for vectorization
    regions_df_indexed = regions_df.set_index('provcd')
    unique_provcds = regions_df['provcd'].unique()
    
    # Ensure all columns are numeric before converting
    numeric_cols = regions_df.select_dtypes(include=np.number).columns.tolist()
    regions_df_np = {
        col: regions_df_indexed.loc[unique_provcds][col].to_numpy()
        for col in numeric_cols if col != 'provcd'
    }
    regions_df_np['provcd'] = unique_provcds

    v_old = np.zeros(n_states)
    for i in range(max_iterations):
        v_new = solve_bellman_iteration_vectorized(
            v_old, utility_function, state_space_np, params, agent_type, beta,
            transition_matrices, regions_df_np, distance_matrix, adjacency_matrix
        )
        
        diff = np.max(np.abs(v_new - v_old))
        if diff < tolerance:
            if verbose: logger.info(f"Bellman equation converged for type {agent_type} in {i+1} iterations.")
            return v_new, i + 1
        
        v_old = v_new

    if verbose:
        logger.warning(f"Warning: Bellman equation did not converge for type {agent_type} after {max_iterations} iterations.")
    return v_old, max_iterations
