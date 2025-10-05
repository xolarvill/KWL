"""
This module constructs the log-likelihood function for the structural model,
connecting the Bellman solver and utility functions to the observed data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

from src.model.bellman import solve_bellman_equation
from src.model.utility import calculate_flow_utility_vectorized # 导入新的向量化函数

def calculate_choice_probabilities(
    converged_v: np.ndarray,
    utility_function: Callable,
    state_space: pd.DataFrame,
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculates the conditional choice probabilities (CCPs) for each state.
    NOTE: This function will also be vectorized or updated to work with vectorized inputs.
    For now, we adapt it to work with the output of the vectorized Bellman solver.
    """
    # This function needs to be fully vectorized as well for max performance.
    # For now, let's ensure it's compatible.
    # The logic here is largely superseded by the vectorized Bellman calculation,
    # as choice probabilities can be derived directly from choice_specific_values.
    # We will refactor this to be more efficient.
    
    n_states, n_choices = len(state_space), params["n_choices"]
    
    # Re-calculating choice-specific values. This is inefficient and should be optimized.
    # A better approach would be to have solve_bellman return choice_specific_values.
    # For now, we replicate the calculation.
    
    state_space_np = {col: state_space[col].to_numpy() for col in state_space.columns}
    numeric_cols = regions_df.select_dtypes(include=np.number).columns.tolist()
    regions_df_indexed = regions_df.set_index('provcd')
    unique_provcds = regions_df['provcd'].unique()
    regions_df_np = {
        col: regions_df_indexed.loc[unique_provcds][col].to_numpy()
        for col in numeric_cols if col != 'provcd'
    }
    regions_df_np['provcd'] = unique_provcds

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
        expected_future_value_matrix[:, j_idx] = P_j @ converged_v
        
    choice_specific_values = flow_utility_matrix + beta * expected_future_value_matrix

    max_v = np.max(choice_specific_values, axis=1, keepdims=True)
    exp_v = np.exp(choice_specific_values - max_v)
    sum_exp_v = np.sum(exp_v, axis=1, keepdims=True)
    ccps = exp_v / np.maximum(sum_exp_v, 1e-300)
    
    return np.maximum(ccps, 1e-15)


def calculate_log_likelihood(
    params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    verbose: bool = True,
    cache: Dict[Tuple, np.ndarray] = None,
    cache_key: Tuple = None,
) -> float:
    """
    Calculates the log-likelihood, utilizing a cache for Bellman equation solutions.
    """
    logger = logging.getLogger()
    
    if cache is not None and cache_key in cache:
        converged_v = cache[cache_key]
        logger.info(f"      [Likelihood] Cache HIT for agent_type={agent_type}. Reusing Bellman solution.")
    else:
        logger.info(f"      [Likelihood] Cache MISS for agent_type={agent_type}. Solving Bellman equation...")
        converged_v, _ = solve_bellman_equation(
            utility_function=calculate_flow_utility_vectorized, # FIX: Point to the correct vectorized function
            state_space=state_space, params=params, agent_type=agent_type, beta=beta,
            transition_matrices=transition_matrices, regions_df=regions_df,
            distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
            verbose=verbose,
        )
        if cache is not None:
            cache[cache_key] = converged_v
            logger.info(f"      [Likelihood] Bellman solution stored in cache for agent_type={agent_type}.")

    ccps = calculate_choice_probabilities(
        converged_v, calculate_flow_utility_vectorized, state_space, params, agent_type, beta, # FIX: Point to the correct vectorized function
        transition_matrices, regions_df, distance_matrix, adjacency_matrix,
    )

    # Ensure indices are valid
    state_indices = observed_data['state_index'].values
    choice_indices = observed_data['choice_index'].values
    
    if np.any(state_indices >= ccps.shape[0]) or np.any(choice_indices >= ccps.shape[1]):
        logger.error("Error: Invalid state or choice index detected.")
        return -1e10 # Return a large negative number

    likelihoods = ccps[state_indices, choice_indices]
    log_likelihood = np.sum(np.log(likelihoods))

    return -log_likelihood