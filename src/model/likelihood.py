"""
This module constructs the log-likelihood function for the structural model.
This version is refactored for performance, decoupling Bellman solution from likelihood calculation.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

from src.model.bellman import solve_bellman_equation
from src.model.utility import calculate_flow_utility_vectorized

def solve_bellman_for_params(
    params: Dict[str, Any],
    state_space: pd.DataFrame,
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    initial_v: np.ndarray = None, # <-- Key addition for hot-starts
    verbose: bool = True,
) -> np.ndarray:
    """
    Solves the Bellman equation for a given set of parameters.
    Accepts an initial value function for hot-starting the solver.
    """
    converged_v, _ = solve_bellman_equation(
        utility_function=calculate_flow_utility_vectorized,
        state_space=state_space,
        params=params,
        agent_type=agent_type,
        beta=beta,
        transition_matrices=transition_matrices,
        regions_df=regions_df,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        initial_v=initial_v, # Pass the hot-start vector to the solver
        verbose=verbose,
    )
    return converged_v

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
) -> np.ndarray:
    """
    Calculates the log-likelihood for each observation given a converged value function.
    
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

    max_v = np.max(choice_specific_values, axis=1, keepdims=True)
    exp_v = np.exp(choice_specific_values - max_v)
    sum_exp_v = np.sum(exp_v, axis=1, keepdims=True)
    ccps = exp_v / np.maximum(sum_exp_v, 1e-300)
    ccps = np.maximum(ccps, 1e-15)

    state_indices = observed_data['state_index'].values
    choice_indices = observed_data['choice_index'].values
    
    if np.any(state_indices >= ccps.shape[0]) or np.any(choice_indices >= ccps.shape[1]):
        logger.error("Error: Invalid state or choice index detected.")
        return np.full(len(observed_data), -1e10)

    likelihoods = ccps[state_indices, choice_indices]
    return np.log(likelihoods)