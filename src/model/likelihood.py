
"""
This module constructs the log-likelihood function for the structural model,
connecting the Bellman solver and utility functions to the observed data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

from src.model.bellman import solve_bellman_equation
from src.model.utility import calculate_flow_utility

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
    """
    n_states, n_choices = len(state_space), params["n_choices"]
    choice_specific_values = np.zeros((n_states, n_choices))

    regions_df_indexed = regions_df.set_index(['provcd', 'year'])
    loc_map = {loc: i for i, loc in enumerate(sorted(regions_df['provcd'].unique()))}

    for s_idx, state in state_space.iterrows():
        current_age = state['age']
        prev_loc = state['prev_provcd']
        
        for j_idx in range(n_choices):
            dest_loc = regions_df['provcd'].unique()[j_idx]
            
            try:
                dest_features = regions_df_indexed.loc[(dest_loc, 2012)]
            except KeyError:
                dest_features = pd.Series(dtype=float)

            data_for_utility = pd.Series({
                'age': current_age, 'prev_provcd': prev_loc, 'provcd_dest': dest_loc,
                'hometown': 0, 'hukou_prov': 0, 'wage_predicted': 30000, 'wage_ref': 30000,
                **dest_features.add_suffix('_dest'),
                'distance': distance_matrix[loc_map[prev_loc], loc_map[dest_loc]],
                'is_adjacent': adjacency_matrix[loc_map[prev_loc], loc_map[dest_loc]],
                'is_return_move': 0
            })

            flow_utility = utility_function(
                data=data_for_utility, params=params, agent_type=agent_type, xi_ij=0.0,
            )
            
            P_j = transition_matrices[j_idx]
            expected_future_value = P_j[s_idx] @ converged_v
            choice_specific_values[s_idx, j_idx] = flow_utility + beta * expected_future_value

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
            utility_function=calculate_flow_utility,
            state_space=state_space, params=params, agent_type=agent_type, beta=beta,
            transition_matrices=transition_matrices, regions_df=regions_df,
            distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
            verbose=verbose,
        )
        if cache is not None:
            cache[cache_key] = converged_v
            logger.info(f"      [Likelihood] Bellman solution stored in cache for agent_type={agent_type}.")

    ccps = calculate_choice_probabilities(
        converged_v, calculate_flow_utility, state_space, params, agent_type, beta,
        transition_matrices, regions_df, distance_matrix, adjacency_matrix,
    )

    likelihoods = ccps[observed_data['state_index'], observed_data['choice_index']]
    log_likelihood = np.sum(np.log(likelihoods))

    return -log_likelihood
