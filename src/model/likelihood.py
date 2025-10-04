
"""
This module constructs the log-likelihood function for the structural model,
connecting the Bellman solver and utility functions to the observed data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

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
    Calculates the conditional choice probabilities (CCPs) for each state
    using the new dynamic data assembly logic.
    """
    n_states, n_choices = len(state_space), params["n_choices"]
    choice_specific_values = np.zeros((n_states, n_choices))

    # Prepare region data for faster lookup
    regions_df_indexed = regions_df.set_index(['provcd', 'year'])
    loc_map = {loc: i for i, loc in enumerate(sorted(regions_df['provcd'].unique()))}

    for s_idx, state in state_space.iterrows():
        current_age = state['age']
        prev_loc = state['prev_provcd']
        
        for j_idx in range(n_choices):
            dest_loc = regions_df['provcd'].unique()[j_idx]
            
            # --- Dynamic Data Assembly (similar to Bellman solver) ---
            try:
                dest_features = regions_df_indexed.loc[(dest_loc, 2012)] # Using 2012 as placeholder year
            except KeyError:
                dest_features = pd.Series(dtype=float)

            data_for_utility = pd.Series({
                'age': current_age,
                'prev_provcd': prev_loc,
                'provcd_dest': dest_loc,
                'hometown': 0, # Placeholder
                'hukou_prov': 0, # Placeholder
                'wage_predicted': 30000, # Placeholder
                'wage_ref': 30000, # Placeholder
                **dest_features.add_suffix('_dest'),
                'distance': distance_matrix[loc_map[prev_loc], loc_map[dest_loc]],
                'is_adjacent': adjacency_matrix[loc_map[prev_loc], loc_map[dest_loc]],
                'is_return_move': 0 # Placeholder
            })

            flow_utility = utility_function(
                data=data_for_utility,
                params=params,
                agent_type=agent_type,
                xi_ij=0.0, # Placeholder
            )
            
            P_j = transition_matrices[j_idx]
            expected_future_value = P_j[s_idx] @ converged_v
            
            choice_specific_values[s_idx, j_idx] = flow_utility + beta * expected_future_value

    # Apply softmax to get probabilities with improved numerical stability
    # Step 1: Clip extreme values
    choice_specific_values_clipped = np.clip(choice_specific_values, -500, 500)

    # Step 2: Apply log-sum-exp trick
    max_v = np.max(choice_specific_values_clipped, axis=1, keepdims=True)
    exp_v = np.exp(choice_specific_values_clipped - max_v)
    sum_exp_v = np.sum(exp_v, axis=1, keepdims=True)

    # Step 3: Compute probabilities with underflow protection
    sum_exp_v_safe = np.maximum(sum_exp_v, 1e-300)
    ccps = exp_v / sum_exp_v_safe

    # Step 4: Ensure probabilities are valid (non-negative, sum to 1)
    ccps = np.maximum(ccps, 1e-15)  # Floor to prevent exact zeros
    ccps = ccps / np.sum(ccps, axis=1, keepdims=True)  # Renormalize

    return ccps

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
) -> float:
    """
    Calculates the log-likelihood for a given set of parameters and agent type.
    """
    # 1. Solve the DP problem for the given parameters
    # Create a dummy individual_data_map for now - this needs to be properly constructed
    # based on the actual individual data for state-specific calculations
    individual_data_map = {}  # Placeholder - needs proper implementation
    
    converged_v, _ = solve_bellman_equation(
        utility_function=calculate_flow_utility,
        state_space=state_space,
        params=params,
        agent_type=agent_type,
        beta=beta,
        transition_matrices=transition_matrices,
        regions_df=regions_df,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        individual_data_map=individual_data_map,
    )

    # 2. Calculate conditional choice probabilities (CCPs)
    ccps = calculate_choice_probabilities(
        converged_v,
        calculate_flow_utility,
        state_space,
        params,
        agent_type,
        beta,
        transition_matrices,
        regions_df,
        distance_matrix,
        adjacency_matrix,
    )

    # 3. Match observed choices to predicted probabilities
    likelihoods = ccps[observed_data['state_index'], observed_data['choice_index']]

    # 4. Calculate the log-likelihood
    log_likelihood = np.sum(np.log(np.maximum(likelihoods, 1e-15)))

    return -log_likelihood
