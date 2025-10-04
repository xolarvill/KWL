
"""
This module implements the Bellman equation solver using value function iteration
for the dynamic discrete choice model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple

def solve_bellman_iteration(
    v_old: np.ndarray,
    utility_function: Callable,
    state_space: pd.DataFrame,
    params: Dict[str, Any],
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    individual_data_map: Dict[int, pd.Series] 
) -> np.ndarray:
    """
    Performs a single iteration of the Bellman equation with dynamic data assembly.
    """
    n_states, n_choices = len(state_space), params["n_choices"]
    choice_specific_values = np.zeros((n_states, n_choices))

    # Prepare region data for faster lookup
    regions_df_indexed = regions_df.set_index(['provcd', 'year'])
    
    # Create a map from location code to matrix index
    loc_map = {loc: i for i, loc in enumerate(sorted(regions_df['provcd'].unique()))}

    for s_idx, state in state_space.iterrows():
        current_age = state['age']
        prev_loc = state['prev_provcd']
        
        # This is a simplification: assumes we can map state to a representative individual
        # A more robust implementation would average over individuals in that state
        # For now, we take the first individual who was ever in this state
        # NOTE: This part of the logic is complex. We are missing the individual-specific
        # state variables like hukou_prov, hometown, xi_ij, etc.
        # This requires a significant refactor of the state space definition.
        # As a placeholder, we create a dummy `individual_series`.
        
        # Let's assume the state space also contains individual-specific identifiers
        # For now, we will have to pass dummy values for individual vars.
        
        for j_idx in range(n_choices):
            dest_loc = regions_df['provcd'].unique()[j_idx]
            
            # --- Dynamic Data Assembly ---
            # 1. Get destination features for the current year (simplification: use a fixed year)
            # A proper implementation needs to handle time-varying region features
            try:
                dest_features = regions_df_indexed.loc[(dest_loc, 2012)] # Using 2012 as placeholder year
            except KeyError:
                dest_features = pd.Series(dtype=float) # Empty series if not found

            # 2. Combine into a single series for the utility function
            data_for_utility = pd.Series({
                'age': current_age,
                'prev_provcd': prev_loc,
                'provcd_dest': dest_loc,
                'hometown': 0, # Placeholder
                'hukou_prov': 0, # Placeholder
                'wage_predicted': 30000, # Placeholder
                'wage_ref': 30000, # Placeholder
                # Add destination features
                **dest_features.add_suffix('_dest'),
                # Add distance and adjacency
                'distance': distance_matrix[loc_map[prev_loc], loc_map[dest_loc]],
                'is_adjacent': adjacency_matrix[loc_map[prev_loc], loc_map[dest_loc]],
                'is_return_move': 0 # Placeholder
            })

            # Calculate flow utility u(s, j)
            # This is still missing the individual-specific xi_ij
            flow_utility = utility_function(
                data=data_for_utility,
                params=params,
                agent_type=agent_type,
                xi_ij=0.0, # Placeholder for non-economic preference
            )
            
            # Calculate expected future value E[V(x')]
            P_j = transition_matrices[j_idx]
            expected_future_value = P_j[s_idx] @ v_old
            
            choice_specific_values[s_idx, j_idx] = flow_utility + beta * expected_future_value

    # Update the expected value function V(x) using log-sum-exp
    max_v = np.max(choice_specific_values, axis=1)
    v_new = max_v + np.log(np.sum(np.exp(choice_specific_values - max_v[:, np.newaxis]), axis=1))
    
    if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
        print("Warning: Invalid values in value function update")
        v_new = np.nan_to_num(v_new, nan=-1e10, posinf=-1e10, neginf=-1e10)

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
    individual_data_map: Dict[int, pd.Series],
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, int]:
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
            distance_matrix,
            adjacency_matrix,
            individual_data_map
        )

        # Check for convergence
        if np.max(np.abs(v_new - v_old)) < tolerance:
            print(f"Bellman equation converged for type {agent_type} in {i+1} iterations.")
            return v_new, i + 1

        v_old = v_new

    print(f"Warning: Bellman equation did not converge for type {agent_type} after {max_iterations} iterations.")
    return v_old, max_iterations
