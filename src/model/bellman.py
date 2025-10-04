
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
    individual_data_map: Dict[int, pd.Series] = None  # Make it optional
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
        
        for j_idx in range(n_choices):
            dest_loc = regions_df['provcd'].unique()[j_idx] if len(regions_df['provcd'].unique()) > j_idx else list(loc_map.keys())[j_idx % len(loc_map)]
            
            # --- Dynamic Data Assembly ---
            # 1. Get destination features for the current year (simplification: use a fixed year)
            # A proper implementation needs to handle time-varying region features
            try:
                dest_features = regions_df_indexed.loc[(dest_loc, 2012)] # Using 2012 as placeholder year
            except KeyError:
                # If 2012 is not available, try another year or use first available
                available_years = regions_df_indexed.index.get_level_values('year').unique()
                if len(available_years) > 0:
                    try:
                        dest_features = regions_df_indexed.loc[(dest_loc, available_years[0])]
                    except:
                        dest_features = pd.Series(dtype=float)  # Empty series if still not found
                else:
                    dest_features = pd.Series(dtype=float)

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
                # Add distance and adjacency (with bounds checking)
                'distance': distance_matrix[loc_map[prev_loc], loc_map[dest_loc]] if 
                            prev_loc in loc_map and dest_loc in loc_map and 
                            loc_map[prev_loc] < distance_matrix.shape[0] and 
                            loc_map[dest_loc] < distance_matrix.shape[1] else 0.0,
                'is_adjacent': adjacency_matrix[loc_map[prev_loc], loc_map[dest_loc]] if 
                               prev_loc in loc_map and dest_loc in loc_map and 
                               loc_map[prev_loc] < adjacency_matrix.shape[0] and 
                               loc_map[dest_loc] < adjacency_matrix.shape[1] else 0.0,
                'is_return_move': 0 # Placeholder
            })

            # Calculate flow utility u(s, j)
            # This is still missing the individual-specific xi_ij
            try:
                flow_utility = utility_function(
                    data=data_for_utility,
                    params=params,
                    agent_type=agent_type,
                    xi_ij=0.0, # Placeholder for non-economic preference
                )
            except KeyError as e:
                print(f"KeyError in utility function: {e}")
                print(f"Available keys in data_for_utility: {list(data_for_utility.keys())}")
                flow_utility = 0.0  # Use 0 as fallback utility

            # Calculate expected future value E[V(x')]
            P_j = transition_matrices[j_idx]
            
            # Check bounds before matrix operation
            if s_idx < P_j.shape[0] and v_old.shape[0] > 0:
                expected_future_value = P_j[s_idx] @ v_old
                choice_specific_values[s_idx, j_idx] = flow_utility + beta * expected_future_value
            else:
                choice_specific_values[s_idx, j_idx] = flow_utility  # Just current utility if future is invalid

    # Update the expected value function V(x) using log-sum-exp with improved numerical stability
    # Step 1: Clip choice_specific_values to prevent extreme values
    choice_specific_values_clipped = np.clip(choice_specific_values, -500, 500)

    # Step 2: Apply log-sum-exp trick
    max_v = np.max(choice_specific_values_clipped, axis=1)

    # Step 3: Compute exp differences with underflow protection
    exp_diff = np.exp(choice_specific_values_clipped - max_v[:, np.newaxis])
    sum_exp = np.sum(exp_diff, axis=1)

    # Step 4: Handle edge cases where sum_exp is too small
    sum_exp_safe = np.maximum(sum_exp, 1e-300)

    # Step 5: Compute log-sum-exp
    log_sum_exp = np.log(sum_exp_safe)
    v_new = max_v + log_sum_exp

    # Step 6: Final validation and clipping
    v_new = np.clip(v_new, -1e6, 1e6)

    # Handle potential NaN/Inf values as last resort
    if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
        n_invalid = np.sum(np.isnan(v_new) | np.isinf(v_new))
        print(f"Warning: Invalid values detected in {n_invalid}/{len(v_new)} states")
        print(f"  Choice values range: [{np.min(choice_specific_values):.2f}, {np.max(choice_specific_values):.2f}]")
        print(f"  Max exp difference sum: {np.max(sum_exp):.2e}, Min: {np.min(sum_exp):.2e}")
        # Provide a reasonable fallback
        v_new = np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=-1e6)

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
    individual_data_map: Dict[int, pd.Series] = None,  # Make it optional with default
    tolerance: float = 1e-6,
    max_iterations: int = 100,  # Reduced default iterations to avoid hanging
    verbose: bool = True,  # Control printing
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
        regions_df (pd.DataFrame): DataFrame containing regional data for utility calculation.
        distance_matrix (np.ndarray): Matrix of distances between regions.
        adjacency_matrix (np.ndarray): Matrix indicating adjacent regions.
        individual_data_map (Dict[int, pd.Series]): Mapping of individual IDs to their data (optional).
        tolerance (float): Convergence criterion.
        max_iterations (int): Maximum number of iterations.
        verbose (bool): Whether to print convergence messages.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - The converged value function (np.ndarray).
            - The number of iterations performed (int).
    """
    n_states = len(state_space)
    if n_states == 0:
        if verbose:
            print("Warning: Empty state space, returning zero vector")
        return np.zeros(0), 0

    v_old = np.zeros(n_states)

    for i in range(max_iterations):
        try:
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
        except Exception as e:
            if verbose:
                print(f"Error in Bellman iteration {i+1}: {e}")
            # Return current value function if iteration fails
            return v_old, i

        # Check for convergence
        diff = np.max(np.abs(v_new - v_old))

        # Check for invalid values
        if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
            if verbose:
                print(f"Warning: Invalid values in iteration {i+1}, stopping.")
            return v_old, i

        if diff < tolerance:
            if verbose:
                print(f"Bellman equation converged for type {agent_type} in {i+1} iterations.")
            return v_new, i + 1

        v_old = v_new

    if verbose:
        print(f"Warning: Bellman equation did not converge for type {agent_type} after {max_iterations} iterations.")
    return v_old, max_iterations
