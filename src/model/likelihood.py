
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
    regions_df: pd.DataFrame = None,  # Additional parameter for regional data
) -> np.ndarray:
    """
    Calculates the conditional choice probabilities (CCPs) for each state.

    Args:
        converged_v (np.ndarray): The converged value function from the Bellman solver.
        utility_function (Callable): The function to calculate flow utility.
        state_space (pd.DataFrame): DataFrame representing the state space.
        params (Dict[str, Any]): Model parameters.
        agent_type (int): The agent's unobserved type.
        beta (float): The discount factor.
        transition_matrices (Dict[str, np.ndarray]): State transition matrices.
        regions_df (pd.DataFrame): DataFrame containing regional data for utility calculation.

    Returns:
        np.ndarray: A matrix of CCPs. Shape: (n_states, n_choices).
    """
    n_states, n_choices = len(state_space), params["n_choices"]
    choice_specific_values = np.zeros((n_states, n_choices))

    # Calculate choice-specific value function v(x, j) using the converged V*
    for j in range(n_choices):
        # Calculate flow utility u(x, j) for all states
        # We need to dynamically construct the data for utility calculation
        for i in range(n_states):
            # Get current state
            current_state = state_space.iloc[i]
            
            # Create a temporary data object that contains information for 
            # calculating utility of choosing region j from the current state
            current_data = current_state.to_dict()
            
            # Add information about the destination region j
            if regions_df is not None and j < len(regions_df):
                destination_region = regions_df.iloc[j]
                # Add the destination region properties to the data dict
                for col in regions_df.columns:
                    current_data[f"{col}_j"] = destination_region[col]
            
            # Add the preference match xi_ij (for now, using a placeholder)
            # In a more complete implementation, xi_ij would be part of the params or state space
            xi_ij = params.get(f"xi_{i}_{j}", 0.0)  # Placeholder approach
            
            # Calculate flow utility for this specific state and choice
            flow_utility = utility_function(
                data=current_data,
                params=params,
                agent_type=agent_type,
                xi_ij=xi_ij,
            )
            
            P_j = transition_matrices[j]
            expected_future_value = P_j @ converged_v
            choice_specific_values[i, j] = flow_utility + beta * expected_future_value[i]

    # Apply softmax to get probabilities (Eq. 15 in the paper)
    exp_v = np.exp(choice_specific_values)
    sum_exp_v = exp_v.sum(axis=1, keepdims=True)
    ccps = exp_v / sum_exp_v

    return ccps

def calculate_log_likelihood(
    params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    agent_type: int,
    beta: float,
    transition_matrices: Dict[str, np.ndarray],
    regions_df: pd.DataFrame = None,  # Additional parameter for regional data
) -> float:
    """
    Calculates the log-likelihood for a given set of parameters and agent type.
    This is the objective function for the M-step of the EM algorithm.

    Args:
        params (Dict[str, Any]): Model parameters to be estimated.
        observed_data (pd.DataFrame): Panel data of observed choices.
                                      Must contain 'state_index' and 'choice_index'.
        state_space (pd.DataFrame): DataFrame representing the state space.
        agent_type (int): The agent's unobserved type.
        beta (float): The discount factor.
        transition_matrices (Dict[str, np.ndarray]): State transition matrices.
        regions_df (pd.DataFrame): DataFrame containing regional data for utility calculation.

    Returns:
        float: The total log-likelihood value.
    """
    # 1. Solve the DP problem for the given parameters
    converged_v, _ = solve_bellman_equation(
        utility_function=calculate_flow_utility,
        state_space=state_space,
        params=params,
        agent_type=agent_type,
        beta=beta,
        transition_matrices=transition_matrices,
        regions_df=regions_df,  # Pass regional data
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
        regions_df,  # Pass regional data
    )

    # 3. Match observed choices to predicted probabilities
    # 'state_index' in observed_data maps each observation to a row in state_space/ccps
    # 'choice_index' is the column index of the choice that was made
    
    # Get the probabilities of the choices that were actually made
    likelihoods = ccps[observed_data['state_index'], observed_data['choice_index']]

    # 4. Calculate the log-likelihood
    # Add a small epsilon to prevent log(0)
    likelihoods = np.maximum(likelihoods, 1e-15)
    log_likelihood = np.sum(np.log(likelihoods))

    # The optimizer will minimize, so we return the negative log-likelihood
    return -log_likelihood
