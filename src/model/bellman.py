
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

    Returns:
        np.ndarray: The updated value function, v_new. Shape: (n_states,).
    """
    n_states, n_choices = len(state_space), params["n_choices"]
    choice_specific_values = np.zeros((n_states, n_choices))

    # Step 1: Calculate choice-specific value function v(x, j)
    for j in range(n_choices):
        # Calculate flow utility u(x, j) for all states
        # This requires a vectorized operation or a loop
        flow_utilities = state_space.apply(
            lambda x: utility_function(
                data=x,  # This needs to be adapted based on how utility_function expects data
                params=params,
                agent_type=agent_type,
                xi_ij=x[f"xi_{j}"], # Assuming xi values are in state space
            ),
            axis=1,
        ).values

        # Calculate expected future value E[V(x')]
        # P_j is the transition matrix for choice j: (n_states, n_states)
        P_j = transition_matrices[j]
        expected_future_value = P_j @ v_old

        choice_specific_values[:, j] = flow_utilities + beta * expected_future_value

    # Step 2: Update the expected value function V(x) using log-sum-exp
    # This is the Emax operator for Type I EV shocks
    v_new = np.log(np.sum(np.exp(choice_specific_values), axis=1))

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
) -> (np.ndarray, int):
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
        )

        # Check for convergence
        if np.max(np.abs(v_new - v_old)) < tolerance:
            print(f"Bellman equation converged for type {agent_type} in {i+1} iterations.")
            return v_new, i + 1

        v_old = v_new

    print(f"Warning: Bellman equation did not converge for type {agent_type} after {max_iterations} iterations.")
    return v_old, max_iterations
