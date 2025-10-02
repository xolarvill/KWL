
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, List

from src.model.likelihood import calculate_log_likelihood

# --- Helper functions for parameter handling ---

def _pack_params(params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Packs a dictionary of parameters into a numpy array for the optimizer."""
    param_names = sorted(params.keys())
    param_values = np.array([params[name] for name in param_names])
    return param_values, param_names

def _unpack_params(param_values: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
    """Unpacks a numpy array back into a dictionary of parameters."""
    return dict(zip(param_names, param_values))

# --- Log-Likelihood Calculation ---

def _calculate_mixture_log_likelihood(
    log_likelihood_matrix: np.ndarray, pi_k: np.ndarray
) -> float:
    """
    Calculates the total log-likelihood of the mixture model.
    
    Args:
        log_likelihood_matrix (np.ndarray): Matrix of log-likelihoods (N, K).
        pi_k (np.ndarray): Type probabilities (K,).

    Returns:
        float: The total log-likelihood of the sample.
    """
    log_pi_k = np.log(pi_k)
    weighted_log_lik = log_likelihood_matrix + log_pi_k
    
    # Use log-sum-exp for numerical stability
    max_log_lik = np.max(weighted_log_lik, axis=1)
    log_marginal_lik = max_log_lik + np.log(
        np.sum(np.exp(weighted_log_lik - max_log_lik[:, np.newaxis]), axis=1)
    )
    
    return np.sum(log_marginal_lik)

# --- EM Algorithm Steps ---

def e_step(
    params: Dict[str, Any],
    pi_k: np.ndarray,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame = None,  # Additional parameter for regional data
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the E-step: calculates posterior probabilities and the log-likelihood matrix.
    """
    N = observed_data["individual_id"].nunique()
    K = len(pi_k)
    log_likelihood_matrix = np.zeros((N, K))

    # Group data by individual once
    grouped_data = observed_data.groupby("individual_id")
    
    for k in range(K):
        # Calculate log-likelihood for all observations for a given type k
        # Note: calculate_log_likelihood returns the *negative* log-likelihood
        log_lik_obs = -calculate_log_likelihood(
            params=params,
            observed_data=observed_data,
            state_space=state_space,
            agent_type=k,
            beta=beta,
            transition_matrices=transition_matrices,
            regions_df=regions_df,  # Pass regional data
        )
        
        # Sum log-likelihoods for each individual
        observed_data_copy = observed_data.copy()
        observed_data_copy['log_lik_obs'] = log_lik_obs
        individual_log_lik = observed_data_copy.groupby("individual_id")['log_lik_obs'].sum()
        log_likelihood_matrix[:, k] = individual_log_lik.values

    # Calculate posterior probabilities using Bayes' rule in log space
    log_pi_k = np.log(pi_k)
    weighted_log_lik = log_likelihood_matrix + log_pi_k
    
    log_marginal_lik = np.log(np.sum(np.exp(weighted_log_lik), axis=1, keepdims=True))
    log_posterior = weighted_log_lik - log_marginal_lik
    
    posterior_probs = np.exp(log_posterior)
    return posterior_probs, log_likelihood_matrix

def m_step(
    posterior_probs: np.ndarray,
    initial_params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame = None,  # Additional parameter for regional data
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Performs the M-step: updates parameters and type probabilities.
    """
    K = posterior_probs.shape[1]
    
    # For simplicity, we assume all types share the same structural parameters theta
    # A more complex model could have type-specific parameters
    
    # Create the objective function for the optimizer
    def objective_function(param_values: np.ndarray, param_names: List[str]) -> float:
        params_k = _unpack_params(param_values, param_names)
        total_weighted_log_lik = 0
        
        for k in range(K):
            weights = posterior_probs[:, k]
            
            # Calculate log-likelihood for this type
            log_lik_k_obs = -calculate_log_likelihood(
                params=params_k,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=k,
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,  # Pass regional data
            )
            
            # Weight it by the posterior probabilities
            observed_data_copy = observed_data.copy()
            observed_data_copy['log_lik_k_obs'] = log_lik_k_obs
            individual_log_lik = observed_data_copy.groupby("individual_id")['log_lik_k_obs'].sum()
            total_weighted_log_lik += np.sum(weights * individual_log_lik)
            
        return -total_weighted_log_lik

    # Initial guess for parameters
    initial_param_values, param_names = _pack_params(initial_params)
    
    # Run optimizer
    result = minimize(
        objective_function,
        initial_param_values,
        args=(param_names,),
        method='BFGS',
        options={'disp': True, 'maxiter': 100} # Limit iterations for M-step
    )
    
    updated_params = _unpack_params(result.x, param_names)

    # Update type probabilities
    updated_pi_k = posterior_probs.mean(axis=0)

    return updated_params, updated_pi_k

# --- Main EM Algorithm Runner ---

def run_em_algorithm(
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    n_types: int,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    n_choices: int = 31,  # Default value, but should be passed from caller
    regions_df: pd.DataFrame = None,  # Regional data needed for utility calculation
) -> Dict[str, Any]:
    """
    The main function to run the EM algorithm.
    """
    # 1. Initialize parameters and type probabilities
    # TODO: A more robust initialization is needed
    initial_params = {
        "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0,
        "rho_base_tier_1": 1.0, "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1,
        "gamma_0_type_0": 1.0, "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2,
        "gamma_3": -0.4, "gamma_4": 0.01, "gamma_5": -0.05,
        "alpha_climate": 0.1, "alpha_health": 0.1, "alpha_education": 0.1, "alpha_public_services": 0.1,
        "n_choices": n_choices  # Add n_choices to params
    }
    pi_k = np.full(n_types, 1 / n_types)
    
    old_log_likelihood = -np.inf

    for i in range(max_iterations):
        print(f"--- EM Iteration {i+1}/{max_iterations} ---")
        
        # 2. E-Step
        print("Running E-step...")
        posterior_probs, log_likelihood_matrix = e_step(
            initial_params, pi_k, observed_data, state_space, transition_matrices, beta, regions_df
        )
        
        # 3. M-Step
        print("Running M-step...")
        initial_params, pi_k = m_step(
            posterior_probs, initial_params, observed_data, state_space, transition_matrices, beta, regions_df
        )
        
        # 4. Check for convergence
        new_log_likelihood = _calculate_mixture_log_likelihood(log_likelihood_matrix, pi_k)
        
        if abs(new_log_likelihood - old_log_likelihood) < tolerance:
            print(f"EM algorithm converged after {i+1} iterations.")
            break
        
        old_log_likelihood = new_log_likelihood
        print(f"Log-Likelihood: {new_log_likelihood:.4f}")

    # 返回完整的估计结果
    return {
        "structural_params": initial_params, 
        "type_probabilities": pi_k,
        "final_log_likelihood": new_log_likelihood,
        "n_iterations": i+1,
        "posterior_probs": posterior_probs,
        "log_likelihood_matrix": log_likelihood_matrix
    }
