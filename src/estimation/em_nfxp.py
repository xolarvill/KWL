import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, List
from joblib import Parallel, delayed
import logging
import time
import os
from functools import wraps

from src.model.likelihood import solve_bellman_for_params, calculate_likelihood_from_v
from src.estimation.migration_behavior_analysis import identify_migration_behavior_types, create_behavior_based_initial_params

# --- Logging and Timing Setup ---

def setup_logging():
    log_dir = 'progress/log'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"em_estimation_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        logger.info(f"--- Starting execution of {func.__name__} ---")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- Finished execution of {func.__name__} in {duration:.2f} seconds ---")
        return result
    return wrapper

# --- Helper functions for parameter handling ---

def _pack_params(params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    param_names = sorted([k for k in params.keys() if k != 'n_choices'])
    param_values = np.array([params[name] for name in param_names])
    return param_values, param_names

def _unpack_params(param_values: np.ndarray, param_names: List[str], n_choices: int) -> Dict[str, Any]:
    params_dict = dict(zip(param_names, param_values))
    params_dict['n_choices'] = n_choices
    return params_dict

# --- Log-Likelihood Calculation ---

def _calculate_mixture_log_likelihood(log_likelihood_matrix: np.ndarray, pi_k: np.ndarray) -> float:
    logger = logging.getLogger()
    try:
        pi_k_safe = np.maximum(pi_k, 1e-10)
        pi_k_safe = pi_k_safe / np.sum(pi_k_safe)
        log_pi_k = np.log(pi_k_safe)
        weighted_log_lik = log_likelihood_matrix + log_pi_k
        max_log_lik = np.max(weighted_log_lik, axis=1, keepdims=True)
        log_marginal_lik = max_log_lik.squeeze() + np.log(np.sum(np.exp(weighted_log_lik - max_log_lik), axis=1))
        if np.any(np.isnan(log_marginal_lik)) or np.any(np.isinf(log_marginal_lik)):
            logger.warning("Invalid values in log_marginal_lik")
            return -1e10
        return np.sum(log_marginal_lik)
    except Exception as e:
        logger.error(f"Error in mixture log-likelihood calculation: {e}")
        return -1e10

# --- EM Algorithm Steps ---

def e_step(
    params: Dict[str, Any], pi_k: np.ndarray, observed_data: pd.DataFrame, state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray], beta: float, regions_df: pd.DataFrame,
    distance_matrix: np.ndarray, adjacency_matrix: np.ndarray, n_types: int = None,
    force_type_separation: bool = True, n_jobs: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    logger = logging.getLogger()
    unique_individuals, individual_indices = np.unique(observed_data['individual_id'], return_inverse=True)
    N = len(unique_individuals)
    K = len(pi_k) if n_types is None else n_types
    log_likelihood_matrix = np.zeros((N, K))
    logger.info(f"  Computing log-likelihoods for {N} individuals across {K} types (parallel={n_jobs>1})...")

    def compute_type_likelihood(k):
        try:
            type_specific_params = params.copy()
            if f'gamma_0_type_{k}' in params: type_specific_params['gamma_0'] = params[f'gamma_0_type_{k}']
            if f'gamma_1_type_{k}' in params: type_specific_params['gamma_1'] = params[f'gamma_1_type_{k}']
            if f'alpha_home_type_{k}' in params: type_specific_params['alpha_home'] = params[f'alpha_home_type_{k}']
            if f'lambda_type_{k}' in params: type_specific_params['lambda'] = params[f'lambda_type_{k}']

            # Solve Bellman equation once for this type
            converged_v = solve_bellman_for_params(
                params=type_specific_params, state_space=state_space, agent_type=int(k),
                beta=beta, transition_matrices=transition_matrices, regions_df=regions_df,
                distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
                initial_v=None, verbose=False
            )

            # Calculate likelihood using the converged value function
            log_lik_obs_vector = calculate_likelihood_from_v(
                converged_v=converged_v, params=type_specific_params, observed_data=observed_data,
                state_space=state_space, agent_type=int(k), beta=beta,
                transition_matrices=transition_matrices, regions_df=regions_df,
                distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix
            )
            log_lik_obs_vector = np.nan_to_num(log_lik_obs_vector, nan=-1e10, posinf=-1e10, neginf=-1e10)

            # Use fast numpy aggregation instead of slow pandas groupby
            individual_log_lik = np.bincount(individual_indices, weights=log_lik_obs_vector)
            return k, individual_log_lik
        except Exception as e:
            logger.error(f"  Error computing likelihood for type {k}: {e}", exc_info=True)
            return k, np.full(N, -1e10)

    if n_jobs > 1:
        results = Parallel(n_jobs=min(n_jobs, K))(delayed(compute_type_likelihood)(k) for k in range(K))
        for k, log_lik_values in results: log_likelihood_matrix[:, k] = log_lik_values
    else:
        for k in range(K):
            k_idx, log_lik_values = compute_type_likelihood(k)
            log_likelihood_matrix[:, k_idx] = log_lik_values

    try:
        pi_k_safe = np.maximum(pi_k, 1e-10)
        pi_k_safe = pi_k_safe / np.sum(pi_k_safe)
        log_pi_k = np.log(pi_k_safe)
        weighted_log_lik = log_likelihood_matrix + log_pi_k
        max_log_lik = np.max(weighted_log_lik, axis=1, keepdims=True)
        log_marginal_lik = max_log_lik + np.log(np.sum(np.exp(weighted_log_lik - max_log_lik), axis=1, keepdims=True))
        log_posterior = weighted_log_lik - log_marginal_lik
        posterior_probs = np.exp(log_posterior)
        posterior_probs = np.maximum(posterior_probs, 0)
        row_sums = posterior_probs.sum(axis=1, keepdims=True)
        posterior_probs = posterior_probs / np.maximum(row_sums, 1e-10)
        if force_type_separation:
            MIN_TYPE_PROB = 0.05
            posterior_probs = np.maximum(posterior_probs, MIN_TYPE_PROB)
            row_sums = posterior_probs.sum(axis=1, keepdims=True)
            posterior_probs = posterior_probs / np.maximum(row_sums, 1e-10)
    except Exception as e:
        logger.error(f"Error in E-step posterior probability calculation: {e}")
        posterior_probs = np.zeros((N, K))
    
    logger.info(f"  E-step completed. Posterior probability range: [{posterior_probs.min():.6f}, {posterior_probs.max():.6f}]")
    return posterior_probs, log_likelihood_matrix

def m_step(
    posterior_probs: np.ndarray, initial_params: Dict[str, Any], observed_data: pd.DataFrame,
    state_space: pd.DataFrame, transition_matrices: Dict[str, np.ndarray], beta: float,
    regions_df: pd.DataFrame, distance_matrix: np.ndarray, adjacency_matrix: np.ndarray,
    n_types: int = None, hot_start_state: Dict[int, np.ndarray] = None
) -> Tuple[Dict[str, Any], np.ndarray, Dict[int, np.ndarray]]:
    logger = logging.getLogger()
    K = posterior_probs.shape[1] if n_types is None else n_types

    # Initialize hot-start state if not provided
    if hot_start_state is None:
        hot_start_state = {}

    logger.info(f"\n  Type weight diagnostics:")
    for k in range(K):
        weight_sum = np.sum(posterior_probs[:, k])
        weight_mean = np.mean(posterior_probs[:, k])
        logger.info(f"    Type {k}: sum={weight_sum:10.4f}, mean={weight_mean:.6f}")

    # --- PERFORMANCE FIX: Pre-calculate observation-to-individual mapping ---
    unique_individuals, individual_indices = np.unique(observed_data['individual_id'], return_inverse=True)

    def objective_function(param_values: np.ndarray, param_names: List[str]) -> float:
        if not hasattr(objective_function, 'call_count'):
            objective_function.call_count = 0
        objective_function.call_count += 1

        params_k = _unpack_params(param_values, param_names, initial_params['n_choices'])
        total_weighted_log_lik = 0
        try:
            for k in range(K):
                weights = posterior_probs[:, k]
                if np.sum(weights) < 1e-10: continue

                type_specific_params = params_k.copy()
                if f'gamma_0_type_{k}' in params_k: type_specific_params['gamma_0'] = params_k[f'gamma_0_type_{k}']
                if f'gamma_1_type_{k}' in params_k: type_specific_params['gamma_1'] = params_k[f'gamma_1_type_{k}']
                if f'alpha_home_type_{k}' in params_k: type_specific_params['alpha_home'] = params_k[f'alpha_home_type_{k}']
                if f'lambda_type_{k}' in params_k: type_specific_params['lambda'] = params_k[f'lambda_type_{k}']

                # Hot-start: Use previous value function as initial guess
                initial_v = hot_start_state.get(k, None)

                # Solve Bellman equation with hot-start
                converged_v = solve_bellman_for_params(
                    params=type_specific_params, state_space=state_space, agent_type=int(k),
                    beta=beta, transition_matrices=transition_matrices, regions_df=regions_df,
                    distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
                    initial_v=initial_v, verbose=(objective_function.call_count == 1)
                )

                # Update hot-start state with converged value function
                hot_start_state[k] = converged_v

                # Calculate likelihood using the converged value function
                log_lik_k_obs_vector = calculate_likelihood_from_v(
                    converged_v=converged_v, params=type_specific_params, observed_data=observed_data,
                    state_space=state_space, agent_type=int(k), beta=beta,
                    transition_matrices=transition_matrices, regions_df=regions_df,
                    distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix
                )
                log_lik_k_obs_vector = np.nan_to_num(log_lik_k_obs_vector, nan=-1e10, posinf=-1e10, neginf=-1e10)

                # --- PERFORMANCE FIX: Use fast NumPy aggregation ---
                individual_log_lik = np.bincount(individual_indices, weights=log_lik_k_obs_vector)
                total_weighted_log_lik += np.sum(weights * individual_log_lik)

            return -total_weighted_log_lik
        except Exception as e:
            logger.error(f"  Error in objective function: {e}", exc_info=True)
            return 1e10

    initial_param_values, param_names = _pack_params(initial_params)
    logger.info(f"\n  Starting L-BFGS-B optimization...")

    try:
        result = minimize(
            objective_function, initial_param_values, args=(param_names,), method='L-BFGS-B',
            options={'disp': False, 'maxiter': 15, 'gtol': 1e-3, 'ftol': 1e-3}
        )
        if result.success:
            updated_params = _unpack_params(result.x, param_names, initial_params['n_choices'])
            logger.info(f"  ✓ M-step optimization successful in {result.nit} iterations.")
        else:
            logger.warning(f"  ⚠ M-step optimization stopped early: {result.message}")
            updated_params = _unpack_params(result.x, param_names, initial_params['n_choices'])
    except Exception as e:
        logger.error(f"  ✗ Error in M-step optimization: {e}", exc_info=True)
        updated_params = initial_params

    updated_pi_k = posterior_probs.mean(axis=0)
    MIN_PROB = 0.05
    updated_pi_k = np.maximum(updated_pi_k, MIN_PROB)
    updated_pi_k = updated_pi_k / np.sum(updated_pi_k)

    logger.info(f"  M-step completed. Type probabilities: {updated_pi_k}")
    return updated_params, updated_pi_k, hot_start_state

@timing_decorator
def run_em_algorithm(
    observed_data: pd.DataFrame, state_space: pd.DataFrame, transition_matrices: Dict[str, np.ndarray],
    beta: float, n_types: int, regions_df: pd.DataFrame, distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-5,
    n_choices: int = 31, use_migration_behavior_init: bool = True,
) -> Dict[str, Any]:
    logger = setup_logging()
    
    if use_migration_behavior_init:
        logger.info("Initializing with migration behavior analysis...")
        try:
            _, initial_posterior_probs = identify_migration_behavior_types(observed_data, n_types)
            initial_params = create_behavior_based_initial_params(n_types)
            pi_k = initial_posterior_probs.mean(axis=0)
            pi_k = np.maximum(pi_k, 1e-6)
            pi_k = pi_k / np.sum(pi_k)
        except Exception as e:
            logger.error(f"Error in migration behavior initialization: {e}", exc_info=True)
            logger.warning("Falling back to uniform initialization...")
            initial_params = {
                "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0, "rho_base_tier_1": 1.0,
                "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1, "gamma_0_type_0": 1.0,
                "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2, "gamma_3": -0.4,
                "gamma_4": 0.01, "gamma_5": -0.05, "alpha_climate": 0.1, "alpha_health": 0.1,
                "alpha_education": 0.1, "alpha_public_services": 0.1, "n_choices": n_choices
            }
            pi_k = np.full(n_types, 1 / n_types)
    else:
        logger.info("Using standard uniform initialization...")
        initial_params = {
            "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0, "rho_base_tier_1": 1.0,
            "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1, "gamma_0_type_0": 1.0,
            "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2, "gamma_3": -0.4,
            "gamma_4": 0.01, "gamma_5": -0.05, "alpha_climate": 0.1, "alpha_health": 0.1,
            "alpha_education": 0.1, "alpha_public_services": 0.1, "n_choices": n_choices
        }
        pi_k = np.full(n_types, 1 / n_types)

    # Initialize hot-start state dictionary
    hot_start_state = {}

    old_log_likelihood = -np.inf
    for i in range(max_iterations):
        logger.info(f"\n--- EM Iteration {i+1}/{max_iterations} ---")
        
        logger.info("Running E-step...")
        posterior_probs, log_likelihood_matrix = e_step(
            initial_params, pi_k, observed_data, state_space, transition_matrices, beta,
            regions_df, distance_matrix, adjacency_matrix, n_types=n_types
        )
        
        logger.info("Running M-step...")
        initial_params, pi_k, hot_start_state = m_step(
            posterior_probs, initial_params, observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix, n_types=n_types,
            hot_start_state=hot_start_state
        )
        
        new_log_likelihood = _calculate_mixture_log_likelihood(log_likelihood_matrix, pi_k)
        logger.info(f"Log-Likelihood: {new_log_likelihood:.4f}")
        change = abs(new_log_likelihood - old_log_likelihood)
        logger.info(f"Change in log-likelihood: {change:.6f} (Tolerance: {tolerance})")

        if change < tolerance and i > 0:
            logger.info(f"EM algorithm converged after {i+1} iterations.")
            break
        
        old_log_likelihood = new_log_likelihood
        if i == max_iterations - 1:
            logger.warning("EM algorithm reached max iterations without converging.")

    return {
        "structural_params": initial_params, "type_probabilities": pi_k,
        "final_log_likelihood": new_log_likelihood, "n_iterations": i + 1,
        "posterior_probs": posterior_probs, "log_likelihood_matrix": log_likelihood_matrix
    }