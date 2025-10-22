import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, List
from joblib import Parallel, delayed
import logging
import time
import os
from functools import wraps

from src.model.likelihood import calculate_likelihood_from_v, clear_bellman_cache
from src.model.bellman import solve_bellman_equation_individual # Import the new individual solver
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
    # **关键修复**: 显式排除 gamma_0_type_0，因为它已被归一化为0
    param_names = sorted([k for k in params.keys() if k not in ['n_choices', 'gamma_0_type_0']])
    param_values = np.array([params[name] for name in param_names])
    return param_values, param_names

def _unpack_params(param_values: np.ndarray, param_names: List[str], n_choices: int) -> Dict[str, Any]:
    params_dict = dict(zip(param_names, param_values))
    params_dict['n_choices'] = n_choices
    # **关键修复**: 将被归一化的参数重新加回字典
    params_dict['gamma_0_type_0'] = 0.0
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

@timing_decorator
def e_step(
    params: Dict[str, Any], pi_k: np.ndarray, observed_data: pd.DataFrame, state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray], beta: float, regions_df: pd.DataFrame,
    distance_matrix: np.ndarray, adjacency_matrix: np.ndarray, n_types: int = None,
    force_type_separation: bool = True, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    logger = logging.getLogger()
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    unique_individuals = observed_data['individual_id'].unique()
    N = len(unique_individuals)
    K = len(pi_k) if n_types is None else n_types
    log_likelihood_matrix = np.zeros((N, K))
    
    # Create a mapping from individual_id to matrix row index for easy aggregation
    individual_to_idx = {ind_id: i for i, ind_id in enumerate(unique_individuals)}

    logger.info(f"  Computing log-likelihoods for {N} individuals across {K} types using {n_jobs} cores (granularity: individual chunks)...")

    # --- New Worker Function for a chunk of individuals ---
    def _compute_likelihood_for_chunk(individual_ids_chunk: np.ndarray, k: int):
        chunk_results = {}
        total_bellman_time = 0
        total_likelihood_time = 0
        
        type_specific_params = params.copy()
        if f'gamma_0_type_{k}' in params:
            type_specific_params['gamma_0'] = params[f'gamma_0_type_{k}']

        chunk_data = observed_data[observed_data['individual_id'].isin(individual_ids_chunk)]
        
        for individual_id, individual_df in chunk_data.groupby('individual_id'):
            try:
                bellman_start = time.time()
                converged_v_individual, _ = solve_bellman_equation_individual(
                    utility_function=None, individual_data=individual_df,
                    params=type_specific_params, agent_type=int(k), beta=beta,
                    transition_matrices=transition_matrices, regions_df=regions_df,
                    distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
                    verbose=False
                )
                total_bellman_time += time.time() - bellman_start

                likelihood_start = time.time()
                log_lik_obs_vector = calculate_likelihood_from_v(
                    converged_v=converged_v_individual, params=type_specific_params,
                    observed_data=individual_df, state_space=state_space,
                    agent_type=int(k), beta=beta, transition_matrices=transition_matrices,
                    regions_df=regions_df, distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix
                )
                total_likelihood_time += time.time() - likelihood_start
                
                chunk_results[individual_id] = np.sum(log_lik_obs_vector)
            except Exception as e:
                logger.error(f"  Error on individual {individual_id} for type {k}: {e}")
                chunk_results[individual_id] = -1e10
        return k, chunk_results, total_bellman_time, total_likelihood_time

    # --- New Parallelization Strategy ---
    n_chunks = n_jobs * 4
    if N > 0:
        individual_chunks = np.array_split(unique_individuals, n_chunks)
        tasks = [(chunk, k) for chunk in individual_chunks for k in range(K)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_likelihood_for_chunk)(chunk, k) for chunk, k in tasks
        )

        # Aggregate results and timings
        total_bellman_time_all = 0
        total_likelihood_time_all = 0
        for k, chunk_results, bellman_time, likelihood_time in results:
            total_bellman_time_all += bellman_time
            total_likelihood_time_all += likelihood_time
            for ind_id, log_lik in chunk_results.items():
                if ind_id in individual_to_idx:
                    log_likelihood_matrix[individual_to_idx[ind_id], k] = log_lik
        
        logger.info(f"  E-step timing breakdown (sum over all chunks):")
        logger.info(f"    - Total time in Bellman solves: {total_bellman_time_all:.2f}s")
        logger.info(f"    - Total time in Likelihood calcs: {total_likelihood_time_all:.2f}s")
    
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

@timing_decorator
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

    # --- Objective Function for L-BFGS-B ---
    # Use a class to maintain state like call count
    class Objective:
        def __init__(self):
            self.call_count = 0
            self.last_call_time = time.time()

        def function(self, param_values: np.ndarray, param_names: List[str]) -> float:
            self.call_count += 1
            start_time = time.time()
            
            params_k = _unpack_params(param_values, param_names, initial_params['n_choices'])
            total_weighted_log_lik = 0
            
            log_msg_header = f"    [M-step Objective Call #{self.call_count}]"
            logger.info(f"{log_msg_header} Starting...")

            try:
                # --- This inner loop is the most expensive part ---
                for k in range(K):
                    weights = posterior_probs[:, k]
                    if np.sum(weights) < 1e-10: continue

                    type_specific_params = params_k.copy()
                    if f'gamma_0_type_{k}' in params_k:
                        type_specific_params['gamma_0'] = params_k[f'gamma_0_type_{k}']
                    
                    # CRITICAL FOR DIAGNOSIS: Disable caching during optimization
                    # The optimizer needs to see the function's true curvature.
                    converged_v = solve_bellman_for_params(
                        params=type_specific_params, state_space=state_space, agent_type=int(k),
                        beta=beta, transition_matrices=transition_matrices, regions_df=regions_df,
                        distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
                        initial_v=hot_start_state.get(k), verbose=False, use_cache=False
                    )

                    log_lik_k_obs_vector = calculate_likelihood_from_v(
                        converged_v=converged_v, params=type_specific_params, observed_data=observed_data,
                        state_space=state_space, agent_type=int(k), beta=beta,
                        transition_matrices=transition_matrices, regions_df=regions_df,
                        distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix
                    )
                    log_lik_k_obs_vector = np.nan_to_num(log_lik_k_obs_vector, nan=-1e10, posinf=-1e10, neginf=-1e10)
                    individual_log_lik = np.bincount(individual_indices, weights=log_lik_k_obs_vector)
                    total_weighted_log_lik += np.sum(weights * individual_log_lik)

                neg_ll = -total_weighted_log_lik
                duration = time.time() - start_time
                time_since_last = duration + (start_time - self.last_call_time)
                self.last_call_time = time.time()

                logger.info(f"{log_msg_header} Finished. Duration: {duration:.2f}s. Total time since last call: {time_since_last:.2f}s. NegLogLik: {neg_ll:.4f}")
                
                # Log first 3 parameter values for progress tracking
                param_subset = param_values[:3]
                logger.info(f"{log_msg_header} Params (first 3): {np.round(param_subset, 4)}")

                return neg_ll
            except Exception as e:
                logger.error(f"{log_msg_header} Error: {e}", exc_info=True)
                return 1e10

    objective = Objective()
    initial_param_values, param_names = _pack_params(initial_params)
    logger.info(f"\n  Starting L-BFGS-B optimization...")
    logger.info(f"  Optimizing {len(param_names)} parameters.")
    
    # 为日志记录一个初始目标函数值
    initial_neg_ll = objective.function(initial_param_values, param_names)
    logger.info(f"  Initial objective value (neg_log_lik): {initial_neg_ll:.4f}")
    objective.call_count = 0 # 重置计数器

    try:
        result = minimize(
            objective.function, initial_param_values, args=(param_names,), method='L-BFGS-B',
            options={'disp': False, 'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-5, 'eps': 1e-6}
        )
        
        final_neg_ll = result.fun
        logger.info(f"  L-BFGS-B result: success={result.success}, nit={result.nit}, nfev={result.nfev}, message='{result.message.decode('utf-8') if isinstance(result.message, bytes) else result.message}'")
        logger.info(f"  Objective value change: {initial_neg_ll:.4f} -> {final_neg_ll:.4f} (Δ {- (initial_neg_ll - final_neg_ll):.4f})")

        if result.success or result.nit > 0:
            updated_params = _unpack_params(result.x, param_names, initial_params['n_choices'])

            # 显示前5个参数的变化
            param_changes = []
            for pname in param_names[:5]:
                old_val = initial_params.get(pname, 0)
                new_val = updated_params.get(pname, 0)
                change = new_val - old_val
                param_changes.append(f"{pname}: {old_val:.4f}→{new_val:.4f} (Δ{change:+.4f})")
            logger.info(f"  Parameter changes (first 5): {'; '.join(param_changes)}")

            if result.success:
                logger.info(f"  ✓ M-step optimization successful in {result.nit} iterations.")
            else:
                logger.warning(f"  ⚠ M-step optimization stopped after {result.nit} iterations but parameters were updated.")
        else:
            logger.warning(f"  ⚠ M-step optimization failed to make progress: {result.message}")
            updated_params = initial_params # 如果完全没进展，则不更新参数
            
    except Exception as e:
        logger.error(f"  ✗ Error in M-step optimization: {e}", exc_info=True)
        updated_params = initial_params

    # After finding the best parameters, calculate the final V functions
    # to pass as a hot start to the NEXT EM iteration.
    new_hot_start_state = {}
    for k in range(K):
        type_specific_params = updated_params.copy()
        if f'gamma_0_type_{k}' in updated_params:
            type_specific_params['gamma_0'] = updated_params[f'gamma_0_type_{k}']
        
        converged_v = solve_bellman_for_params(
            params=type_specific_params, state_space=state_space, agent_type=int(k),
            beta=beta, transition_matrices=transition_matrices, regions_df=regions_df,
            distance_matrix=distance_matrix, adjacency_matrix=adjacency_matrix,
            initial_v=hot_start_state.get(k, None), # Use the old hot_start as a guess
            use_cache=False
        )
        new_hot_start_state[k] = converged_v

    updated_pi_k = posterior_probs.mean(axis=0)
    MIN_PROB = 0.05
    updated_pi_k = np.maximum(updated_pi_k, MIN_PROB)
    updated_pi_k = updated_pi_k / np.sum(updated_pi_k)

    logger.info(f"  M-step completed. Type probabilities: {updated_pi_k}")
    return updated_params, updated_pi_k, new_hot_start_state

@timing_decorator
def run_em_algorithm(
    observed_data: pd.DataFrame, state_space: pd.DataFrame, transition_matrices: Dict[str, np.ndarray],
    beta: float, n_types: int, regions_df: pd.DataFrame, distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-5,
    n_choices: int = 31, use_migration_behavior_init: bool = True,
    initial_params: Dict[str, Any] = None, initial_pi_k: np.ndarray = None
) -> Dict[str, Any]:
    logger = setup_logging()

    # 如果提供了自定义初始值，直接使用
    if initial_params is not None:
        logger.info("Using provided initial parameters...")
        EM_initial_params = initial_params
        pi_k = initial_pi_k if initial_pi_k is not None else np.full(n_types, 1 / n_types)
    elif use_migration_behavior_init:
        logger.info("Initializing with migration behavior analysis...")
        try:
            _, initial_posterior_probs = identify_migration_behavior_types(observed_data, n_types)
            EM_initial_params = create_behavior_based_initial_params(n_types)
            pi_k = initial_posterior_probs.mean(axis=0)
            pi_k = np.maximum(pi_k, 1e-6)
            pi_k = pi_k / np.sum(pi_k)
        except Exception as e:
            logger.error(f"Error in migration behavior initialization: {e}", exc_info=True)
            logger.warning("Falling back to uniform initialization...")
            EM_initial_params = {
                "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0, "rho_base_tier_1": 1.0,
                "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1, "gamma_0_type_0": 1.0,
                "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2, "gamma_3": -0.4,
                "gamma_4": 0.01, "gamma_5": -0.05, "alpha_climate": 0.1, "alpha_health": 0.1,
                "alpha_education": 0.1, "alpha_public_services": 0.1, "n_choices": n_choices
            }
            pi_k = np.full(n_types, 1 / n_types)
    else:
        logger.info("Using standard uniform initialization...")
        EM_initial_params = {
            "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0, "rho_base_tier_1": 1.0,
            "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1, "gamma_0_type_0": 1.0,
            "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2, "gamma_3": -0.4,
            "gamma_4": 0.01, "gamma_5": -0.05, "alpha_climate": 0.1, "alpha_health": 0.1,
            "alpha_education": 0.1, "alpha_public_services": 0.1, "n_choices": n_choices
        }
        pi_k = np.full(n_types, 1 / n_types)

    # Initialize hot-start state dictionary
    hot_start_state = {}

    # **关键修复**: 对初始参数进行一次性微小扰动，以“激活”优化器
    # 确保即使从一个看似最优的点开始，优化也能启动
    if initial_params:
        EM_initial_params = {}
        for name, value in initial_params.items():
            if isinstance(value, (int, float)) and name not in ['n_choices', 'gamma_0_type_0']:
                noise_std = max(abs(value) * 0.01, 1e-4) # 1% noise
                noise = np.random.normal(0, noise_std)
                EM_initial_params[name] = value + noise
            else:
                EM_initial_params[name] = value
        logger.info("Perturbed initial parameters to activate optimizer.")
    else:
        # Fallback to behavior-based init if no initial params are given
        # (This path is not used by the current scripts but is good practice)
        logger.info("Initializing with migration behavior analysis...")
        try:
            _, initial_posterior_probs = identify_migration_behavior_types(observed_data, n_types)
            EM_initial_params = create_behavior_based_initial_params(n_types)
            pi_k = initial_posterior_probs.mean(axis=0)
        except Exception as e:
            logger.error(f"Error in migration behavior initialization: {e}", exc_info=True)
            # A simple fallback if everything else fails
            from src.config.model_config import ModelConfig
            EM_initial_params = ModelConfig().get_initial_params()

    old_log_likelihood = -np.inf
    for i in range(max_iterations):
        logger.info(f"\n--- EM Iteration {i+1}/{max_iterations} ---")

        # Clear Bellman cache at the start of each EM iteration
        # This ensures we start fresh with potentially updated parameters from previous iteration
        # But within this iteration, cache is reused for efficiency
        clear_bellman_cache()

        logger.info("Running E-step...")
        posterior_probs, log_likelihood_matrix = e_step(
            EM_initial_params, pi_k, observed_data, state_space, transition_matrices, beta,
            regions_df, distance_matrix, adjacency_matrix, n_types=n_types
        )

        logger.info("Running M-step...")
        EM_initial_params, pi_k, hot_start_state = m_step(
            posterior_probs, EM_initial_params, observed_data, state_space, transition_matrices,
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
        "structural_params": EM_initial_params, "type_probabilities": pi_k,
        "final_log_likelihood": new_log_likelihood, "n_iterations": i + 1,
        "posterior_probs": posterior_probs, "log_likelihood_matrix": log_likelihood_matrix
    }