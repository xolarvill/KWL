import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, List

from src.model.likelihood import calculate_log_likelihood
from src.estimation.migration_behavior_analysis import identify_migration_behavior_types, create_behavior_based_initial_params

# --- Helper functions for parameter handling ---

def _pack_params(params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Packs a dictionary of parameters into a numpy array for the optimizer."""
    # 排除n_choices，它不应该被优化
    param_names = sorted([k for k in params.keys() if k != 'n_choices'])
    param_values = np.array([params[name] for name in param_names])
    return param_values, param_names

def _unpack_params(param_values: np.ndarray, param_names: List[str], n_choices: int) -> Dict[str, Any]:
    """Unpacks a numpy array back into a dictionary of parameters."""
    params_dict = dict(zip(param_names, param_values))
    params_dict['n_choices'] = n_choices  # 添加回n_choices作为整数
    return params_dict

# --- Log-Likelihood Calculation ---

def _calculate_mixture_log_likelihood(
    log_likelihood_matrix: np.ndarray, pi_k: np.ndarray
) -> float:
    """
    Calculates the total log-likelihood of the mixture model with numerical stability.
    
    Args:
        log_likelihood_matrix (np.ndarray): Matrix of log-likelihoods (N, K).
        pi_k (np.ndarray): Type probabilities (K,).

    Returns:
        float: The total log-likelihood of the sample.
    """
    try:
        # Ensure pi_k has a minimum value to avoid log(0)
        pi_k_safe = np.maximum(pi_k, 1e-10)
        pi_k_safe = pi_k_safe / np.sum(pi_k_safe)  # Renormalize
        
        log_pi_k = np.log(pi_k_safe)
        weighted_log_lik = log_likelihood_matrix + log_pi_k
        
        # Use log-sum-exp for numerical stability
        max_log_lik = np.max(weighted_log_lik, axis=1, keepdims=True)
        log_marginal_lik = max_log_lik.squeeze() + np.log(
            np.sum(np.exp(weighted_log_lik - max_log_lik), axis=1)
        )
        
        # Check for invalid values
        if np.any(np.isnan(log_marginal_lik)) or np.any(np.isinf(log_marginal_lik)):
            print("Warning: Invalid values in log_marginal_lik")
            return -1e10  # Return a very negative value instead of inf
        
        return np.sum(log_marginal_lik)
    except Exception as e:
        print(f"Error in mixture log-likelihood calculation: {e}")
        return -1e10

# --- EM Algorithm Steps ---

def e_step(
    params: Dict[str, Any],
    pi_k: np.ndarray,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_types: int = None,
    force_type_separation: bool = True  # 新增参数：强制类型分离
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the E-step: calculates posterior probabilities and the log-likelihood matrix
    with numerical stability improvements. Supports type-specific parameters.
    """
    N = observed_data["individual_id"].nunique()
    K = len(pi_k) if n_types is None else n_types
    log_likelihood_matrix = np.zeros((N, K))

    # Group data by individual once
    grouped_data = observed_data.groupby("individual_id")
    
    print(f"  Computing log-likelihoods for {N} individuals across {K} types...")
    
    for k in range(K):
        try:
            # Create type-specific parameters for this type k
            type_specific_params = params.copy()
            
            # Update parameters that are type-specific
            if f'gamma_0_type_{k}' in params:
                type_specific_params['gamma_0'] = params[f'gamma_0_type_{k}']
            if f'gamma_1_type_{k}' in params:
                type_specific_params['gamma_1'] = params[f'gamma_1_type_{k}']
            if f'alpha_home_type_{k}' in params:
                type_specific_params['alpha_home'] = params[f'alpha_home_type_{k}']
            if f'lambda_type_{k}' in params:
                type_specific_params['lambda'] = params[f'lambda_type_{k}']
            
            # Ensure agent_type is an integer
            agent_type_int = int(k)
            log_lik_obs = -calculate_log_likelihood(
                params=type_specific_params,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=int(k),
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
            )
            
            # Check for invalid values
            if np.any(np.isnan(log_lik_obs)) or np.any(np.isinf(log_lik_obs)):
                print(f"  Warning: Invalid log-likelihood values for type {k}")
                log_lik_obs = np.nan_to_num(log_lik_obs, nan=-1e10, posinf=-1e10, neginf=-1e10)
            
            # Sum log-likelihoods for each individual
            observed_data_copy = observed_data.copy()
            observed_data_copy['log_lik_obs'] = log_lik_obs
            individual_log_lik = observed_data_copy.groupby("individual_id")['log_lik_obs'].sum()
            log_likelihood_matrix[:, k] = individual_log_lik.values
            
        except Exception as e:
            print(f"  Error computing likelihood for type {k}: {e}")
            log_likelihood_matrix[:, k] = -1e10

    # Calculate posterior probabilities using Bayes' rule in log space with numerical stability
    try:
        # Ensure pi_k has minimum value
        pi_k_safe = np.maximum(pi_k, 1e-10)
        pi_k_safe = pi_k_safe / np.sum(pi_k_safe)
        
        log_pi_k = np.log(pi_k_safe)
        weighted_log_lik = log_likelihood_matrix + log_pi_k
        
        # Use log-sum-exp trick for numerical stability
        max_log_lik = np.max(weighted_log_lik, axis=1, keepdims=True)
        log_marginal_lik = max_log_lik + np.log(
            np.sum(np.exp(weighted_log_lik - max_log_lik), axis=1, keepdims=True)
        )
        log_posterior = weighted_log_lik - log_marginal_lik
        
        # Compute posterior probabilities
        posterior_probs = np.exp(log_posterior)
        
        # Ensure probabilities sum to 1 and are non-negative
        posterior_probs = np.maximum(posterior_probs, 0)
        row_sums = posterior_probs.sum(axis=1, keepdims=True)
        posterior_probs = posterior_probs / np.maximum(row_sums, 1e-10)
        
        # 强制类型分离：防止某个类型完全消失
        if force_type_separation:
            # 设置最小类型概率阈值，确保每个类型都有一定代表性
            MIN_TYPE_PROB = 0.05  # 每个类型至少占5%
            for i in range(N):
                for k_idx in range(K):
                    # 如果后验概率过低，提升到最小值
                    if posterior_probs[i, k_idx] < MIN_TYPE_PROB:
                        # 重新分配概率以确保每个类型都有最小占比
                        remaining_prob = 1.0 - (K - 1) * MIN_TYPE_PROB
                        if remaining_prob > MIN_TYPE_PROB:
                            posterior_probs[i, k_idx] = MIN_TYPE_PROB
                            # 重新标准化其他类型
                            other_types_total = np.sum(posterior_probs[i, :]) - MIN_TYPE_PROB
                            if other_types_total > 0:
                                for j in range(K):
                                    if j != k_idx:
                                        posterior_probs[i, j] = posterior_probs[i, j] * (1 - MIN_TYPE_PROB) / other_types_total
            
            # 再次归一化确保每行和为1
            row_sums = posterior_probs.sum(axis=1, keepdims=True)
            posterior_probs = posterior_probs / np.maximum(row_sums, 1e-10)
            
    except Exception as e:
        print(f"Error in E-step posterior probability calculation: {e}")
        posterior_probs = np.zeros((N, K))
    
    print(f"  E-step completed. Posterior probability range: [{posterior_probs.min():.6f}, {posterior_probs.max():.6f}]")
    
    return posterior_probs, log_likelihood_matrix

def m_step(
    posterior_probs: np.ndarray,
    initial_params: Dict[str, Any],
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_types: int = None
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Performs the M-step: updates parameters and type probabilities with numerical safeguards.
    Supports type-specific parameters.
    """
    K = posterior_probs.shape[1] if n_types is None else n_types
    
    # Print type weight diagnostics
    print(f"\n  Type weight diagnostics:")
    for k in range(K):
        weight_sum = np.sum(posterior_probs[:, k])
        weight_mean = np.mean(posterior_probs[:, k])
        print(f"    Type {k}: sum={weight_sum:10.4f}, mean={weight_mean:.6f}")

    # Create the objective function for the optimizer
    def objective_function(param_values: np.ndarray, param_names: List[str]) -> float:
        # Add function call counter
        if not hasattr(objective_function, 'call_count'):
            objective_function.call_count = 0
        objective_function.call_count += 1

        params_k = _unpack_params(param_values, param_names, initial_params['n_choices'])
        total_weighted_log_lik = 0

        try:
            for k in range(K):
                weights = posterior_probs[:, k]
                weight_sum = np.sum(weights)

                # Skip if weights are too small
                if weight_sum < 1e-10:
                    if objective_function.call_count == 1:
                        print(f"    Skipping type {k} (weight sum {weight_sum:.2e} < 1e-10)")
                    continue
                
                # Create type-specific parameters for this type k
                type_specific_params = params_k.copy()
                
                # Update parameters that are type-specific
                if f'gamma_0_type_{k}' in params_k:
                    type_specific_params['gamma_0'] = params_k[f'gamma_0_type_{k}']
                if f'gamma_1_type_{k}' in params_k:
                    type_specific_params['gamma_1'] = params_k[f'gamma_1_type_{k}']
                if f'alpha_home_type_{k}' in params_k:
                    type_specific_params['alpha_home'] = params_k[f'alpha_home_type_{k}']
                if f'lambda_type_{k}' in params_k:
                    type_specific_params['lambda'] = params_k[f'lambda_type_{k}']
                
                # Calculate log-likelihood for this type
                # 确保agent_type是整数
                agent_type_int = int(k)
                # Disable verbose for M-step optimization (except first call)
                verbose_flag = (objective_function.call_count == 1)
                log_lik_k_obs = -calculate_log_likelihood(
                    params=type_specific_params,
                    observed_data=observed_data,
                    state_space=state_space,
                    agent_type=int(k),
                    beta=beta,
                    transition_matrices=transition_matrices,
                    regions_df=regions_df,
                    distance_matrix=distance_matrix,
                    adjacency_matrix=adjacency_matrix,
                    verbose=verbose_flag,  # Control verbosity
                )
                
                # Check for invalid values
                if np.any(np.isnan(log_lik_k_obs)) or np.any(np.isinf(log_lik_k_obs)):
                    log_lik_k_obs = np.nan_to_num(log_lik_k_obs, nan=-1e10, posinf=-1e10, neginf=-1e10)
                
                # Weight it by the posterior probabilities
                observed_data_copy = observed_data.copy()
                observed_data_copy['log_lik_k_obs'] = log_lik_k_obs
                individual_log_lik = observed_data_copy.groupby("individual_id")['log_lik_k_obs'].sum()
                total_weighted_log_lik += np.sum(weights * individual_log_lik)
            
            # Apply entropy regularization to encourage balanced type assignment
            # This helps prevent degeneracy by penalizing concentrated assignments
            type_probs = posterior_probs.mean(axis=0)
            entropy_reg = -0.1 * np.sum(type_probs * np.log(type_probs + 1e-10))
            
            # Return negative for minimization (add regularization to make it a penalty in the negative log-lik context)
            return -(total_weighted_log_lik + entropy_reg)
            
        except Exception as e:
            print(f"  Error in objective function: {e}")
            return 1e10  # Return large positive value to discourage this parameter set

    # Initial guess for parameters
    initial_param_values, param_names = _pack_params(initial_params)
    
    # Run optimizer with error handling
    print(f"\n  Starting BFGS optimization...")
    try:
        result = minimize(
            objective_function,
            initial_param_values,
            args=(param_names,),
            method='BFGS',
            options={
                'disp': False,  # We'll print our own summary
                'maxiter': 50,  # Limit iterations to prevent excessive computation
                'gtol': 1e-4,   # Gradient tolerance for convergence
            }
        )

        if result.success:
            updated_params = _unpack_params(result.x, param_names, initial_params['n_choices'])
            print(f"  ✓ M-step optimization successful:")
            print(f"    - Function evaluations: {objective_function.call_count}")
            print(f"    - Final objective: {result.fun:.4f}")
            print(f"    - Iterations: {result.nit}")
        else:
            print(f"  ✗ M-step optimization did not converge: {result.message}")
            print(f"    - Using previous parameters.")
            updated_params = initial_params

    except Exception as e:
        print(f"  ✗ Error in M-step optimization: {e}")
        print(f"    - Using previous parameters.")
        updated_params = initial_params

    # Update type probabilities with lower bound constraint
    updated_pi_k = posterior_probs.mean(axis=0)
    
    # Enforce minimum probability with a more balanced approach for low migration data
    MIN_PROB = 0.05  # 增加最小概率以防止类型完全消失
    updated_pi_k = np.maximum(updated_pi_k, MIN_PROB)
    updated_pi_k = updated_pi_k / np.sum(updated_pi_k)  # Renormalize
    
    print(f"  M-step completed. Type probabilities: {updated_pi_k}")

    return updated_params, updated_pi_k

# --- Main EM Algorithm Runner ---

def run_em_algorithm(
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: Dict[str, np.ndarray],
    beta: float,
    n_types: int,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    n_choices: int = 31,
    use_migration_behavior_init: bool = True,  # New flag to control initialization method
) -> Dict[str, Any]:
    """
    The main function to run the EM algorithm with migration behavior-based initialization.
    """
    # 1. Initialize parameters and type probabilities based on migration behavior
    if use_migration_behavior_init:
        print("Initializing with migration behavior analysis...")
        # Use migration behavior to initialize type probabilities
        try:
            _, initial_posterior_probs = identify_migration_behavior_types(observed_data, n_types)
            
            # Use behavior-based parameters
            initial_params = create_behavior_based_initial_params(n_types)
            
            # Compute initial type probabilities from the posterior
            pi_k = initial_posterior_probs.mean(axis=0)
            # Ensure minimum probability to prevent degeneracy
            MIN_PROB = 1e-6
            pi_k = np.maximum(pi_k, MIN_PROB)
            pi_k = pi_k / np.sum(pi_k)  # Renormalize
            
        except Exception as e:
            print(f"Error in migration behavior initialization: {e}")
            print("Falling back to uniform initialization...")
            initial_params = {
                "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0,
                "rho_base_tier_1": 1.0, "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1,
                "gamma_0_type_0": 1.0, "gamma_0_type_1": 1.5, "gamma_1": -0.1, "gamma_2": 0.2,
                "gamma_3": -0.4, "gamma_4": 0.01, "gamma_5": -0.05,
                "alpha_climate": 0.1, "alpha_health": 0.1, "alpha_education": 0.1, "alpha_public_services": 0.1,
                "n_choices": n_choices  # Add n_choices to params
            }
            pi_k = np.full(n_types, 1 / n_types)
    else:
        # Standard initialization
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
            initial_params, pi_k, observed_data, state_space, transition_matrices, 
            beta, regions_df, distance_matrix=distance_matrix, 
            adjacency_matrix=adjacency_matrix, n_types=n_types
        )
        
        # 3. M-Step
        print("Running M-step...")
        initial_params, pi_k = m_step(
            posterior_probs, initial_params, observed_data, state_space, transition_matrices, 
            beta, regions_df, distance_matrix=distance_matrix, 
            adjacency_matrix=adjacency_matrix, n_types=n_types
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
