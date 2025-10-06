"""
This module defines the flow utility function for the dynamic discrete choice model
of migration. This version is optimized for vectorized computation.
"""

import numpy as np
from typing import Dict, Any

def calculate_flow_utility_vectorized(
    state_data: Dict[str, np.ndarray],
    region_data: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    params: Dict[str, Any],
    agent_type: int,
    n_states: int,
    n_choices: int
) -> np.ndarray:
    """
    Calculates the total flow utility for all states and choices in a vectorized manner.

    Returns:
        np.ndarray: A matrix of shape (n_states, n_choices) containing flow utilities.
    """
    # --- 1. Prepare data arrays by broadcasting state and choice variables ---
    
    # State variables (shape: (n_states, 1))
    age = state_data['age'][:, np.newaxis]
    prev_loc_indices = state_data['prev_provcd_idx'][:, np.newaxis]
    
    # Choice variables (shape: (1, n_choices))
    dest_loc_indices = np.arange(n_choices)[np.newaxis, :]
    
    # Placeholder values (can be expanded later)
    wage_predicted = np.full((n_states, n_choices), 30000.0)
    wage_ref = np.full((n_states, n_choices), 30000.0)
    
    # --- 2. Calculate Utility Components ---

    # Income Utility (Prospect Theory)
    log_diff = np.log(np.maximum(wage_predicted, 1e-6)) - np.log(np.maximum(wage_ref, 1e-6))
    gain_utility = params["alpha_w"] * log_diff
    loss_utility = params["alpha_w"] * params["lambda"] * log_diff
    income_utility = np.where(wage_predicted >= wage_ref, gain_utility, loss_utility)

    # Amenities Utility
    # Make sure we only use the first n_choices entries to match the number of destinations
    n_regions_in_data = len(region_data["amenity_climate"])
    if n_regions_in_data >= n_choices:
        # Take only the first n_choices entries
        region_climate = region_data["amenity_climate"][:n_choices]
        region_health = region_data["amenity_health"][:n_choices] 
        region_education = region_data["amenity_education"][:n_choices]
        region_public_services = region_data["amenity_public_services"][:n_choices]
        region_population = region_data["常住人口万"][:n_choices]
    else:
        # If we have fewer entries than choices, we need to handle this appropriately
        # This should not happen in normal cases but add safety check
        raise ValueError(f"Number of regions ({n_regions_in_data}) is less than number of choices ({n_choices})")
    
    amenity_utility = (
        params["alpha_climate"] * region_climate[np.newaxis, :]
        + params["alpha_health"] * region_health[np.newaxis, :]
        + params["alpha_education"] * region_education[np.newaxis, :]
        + params["alpha_public_services"] * region_public_services[np.newaxis, :]
    )
    
    # Home Premium (assuming hometown is not state-dependent for now)
    # This would need 'hometown_idx' in state_data for full implementation
    # is_home = (dest_loc_indices == state_data['hometown_idx'][:, np.newaxis])
    # home_premium = params["alpha_home"] * is_home
    home_premium = 0 # Simplified for now

    # Hukou Penalty (simplified)
    # is_hukou_mismatch = (dest_loc_indices != state_data['hukou_prov_idx'][:, np.newaxis])
    # hukou_penalty = params.get(f"rho_base_tier_1", 0.0) * is_hukou_mismatch
    hukou_penalty = 0 # Simplified for now

    # Migration Cost
    is_moving = (dest_loc_indices != prev_loc_indices)
    
    distance = distance_matrix[prev_loc_indices, dest_loc_indices]
    is_adjacent = adjacency_matrix[prev_loc_indices, dest_loc_indices]
    
    fixed_cost = params.get(f"gamma_0_type_{agent_type}", 0.0)
    distance_cost = params["gamma_1"] * distance
    adjacency_discount = params["gamma_2"] * is_adjacent
    age_cost = params["gamma_4"] * age
    
    # Simplified population effect
    log_dest_population = np.log(np.maximum(region_population[np.newaxis, :], 1.0))
    population_discount = params["gamma_5"] * log_dest_population
    
    migration_cost = is_moving * (
        fixed_cost + distance_cost - adjacency_discount + age_cost - population_discount
    )

    # --- 3. Total Utility ---
    # xi_ij (non-economic preference) is assumed to be zero for the deterministic part
    total_utility = (
        income_utility
        + amenity_utility
        + home_premium
        - hukou_penalty
        - migration_cost
    )

    return np.clip(total_utility, -500, 500)

# Keep the original function for compatibility if needed elsewhere, though it's now unused by Bellman
def calculate_flow_utility(*args, **kwargs):
    raise NotImplementedError("Use calculate_flow_utility_vectorized. The non-vectorized version is deprecated.")