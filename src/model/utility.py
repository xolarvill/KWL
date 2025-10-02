
"""
This module defines the flow utility function for the dynamic discrete choice model
of migration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def _calculate_income_utility(
    w_ij: float, w_ref: float, alpha_w: float, lambda_loss: float
) -> float:
    """
    Calculates the utility from income based on prospect theory.

    Args:
        w_ij (float): Realized wage in location j.
        w_ref (float): Reference wage.
        alpha_w (float): Marginal utility of income (in gain domain).
        lambda_loss (float): Loss aversion coefficient.

    Returns:
        float: Utility derived from income.
    """
    log_w_ij = np.log(w_ij)
    log_w_ref = np.log(w_ref)
    
    if log_w_ij >= log_w_ref:
        return alpha_w * (log_w_ij - log_w_ref)
    else:
        return alpha_w * lambda_loss * (log_w_ij - log_w_ref)

def _calculate_hukou_penalty(
    is_hukou_mismatch: bool,
    region_tier: int,
    edu_level: float,
    health_level: float,
    housing_price: float,
    params: Dict[str, float],
) -> float:
    """
    Calculates the penalty for living outside one's hukou registration area.

    Args:
        is_hukou_mismatch (bool): True if living in a non-hukou location.
        region_tier (int): The tier of the city/region.
        edu_level (float): Education amenity level of the region.
        health_level (float): Health amenity level of the region.
        housing_price (float): Housing price in the region.
        params (Dict[str, float]): Dictionary of model parameters.

    Returns:
        float: The calculated hukou penalty.
    """
    if not is_hukou_mismatch:
        return 0.0

    # Base penalty based on city tier
    base_penalty = params.get(f"rho_base_tier_{region_tier}", 0.0)

    # Interaction penalties
    edu_penalty = params["rho_edu"] * edu_level
    health_penalty = params["rho_health"] * health_level
    house_penalty = params["rho_house"] * housing_price

    return base_penalty + edu_penalty + health_penalty + house_penalty

def _calculate_migration_cost(
    is_moving: bool,
    distance: float,
    is_adjacent: bool,
    is_return_move: bool,
    age: int,
    destination_population: float,
    agent_type: int,
    params: Dict[str, float],
) -> float:
    """
    Calculates the one-time cost of migration.

    Args:
        is_moving (bool): True if the agent is moving to a new location.
        distance (float): Distance between origin and destination.
        is_adjacent (bool): True if the destination is adjacent to the origin.
        is_return_move (bool): True if moving back to a previously visited location.
        age (int): Age of the agent.
        destination_population (float): Population of the destination.
        agent_type (int): The unobserved type of the agent.
        params (Dict[str, float]): Dictionary of model parameters.

    Returns:
        float: The calculated migration cost.
    """
    if not is_moving:
        return 0.0

    # Type-specific fixed cost
    fixed_cost = params.get(f"gamma_0_type_{agent_type}", 0.0)

    # Other cost components
    distance_cost = params["gamma_1"] * distance
    adjacency_discount = params["gamma_2"] * is_adjacent
    return_discount = params["gamma_3"] * is_return_move
    age_cost = params["gamma_4"] * age
    population_discount = params["gamma_5"] * destination_population

    total_cost = (
        fixed_cost
        + distance_cost
        - adjacency_discount
        - return_discount
        + age_cost
        - population_discount
    )
    return total_cost

def calculate_flow_utility(
    data: pd.Series,
    params: Dict[str, Any],
    agent_type: int,
    xi_ij: float,
) -> float:
    """
    Calculates the total flow utility for a given choice, state, and parameters.
    This corresponds to equation (3) in the paper.

    Args:
        data (pd.Series): A series/row containing all necessary data for one
                         individual-location choice. Must include columns for wages,
                         amenities, hukou, previous location, etc.
        params (Dict[str, Any]): A dictionary of all model parameters.
        agent_type (int): The agent's unobserved type index.
        xi_ij (float): The non-economic preference match value for the agent and location.

    Returns:
        float: The total deterministic flow utility u_it(j).
    """
    # 1. Income Utility (Eq. 4)
    income_utility = _calculate_income_utility(
        w_ij=data["wage_j"],
        w_ref=data["wage_ref"],
        alpha_w=params["alpha_w"],
        lambda_loss=params["lambda"],
    )

    # 2. Amenities
    amenity_utility = (
        params["alpha_climate"] * data["amenity_climate"]
        + params["alpha_health"] * data["amenity_health"]
        + params["alpha_education"] * data["amenity_education"]
        + params["alpha_public_services"] * data["amenity_public_services"]
    )

    # 3. Home Premium
    is_home = data["j_provcd"] == data["hukou_provcd"]
    home_premium = params["alpha_home"] * is_home

    # 4. Hukou Penalty (Eq. 8)
    hukou_penalty = _calculate_hukou_penalty(
        is_hukou_mismatch=(not is_home),
        region_tier=data["tier_j"],
        edu_level=data["amenity_education"], # Using the composite indicator
        health_level=data["amenity_health"], # Using the composite indicator
        housing_price=data["housing_price_j"],
        params=params,
    )

    # 5. Migration Cost (Eq. 9)
    migration_cost = _calculate_migration_cost(
        is_moving=(data["j_provcd"] != data["prev_provcd"]),
        distance=data["distance_j"],
        is_adjacent=data["is_adjacent_j"],
        is_return_move=data["is_return_j"],
        age=data["age"],
        destination_population=data["population_j"],
        agent_type=agent_type,
        params=params,
    )

    # 6. Non-economic Preference Match
    # This is passed directly as xi_ij

    # Total Utility (Eq. 3)
    total_utility = (
        income_utility
        + amenity_utility
        + home_premium
        - hukou_penalty
        - migration_cost
        + xi_ij
    )

    return total_utility
