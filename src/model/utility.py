
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
    # FIX: Add protection against log(0) or log(negative)
    w_ij_safe = np.maximum(w_ij, 1e-6)
    w_ref_safe = np.maximum(w_ref, 1e-6)
    
    log_w_ij = np.log(w_ij_safe)
    log_w_ref = np.log(w_ref_safe)
    
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
    """
    if not is_hukou_mismatch:
        return 0.0

    # FIX: Ensure region_tier is a clean integer for indexing
    # Handle potential NaNs and cast to int
    region_tier_int = int(region_tier) if pd.notna(region_tier) else 1 # Default to tier 1 if NaN
    base_penalty = params.get(f"rho_base_tier_{region_tier_int}", 0.0)

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
    # 使用LightGBM插件预测的工资，而非原始观测值
    predicted_wage = data.get("wage_predicted", data.get("income", data.get("wage_j", 0.0)))
    income_utility = _calculate_income_utility(
        w_ij=predicted_wage,
        w_ref=data.get("wage_ref", data.get("income", predicted_wage)),  # 如果没有参考工资，使用收入作为参考
        alpha_w=params["alpha_w"],
        lambda_loss=params["lambda"],
    )

    # 2. Amenities (使用目的地地区的特征)
    amenity_utility = (
        params["alpha_climate"] * data.get("amenity_climate_dest", 0.0)
        + params["alpha_health"] * data.get("amenity_health_dest", 0.0)
        + params["alpha_education"] * data.get("amenity_education_dest", 0.0)
        + params["alpha_public_services"] * data.get("amenity_public_services_dest", 0.0)
    )

    # 3. Home Premium
    is_home = data.get("provcd_t", 0) == data.get("hukou_prov", 0)
    home_premium = params["alpha_home"] * is_home

    # 4. Hukou Penalty (Eq. 8)
    hukou_penalty = _calculate_hukou_penalty(
        is_hukou_mismatch=(not is_home),
        region_tier=data.get("tier_dest", 1),  # 假设默认为1级城市
        edu_level=data.get("amenity_education_dest", 0.0), # 使用目的地的教育复合指标
        health_level=data.get("amenity_health_dest", 0.0), # 使用目的地的医疗复合指标
        housing_price=data.get("房价（元每平方）_dest", 0.0), # 使用目的地的房价
        params=params,
    )

    # 5. Migration Cost (Eq. 9)
    migration_cost = _calculate_migration_cost(
        is_moving=(data.get("provcd_t", 0) != data.get("prev_provcd", 0)),
        distance=0.0,  # 距离信息需要通过其他方式获取，暂时设为0
        is_adjacent=0,  # 邻接信息需要通过其他方式获取，暂时设为0
        is_return_move=0,  # 回流信息需要通过其他方式获取，暂时设为0
        age=data.get("age", 0),
        destination_population=data.get("地区基本经济面_dest", 1000000),  # 使用目的地的经济面作为代理
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
