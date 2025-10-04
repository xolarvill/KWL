
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
    # FIX: Add protection against log(0) or log(negative) and extreme values
    w_ij_safe = np.maximum(w_ij, 1e-6)
    w_ref_safe = np.maximum(w_ref, 1e-6)

    log_w_ij = np.log(w_ij_safe)
    log_w_ref = np.log(w_ref_safe)

    # Clip log difference to prevent extreme utility values
    log_diff = np.clip(log_w_ij - log_w_ref, -10, 10)

    if log_w_ij >= log_w_ref:
        utility = alpha_w * log_diff
    else:
        utility = alpha_w * lambda_loss * log_diff

    # Final clipping to ensure utility stays in reasonable range
    return np.clip(utility, -100, 100)

def _calculate_hukou_penalty(
    is_hukou_mismatch: bool,
    region_tier: int,
    edu_level: float,
    health_level: float,
    housing_price_ratio: float,
    params: Dict[str, float],
) -> float:
    """
    Calculates the penalty for living outside one's hukou registration area.

    Args:
        is_hukou_mismatch (bool): Whether there is a hukou mismatch.
        region_tier (int): Tier of the region (1, 2, or 3).
        edu_level (float): Education amenity level.
        health_level (float): Health amenity level.
        housing_price_ratio (float): Housing price-to-income ratio (房价收入比).
        params (Dict[str, float]): Model parameters.

    Returns:
        float: The calculated hukou penalty.
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

    # Use housing price-to-income ratio (already normalized)
    # Handle potential invalid values
    housing_price_ratio_safe = np.maximum(housing_price_ratio, 0.0)
    house_penalty = params["rho_house"] * housing_price_ratio_safe

    total_penalty = base_penalty + edu_penalty + health_penalty + house_penalty

    # Clip to reasonable range to prevent extreme penalties
    return np.clip(total_penalty, 0.0, 50.0)

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
        distance (float): Distance between origin and destination (km).
        is_adjacent (bool): True if the destination is adjacent to the origin.
        is_return_move (bool): True if moving back to a previously visited location.
        age (int): Age of the agent.
        destination_population (float): Log of destination population (log(万人)).
                                       用对数形式以处理量纲问题。
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
    # gamma_5应为负数，表示大城市吸引力(人口规模折扣)
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
    # 修正：家乡溢价应与hometown（家乡）而非hukou_prov（户籍地）关联
    is_home = data.get("provcd_dest", 0) == data.get("hometown", 0)
    home_premium = params["alpha_home"] * is_home

    # 4. Hukou Penalty (Eq. 8)
    # 户籍惩罚与hukou_prov（户籍地）关联
    is_hukou_mismatch = data.get("provcd_dest", 0) != data.get("hukou_prov", 0)
    hukou_penalty = _calculate_hukou_penalty(
        is_hukou_mismatch=is_hukou_mismatch,
        region_tier=data.get("tier_dest", 1),
        edu_level=data.get("amenity_education_dest", 0.0),
        health_level=data.get("amenity_health_dest", 0.0),
        housing_price_ratio=data.get("房价收入比_dest", 0.0),  # 使用房价收入比而非房价
        params=params,
    )

    # 5. Migration Cost (Eq. 9)
    # 获取目标地人口(常住人口万)
    dest_population = data.get("常住人口万_dest", 4000.0)  # 默认值约为全国平均
    # 使用对数形式以处理量纲并降低数值范围
    log_dest_population = np.log(np.maximum(dest_population, 1.0))

    migration_cost = _calculate_migration_cost(
        is_moving=(data.get("provcd_t", 0) != data.get("prev_provcd", 0)),
        distance=data.get("distance", 0.0),
        is_adjacent=data.get("is_adjacent", 0),
        is_return_move=data.get("is_return_move", 0),
        age=data.get("age", 0),
        destination_population=log_dest_population,  # 使用对数人口
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

    # Ensure total utility is within reasonable bounds to prevent numerical issues
    # in downstream calculations (e.g., exp in softmax)
    total_utility_clipped = np.clip(total_utility, -500, 500)

    return total_utility_clipped
