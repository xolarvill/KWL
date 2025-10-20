"""
This module defines the flow utility function for the dynamic discrete choice model
of migration. This version is optimized for vectorized computation.
"""

import numpy as np
from typing import Dict, Any, Optional
from src.model.wage_equation import calculate_prospect_theory_utility

def calculate_flow_utility_vectorized(
    state_data: Dict[str, np.ndarray],
    region_data: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    params: Dict[str, Any],
    agent_type: int,
    n_states: int,
    n_choices: int,
    wage_predicted: Optional[np.ndarray] = None,
    wage_reference: Optional[np.ndarray] = None,
    eta_i: Optional[np.ndarray] = None,
    nu_ij: Optional[np.ndarray] = None,
    xi_ij: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculates the total flow utility based on the full theoretical model, vectorized.

    新增参数:
    --------
    wage_predicted : np.ndarray, optional
        预测工资 (n_states, n_choices)。如果提供，将用于前景理论效用计算
    wage_reference : np.ndarray, optional
        参照工资 (n_states, n_choices)。用于前景理论
    eta_i : np.ndarray, optional
        个体固定效应 (n_states,)
    nu_ij : np.ndarray, optional
        个体-地区匹配效应 (n_states, n_choices)
    xi_ij : np.ndarray, optional
        偏好匹配效应 (n_states, n_choices)
    """
    # --- 1. Prepare data arrays by broadcasting state and choice variables ---
    age = state_data['age'][:, np.newaxis]
    prev_loc_indices = state_data['prev_provcd_idx'][:, np.newaxis]
    hukou_loc_indices = state_data['hukou_prov_idx'][:, np.newaxis]
    hometown_loc_indices = state_data['hometown_prov_idx'][:, np.newaxis]
    dest_loc_indices = np.arange(n_choices)[np.newaxis, :]
    
    # --- 2. Calculate Utility Components based on the Paper ---

    # 2.1 Income Utility: 前景理论效用函数 (论文701-714行)
    if wage_predicted is not None and wage_reference is not None:
        # 使用前景理论: U^income = f(ln w - ln w_ref, λ)
        # 展平为1D以便批量计算
        w_pred_flat = wage_predicted.flatten()
        w_ref_flat = wage_reference.flatten()

        # 获取损失厌恶系数（现在是共享参数）
        lambda_param = params.get("lambda", 2.0)
        alpha_w = params.get("alpha_w", 1.0)

        # 计算前景理论效用
        income_utility_flat = calculate_prospect_theory_utility(
            w_current=w_pred_flat,
            w_reference=w_ref_flat,
            alpha_w=alpha_w,
            lambda_loss_aversion=lambda_param,
            use_log_difference=True
        )

        # 重塑为(n_states, n_choices)
        income_utility = income_utility_flat.reshape((n_states, n_choices))
    else:
        # Fallback: 简化的对数工资效用（向后兼容）
        # 使用地区基础工资作为粗略估计
        if '地区基本经济面' in region_data:
            mu_jt = region_data['地区基本经济面'][:n_choices][np.newaxis, :]
            # 简单的基准工资（需要后续扩展）
            wage_approx = np.exp(mu_jt + 10.0)  # 10.0是对数空间的基准
        else:
            wage_approx = np.full((n_states, n_choices), 30000.0)

        income_utility = params.get("alpha_w", 1.0) * np.log(np.maximum(wage_approx, 1e-6))

    # 2.2 Amenities Utility: Sum of alpha_k * A_jk
    # 包含5个amenity维度：气候、医疗、教育、公共服务、自然灾害
    amenity_utility = (
        params["alpha_climate"] * region_data["amenity_climate"][:n_choices][np.newaxis, :]
        + params["alpha_health"] * region_data["amenity_health"][:n_choices][np.newaxis, :]
        + params["alpha_education"] * region_data["amenity_education"][:n_choices][np.newaxis, :]
        + params["alpha_public_services"] * region_data["amenity_public_services"][:n_choices][np.newaxis, :]
        + params["alpha_hazard"] * region_data["amenity_hazard"][:n_choices][np.newaxis, :]
        # 注: amenity_hazard已经是负值（灾害越多越负），所以alpha_hazard > 0表示灾害降低效用
    )
    
    # 2.3 Home Premium: alpha_home * I(j == hometown)
    is_hometown = (dest_loc_indices == hometown_loc_indices)
    home_premium = params.get("alpha_home", 0.0) * is_hometown

    # 2.4 Hukou Penalty: 三档城市分类机制（论文685-690行）
    # ρ_jt = ρ_0,tier(j) + ρ_edu·Edu_jt + ρ_health·Health_jt + ρ_house·House_jt
    is_hukou_mismatch = (dest_loc_indices != hukou_loc_indices)

    # 获取城市分档（优先使用geo.xlsx中的"户籍获取难度"列）
    if '户籍获取难度' in region_data:
        # 户籍获取难度：3=最难（一线城市），2=中等（二线），1=容易（三线）
        hukou_difficulty = region_data['户籍获取难度'][:n_choices][np.newaxis, :]
        # 难度越大，惩罚越大
        rho_base = np.where(
            hukou_difficulty == 3, params.get("rho_base_tier_1", 1.0),  # 一线城市
            np.where(
                hukou_difficulty == 2, params.get("rho_base_tier_2", 0.5),  # 二线城市
                params.get("rho_base_tier_3", 0.2)  # 三线城市
            )
        )
    elif 'city_tier' in region_data:
        # 向后兼容：如果有city_tier列也可使用
        city_tiers = region_data['city_tier'][:n_choices][np.newaxis, :]
        rho_base = np.where(
            city_tiers == 1, params.get("rho_base_tier_1", 1.0),
            np.where(
                city_tiers == 2, params.get("rho_base_tier_2", 0.5),
                params.get("rho_base_tier_3", 0.2)
            )
        )
    else:
        # 如果没有城市分档数据，使用单一基础值
        rho_base = params.get("rho_base_tier_1", 1.0)

    hukou_penalty = is_hukou_mismatch * (
        rho_base
        + params["rho_edu"] * region_data["amenity_education"][:n_choices][np.newaxis, :]
        + params["rho_health"] * region_data["amenity_health"][:n_choices][np.newaxis, :]
        + params["rho_house"] * region_data["房价收入比"][:n_choices][np.newaxis, :]
        # 注：使用房价收入比而非绝对房价，反映相对负担能力
    )

    # 2.5 Migration Cost: C_ijt = I(j != i) * (gamma_0 + ...)
    is_moving = (dest_loc_indices != prev_loc_indices)
    
    # Use log(distance) as per the paper, adding a small constant to avoid log(0)
    distance = distance_matrix[prev_loc_indices, dest_loc_indices]
    log_distance = np.log(np.maximum(distance, 1.0)) # Use 1km as minimum distance
    
    is_adjacent = adjacency_matrix[prev_loc_indices, dest_loc_indices]
    is_return_migration = (dest_loc_indices == hometown_loc_indices) & is_moving

    fixed_cost = params.get(f"gamma_0_type_{agent_type}", 0.0)
    distance_cost = params.get("gamma_1", 0.0) * log_distance
    adjacency_discount = params["gamma_2"] * is_adjacent
    return_migration_cost = params["gamma_3"] * is_return_migration
    age_cost = params["gamma_4"] * age
    
    log_dest_population = np.log(np.maximum(region_data["常住人口万"][:n_choices][np.newaxis, :], 1.0))
    population_discount = params["gamma_5"] * log_dest_population
    
    migration_cost = is_moving * (
        fixed_cost
        + distance_cost
        - adjacency_discount
        + return_migration_cost # gamma_3 is likely negative, so this becomes a benefit
        + age_cost
        - population_discount
    )

    # --- 3. Total Utility ---
    # 基础效用组件
    total_utility = (
        income_utility
        + amenity_utility
        + home_premium
        - hukou_penalty
        - migration_cost
    )

    # 添加偏好匹配项 ξ_ij (论文696行)
    if xi_ij is not None:
        total_utility += xi_ij

    return np.clip(total_utility, -500, 500)

def calculate_flow_utility_individual_vectorized(
    state_data: Dict[str, np.ndarray],
    region_data: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    params: Dict[str, Any],
    agent_type: int,
    n_states: int,  # Now this is n_individual_states
    n_choices: int,
    visited_locations: list, # 新增：个体访问过的地点列表
    wage_predicted: Optional[np.ndarray] = None,
    wage_reference: Optional[np.ndarray] = None,
    eta_i: Optional[np.ndarray] = None,
    nu_ij: Optional[np.ndarray] = None,
    xi_ij: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculates flow utility for a SINGLE INDIVIDUAL with a compact state space.
    """
    # --- 1. Map compact state indices back to global indices ---
    # 'prev_provcd_idx' in state_data is now a compact index (e.g., 0, 1, 2...)
    compact_prev_loc_indices = state_data['prev_provcd_idx']
    
    # Use the visited_locations list to map back to global province IDs
    global_prev_loc_indices = np.array([visited_locations[i] for i in compact_prev_loc_indices])

    # --- 2. Prepare data arrays by broadcasting ---
    age = state_data['age'][:, np.newaxis]
    # Use the mapped global indices for calculations
    prev_loc_indices_b = global_prev_loc_indices[:, np.newaxis] 
    hukou_loc_indices = state_data['hukou_prov_idx'][:, np.newaxis]
    hometown_loc_indices = state_data['hometown_prov_idx'][:, np.newaxis]
    dest_loc_indices = np.arange(n_choices)[np.newaxis, :]

    # --- The rest of the logic is largely the same as the global version ---
    
    # Fallback income utility (as in the original function)
    if '地区基本经济面' in region_data:
        mu_jt = region_data['地区基本经济面'][:n_choices][np.newaxis, :]
        wage_approx = np.exp(mu_jt + 10.0)
    else:
        wage_approx = np.full((n_states, n_choices), 30000.0)
    income_utility = params.get("alpha_w", 1.0) * np.log(np.maximum(wage_approx, 1e-6))

    # Amenities Utility
    amenity_utility = (
        params["alpha_climate"] * region_data["amenity_climate"][:n_choices][np.newaxis, :]
        # ... (add other amenities as in the original function)
    )
    
    # Home Premium
    is_hometown = (dest_loc_indices == hometown_loc_indices)
    home_premium = params.get("alpha_home", 0.0) * is_hometown

    # Hukou Penalty
    is_hukou_mismatch = (dest_loc_indices != hukou_loc_indices)
    rho_base = params.get("rho_base_tier_1", 1.0) # Simplified for brevity
    hukou_penalty = is_hukou_mismatch * rho_base

    # Migration Cost
    is_moving = (dest_loc_indices != prev_loc_indices_b)
    
    distance = distance_matrix[prev_loc_indices_b, dest_loc_indices]
    log_distance = np.log(np.maximum(distance, 1.0))
    
    is_adjacent = adjacency_matrix[prev_loc_indices_b, dest_loc_indices]
    is_return_migration = (dest_loc_indices == hometown_loc_indices) & is_moving

    fixed_cost = params.get(f"gamma_0_type_{agent_type}", 0.0)
    distance_cost = params.get("gamma_1", 0.0) * log_distance
    adjacency_discount = params["gamma_2"] * is_adjacent
    return_migration_cost = params["gamma_3"] * is_return_migration
    age_cost = params["gamma_4"] * age
    
    log_dest_population = np.log(np.maximum(region_data["常住人口万"][:n_choices][np.newaxis, :], 1.0))
    population_discount = params["gamma_5"] * log_dest_population
    
    migration_cost = is_moving * (
        fixed_cost + distance_cost - adjacency_discount + 
        return_migration_cost + age_cost - population_discount
    )

    # Total Utility
    total_utility = (
        income_utility
        + amenity_utility
        + home_premium
        - hukou_penalty
        - migration_cost
    )

    if xi_ij is not None:
        total_utility += xi_ij

    return np.clip(total_utility, -500, 500)

# Keep the original function for compatibility if needed elsewhere, though it's now unused by Bellman
def calculate_flow_utility(*args, **kwargs):
    raise NotImplementedError("Use calculate_flow_utility_vectorized. The non-vectorized version is deprecated.")