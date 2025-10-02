"""
ABM代理类定义
"""
import numpy as np
from typing import Dict, List, Any


class Agent:
    """
    ABM中的个体代理类
    """
    def __init__(
        self,
        agent_id: int,
        initial_location: int,
        agent_type: int,
        params: Dict[str, float]
    ):
        self.agent_id = agent_id
        self.current_location = initial_location
        self.agent_type = agent_type
        self.params = params
        self.age = np.random.randint(18, 65)  # 随机初始年龄
        self.hukou_location = initial_location  # 户口所在地
        self.known_locations = {initial_location}  # 已知的地点集合
        
    def move_to(self, new_location: int):
        """
        移动到新位置
        """
        self.current_location = new_location
        self.known_locations.add(new_location)
        self.age += 1  # 年龄增长
        
    def make_location_choice(self, environment, period: int):
        """
        根据效用最大化原则选择下一个位置
        """
        n_regions = environment.n_regions
        utilities = np.zeros(n_regions)
        
        for j in range(n_regions):
            # 计算选择位置j的效用
            utilities[j] = self.calculate_utility(j, environment, period)
        
        # 根据logit模型选择位置（添加随机项）
        logit_probs = self.softmax(utilities)
        chosen_location = np.random.choice(n_regions, p=logit_probs)
        
        return chosen_location
    
    def calculate_utility(self, j: int, environment, period: int):
        """
        计算选择位置j的效用
        这里使用与结构估计中相似的效用函数形式
        """
        # 获取位置j的特征
        region_characteristics = environment.get_region_characteristics(j, period)
        
        # 1. 收入效用（使用预测工资）
        predicted_wage = region_characteristics.get('wage_predicted', 
                                                 region_characteristics.get('avg_wage', 50000))
        w_ref = self.get_reference_wage(environment)
        
        income_utility = self.calculate_income_utility(predicted_wage, w_ref)
        
        # 2. 地区舒适度
        amenity_utility = (
            self.params.get('alpha_climate', 0.1) * region_characteristics.get('amenity_climate', 0) +
            self.params.get('alpha_health', 0.1) * region_characteristics.get('amenity_health', 0) +
            self.params.get('alpha_education', 0.1) * region_characteristics.get('amenity_education', 0) +
            self.params.get('alpha_public_services', 0.1) * region_characteristics.get('amenity_public_services', 0)
        )
        
        # 3. 家乡溢价
        is_home = (j == self.hukou_location)
        home_premium = self.params.get('alpha_home', 1.0) * is_home
        
        # 4. 户口惩罚
        hukou_penalty = self.calculate_hukou_penalty(j, region_characteristics)
        
        # 5. 迁移成本（如果是移动）
        is_moving = (j != self.current_location)
        migration_cost = 0
        if is_moving:
            migration_cost = self.calculate_migration_cost(j, environment)
        
        # 6. 非经济偏好匹配（如果该地点已访问过）
        xi_ij = 0
        if j in self.known_locations:
            # 已知地点的偏好匹配值（随机初始化或从参数中获取）
            xi_ij = np.random.normal(0, 0.1)  # 简化处理
        
        # 总效用
        total_utility = income_utility + amenity_utility + home_premium - hukou_penalty - migration_cost + xi_ij
        
        return total_utility
    
    def calculate_income_utility(self, w_ij: float, w_ref: float) -> float:
        """
        计算收入效用（基于前景理论）
        """
        log_w_ij = np.log(max(w_ij, 1))  # 避免log(0)
        log_w_ref = np.log(max(w_ref, 1))
        
        alpha_w = self.params.get('alpha_w', 1.0)
        lambda_loss = self.params.get('lambda', 2.0)
        
        if log_w_ij >= log_w_ref:
            return alpha_w * (log_w_ij - log_w_ref)
        else:
            return alpha_w * lambda_loss * (log_w_ij - log_w_ref)
    
    def get_reference_wage(self, environment) -> float:
        """
        获取参考工资（可以是历史工资或其他参考值）
        """
        # 简化：使用当前所在地区的平均工资作为参考
        return environment.get_average_wage(self.current_location)
    
    def calculate_hukou_penalty(self, j: int, region_characteristics: Dict[str, Any]) -> float:
        """
        计算户口惩罚
        """
        if j == self.hukou_location:
            return 0.0  # 在户口地无惩罚
        
        # 基础惩罚，根据城市等级
        region_tier = region_characteristics.get('tier', 1)
        base_penalty = self.params.get(f'rho_base_tier_{region_tier}', 1.0)
        
        # 与公共服务的交互项
        edu_level = region_characteristics.get('amenity_education', 0)
        health_level = region_characteristics.get('amenity_health', 0)
        housing_price = region_characteristics.get('housing_price', 10000)
        
        edu_penalty = self.params.get('rho_edu', 0.1) * edu_level
        health_penalty = self.params.get('rho_health', 0.1) * health_level
        house_penalty = self.params.get('rho_house', 0.1) * (housing_price / 10000)  # 标准化
        
        total_penalty = base_penalty + edu_penalty + health_penalty + house_penalty
        return total_penalty
    
    def calculate_migration_cost(self, j: int, environment) -> float:
        """
        计算迁移成本
        """
        if j == self.current_location:
            return 0.0  # 未移动无成本
        
        # 获取距离
        distance = environment.get_distance(self.current_location, j)
        
        # 是否相邻
        is_adjacent = environment.is_adjacent(self.current_location, j)
        
        # 是否返回曾居住地（回流）
        is_return_move = j in self.known_locations
        
        # 类型特定的固定成本
        fixed_cost = self.params.get(f'gamma_0_type_{self.agent_type}', 1.0)
        
        # 其他成本组件
        distance_cost = self.params.get('gamma_1', -0.1) * (distance / 100)  # 标准化距离
        adjacency_discount = self.params.get('gamma_2', 0.5) * is_adjacent
        return_discount = self.params.get('gamma_3', -0.4) * is_return_move
        age_effect = self.params.get('gamma_4', 0.01) * self.age
        population_effect = self.params.get('gamma_5', -0.05) * environment.get_population(j) / 1000000  # 标准化人口
        
        total_cost = (
            fixed_cost +
            distance_cost -
            adjacency_discount -
            return_discount +
            age_effect +
            population_effect
        )
        
        return total_cost
    
    def softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Softmax函数，用于计算选择概率
        """
        x = x / temperature
        exp_x = np.exp(x - np.max(x))  # 数值稳定性
        return exp_x / np.sum(exp_x)