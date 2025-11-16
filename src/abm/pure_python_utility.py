"""
纯 Python 实现的效用函数（ABM专用）
不使用 JIT，避免与 EM 的 JIT 冲突
完全复用 estimation 模块的数学形式
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class PurePythonUtility:
    """纯 Python 效用计算（无 JIT）"""
    
    def __init__(self, 
                 distance_matrix: np.ndarray,
                 adjacency_matrix: np.ndarray,
                 region_data: pd.DataFrame,
                 n_regions: int):
        self.distance_matrix = distance_matrix
        self.adjacency_matrix = adjacency_matrix
        self.region_data = region_data
        self.n_regions = n_regions
        
        # 预处理地区数据（如果region_data为None，创建随机数据）
        if self.region_data is None or len(self.region_data) == 0:
            self._create_random_data()
        
        self.region_dict = self._prepare_region_dict()
    
    def _prepare_region_dict(self) -> Dict[str, np.ndarray]:
        """准备地区特征字典"""
        data = {}
        required_cols = ['amenity_climate', 'amenity_health', 'amenity_education',
                        'amenity_public_services', 'amenity_hazard', '房价收入比', '常住人口万']
        
        for col in required_cols:
            if col in self.region_data.columns:
                data[col] = self.region_data[col].values[:self.n_regions]
            else:
                if col == '房价收入比':
                    data[col] = np.ones(self.n_regions) * 10.0
                elif col == '常住人口万':
                    data[col] = np.ones(self.n_regions) * 500.0
                else:
                    data[col] = np.zeros(self.n_regions)
        
        # 城市分级
        if '户籍获取难度' in self.region_data.columns:
            data['户籍获取难度'] = self.region_data['户籍获取难度'].values[:self.n_regions]
        else:
            tiers = np.ones(self.n_regions, dtype=int) * 3
            if self.n_regions > 8:
                tiers[:8] = 2
            if self.n_regions > 3:
                tiers[:3] = 1
            data['户籍获取难度'] = tiers
        
        return data
    
    def calculate_utility(self,
                         agent_data: Dict[str, Any],
                         params: Dict[str, float],
                         eta_i: float = 0.0,
                         nu_ij: Optional[np.ndarray] = None,
                         xi_ij: Optional[np.ndarray] = None) -> np.ndarray:
        """为单个Agent计算所有地区的效用"""
        n_regions = self.n_regions
        utilities = np.zeros(n_regions)
        
        # 提取Agent信息
        age = agent_data['age']
        current_loc = agent_data['current_location']
        hukou_loc = agent_data['hukou_location']
        agent_type = agent_data.get('agent_type', 0)
        
        # 如果提供了nu和xi，确保是正确形状
        if nu_ij is None:
            nu_ij = np.zeros(n_regions)
        if xi_ij is None:
            xi_ij = np.zeros(n_regions)
        
        # 为每个地区计算效用
        for j in range(n_regions):
            utilities[j] = self._calculate_single_utility(
                j, age, current_loc, hukou_loc, agent_type, params, eta_i, 
                nu_ij[j], xi_ij[j]
            )
        
        return utilities
    
    def _calculate_single_utility(self,
                                region_j: int,
                                age: float,
                                current_loc: int,
                                hukou_loc: int,
                                agent_type: int,
                                params: Dict[str, float],
                                eta_i: float,
                                nu_ij: float,
                                xi_ij: float) -> float:
        """计算单个地区的效用"""
        
        # ========== 1. 收入效用（对数形式） ==========
        alpha_w = params.get('alpha_w', 1.0)
        # 使用地区常住人口作为工资的代理变量（实际应使用真实工资数据）
        population = self.region_dict['常住人口万'][region_j]
        wage_proxy = np.exp(np.log(population) + 10)  # 模拟工资
        income_utility = alpha_w * np.log(max(wage_proxy, 1.0))
        
        # ========== 2. 地区舒适度 ==========
        amenity_utility = 0.0
        amenity_coeffs = {
            'amenity_climate': 'alpha_climate',
            'amenity_health': 'alpha_health',
            'amenity_education': 'alpha_education',
            'amenity_public_services': 'alpha_public_services',
            'amenity_hazard': 'alpha_hazard'
        }
        
        for amenity_key, param_key in amenity_coeffs.items():
            alpha = params.get(param_key, 0.1)
            amenity_val = self.region_dict[amenity_key][region_j]
            amenity_utility += alpha * amenity_val
        
        # ========== 3. 家乡溢价 ==========
        alpha_home = params.get('alpha_home', 0.0)
        is_home = (region_j == hukou_loc)
        home_premium = alpha_home * is_home
        
        # ========== 4. 户籍惩罚 ==========
        hukou_penalty = 0.0
        if region_j != hukou_loc:
            # 城市分级
            tier_difficulty = self.region_dict['户籍获取难度'][region_j]
            if tier_difficulty == 1:
                rho_base = params.get('rho_base_tier_3', 0.2)  # 容易
            elif tier_difficulty == 2:
                rho_base = params.get('rho_base_tier_2', 0.5)  # 中等
            else:
                rho_base = params.get('rho_base_tier_1', 1.0)  # 难
            
            # 交互项
            rho_edu = params.get('rho_edu', 0.0) * self.region_dict['amenity_education'][region_j]
            rho_health = params.get('rho_health', 0.0) * self.region_dict['amenity_health'][region_j]
            rho_house = params.get('rho_house', 0.0) * (self.region_dict['房价收入比'][region_j] / 10.0)  # 标准化
            
            hukou_penalty = rho_base + rho_edu + rho_health + rho_house
        
        # ========== 5. 迁移成本 ==========
        migration_cost = 0.0
        if region_j != current_loc:
            # 固定成本（类型特定）
            gamma_0 = params.get(f'gamma_0_type_{agent_type}', 0.0)
            
            # 距离成本
            distance = self.distance_matrix[current_loc, region_j]
            gamma_1 = params.get('gamma_1', 0.0) * np.log(max(distance, 1.0))
            
            # 邻接性折扣
            is_adjacent = self.adjacency_matrix[current_loc, region_j]
            gamma_2 = -params.get('gamma_2', 0.0) * is_adjacent  # 负数表示折扣
            
            # 回流迁移优惠
            is_return = (region_j == hukou_loc)
            gamma_3 = -params.get('gamma_3', 0.0) * is_return  # 负数表示优惠
            
            # 年龄效应
            gamma_4 = params.get('gamma_4', 0.0) * age
            
            # 人口规模效应
            log_pop = np.log(max(self.region_dict['常住人口万'][region_j], 1.0))
            gamma_5 = -params.get('gamma_5', 0.0) * log_pop  # 大目的地成本更低
            
            migration_cost = gamma_0 + gamma_1 + gamma_2 + gamma_3 + gamma_4 + gamma_5
        
        # ========== 6. 个体效应和偏好匹配 ==========
        individual_effect = eta_i  # 个体固定效应
        preference_match = xi_ij    # 偏好匹配
        
        # 收入匹配（用于未来扩展）
        # wage_match = nu_ij
        
        # ========== 总效用 ==========
        total_utility = (income_utility + 
                        amenity_utility + 
                        home_premium - 
                        hukou_penalty - 
                        migration_cost + 
                        individual_effect + 
                        preference_match)
        
        # 限制效用范围（防止数值溢出）
        return np.clip(total_utility, -500, 500)
    
    def get_region_characteristics(self, region_id: int) -> Dict[str, float]:
        """获取地区特征"""
        return {k: v[region_id] for k, v in self.region_dict.items()}
    
    def load_data_from_files(self,
                           distance_path: str = "data/processed/distance_matrix.xlsx",
                           adjacency_path: str = "data/processed/adjacent_matrix.xlsx",
                           region_path: str = "data/processed/geo.xlsx"):
        """从文件加载数据"""
        try:
            # 加载距离矩阵
            df_dist = pd.read_excel(distance_path, index_col=0)
            self.distance_matrix = df_dist.values
            
            # 加载邻接矩阵
            df_adj = pd.read_excel(adjacency_path, index_col=0)
            self.adjacency_matrix = df_adj.values.astype(int)
            
            # 加载地区数据
            self.region_data = pd.read_excel(region_path)
            self.region_dict = self._prepare_region_dict()
            
            print(f"数据加载成功: {self.n_regions}个地区")
            
        except Exception as e:
            print(f"数据加载错误: {e}")
            print("使用随机生成数据")
            self._create_random_data()
    
    def _create_random_data(self):
        """创建随机数据（用于测试）"""
        np.random.seed(42)
        
        self.distance_matrix = np.random.uniform(100, 2000, (self.n_regions, self.n_regions))
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.adjacency_matrix = (np.random.rand(self.n_regions, self.n_regions) > 0.7).astype(int)
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        region_df = pd.DataFrame({
            'amenity_climate': np.random.normal(0, 1, self.n_regions),
            'amenity_health': np.random.normal(0, 1, self.n_regions),
            'amenity_education': np.random.normal(0, 1, self.n_regions),
            'amenity_public_services': np.random.normal(0, 1, self.n_regions),
            'amenity_hazard': np.random.normal(0, 1, self.n_regions),
            '房价收入比': np.random.uniform(5, 30, self.n_regions),
            '常住人口万': np.random.uniform(100, 5000, self.n_regions),
            '户籍获取难度': np.random.randint(1, 4, self.n_regions)
        })
        
        self.region_data = region_df
        self.region_dict = self._prepare_region_dict()


def demo_python_utility():
    """演示纯Python效用函数"""
    print("\n" + "="*80)
    print("纯Python效用函数演示（无JIT）")
    print("="*80)
    
    n_regions = 10
    
    # 创建Utility
    utility = PurePythonUtility(
        distance_matrix=np.random.uniform(100, 2000, (n_regions, n_regions)),
        adjacency_matrix=(np.random.rand(n_regions, n_regions) > 0.8).astype(int),
        region_data=PurePythonUtility(n_regions, None, None, n_regions)._create_random_data(),
        n_regions=n_regions
    )
    
    # Agent数据
    agent_data = {
        'age': 28,
        'current_location': 0,
        'hukou_location': 0,
        'agent_type': 0,
        'education': 16  # 大学本科
    }
    
    # 参数
    params = {
        'alpha_w': 1.0, 'alpha_home': 1.0, 'alpha_climate': 0.1, 'alpha_health': 0.1,
        'alpha_education': 0.1, 'alpha_public_services': 0.1, 'alpha_hazard': 0.1,
        'rho_base_tier_1': 2.0, 'rho_base_tier_2': 1.0, 'rho_base_tier_3': 0.5,
        'rho_edu': 0.3, 'rho_health': 0.2, 'rho_house': 0.4,
        'gamma_0_type_0': 0.5, 'gamma_0_type_1': 1.5, 'gamma_0_type_2': 2.0,
        'gamma_1': -0.15, 'gamma_2': 0.3, 'gamma_3': -0.35, 'gamma_4': 0.02, 'gamma_5': -0.1
    }
    
    # 计算效用
    utilities = utility.calculate_utility(
        agent_data=agent_data,
        params=params,
        eta_i=0.5,
        nu_ij=None,
        xi_ij=None
    )
    
    print(f"Agent特征:")
    print(f"  年龄: {agent_data['age']}岁")
    print(f"  教育: {agent_data['education']}年")
    print(f"  户籍: 省份{agent_data['hukou_location']}")
    
    print(f"\n各地区效用值（前5个）:")
    for i in range(min(5, len(utilities))):
        print(f"  省份{i}: {utilities[i]:.3f}")
    
    print(f"\n效用统计:")
    print(f"  最高效用: {utilities.max():.3f} (省份{np.argmax(utilities)})")
    print(f"  最低效用: {utilities.min():.3f} (省份{np.argmin(utilities)})")
    
    # Softmax选择
    probs = np.exp(utilities - utilities.max()) / np.sum(np.exp(utilities - utilities.max()))
    chosen = np.random.choice(n_regions, p=probs)
    print(f"\n选择结果: 省份{chosen} (概率={probs[chosen]:.3f})")
    
    return utility, utilities


if __name__ == '__main__':
    demo_python_utility()