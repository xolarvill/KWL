"""
效用函数复用包装器
将estimation模块的vectorized效用函数适配到ABM的单个Agent计算
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import sys
import os

# 添加路径以导入estimation模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.utility import calculate_flow_utility_vectorized


class UtilityReuser:
    """
    效用函数复用包装器
    利用estimation模块已经优化过的效用计算
    """
    
    def __init__(self, 
                 distance_matrix: np.ndarray,
                 adjacency_matrix: np.ndarray,
                 region_data: pd.DataFrame,
                 n_regions: int):
        """
        初始化
        
        Args:
            distance_matrix: 距离矩阵 (n_regions, n_regions)
            adjacency_matrix: 邻接矩阵 (n_regions, n_regions)
            region_data: 地区特征数据
            n_regions: 地区数量
        """
        self.distance_matrix = distance_matrix
        self.adjacency_matrix = adjacency_matrix
        self.region_data = region_data
        self.n_regions = n_regions
        
        # 预处理地区数据字典（用于vectorized计算）
        self.region_data_dict = self._prepare_region_data_dict()
    
    def _prepare_region_data_dict(self) -> Dict[str, np.ndarray]:
        """将region_data转换为dict格式"""
        data_dict = {}
        
        # 确保所有需要的列都存在
        required_cols = ['amenity_climate', 'amenity_health', 'amenity_education', 
                        'amenity_public_services', 'amenity_hazard', '房价收入比', '常住人口万']
        
        for col in required_cols:
            if col in self.region_data.columns:
                data_dict[col] = self.region_data[col].values[:self.n_regions]
            else:
                # 创建默认值
                if col == '房价收入比':
                    data_dict[col] = np.ones(self.n_regions) * 10.0  # 10倍
                elif col == '常住人口万':
                    data_dict[col] = np.ones(self.n_regions) * 500.0  # 500万人
                else:
                    data_dict[col] = np.zeros(self.n_regions)
        
        # 城市分级（用于户籍惩罚）
        if '户籍获取难度' in self.region_data.columns:
            data_dict['户籍获取难度'] = self.region_data['户籍获取难度'].values[:self.n_regions]
        elif 'city_tier' in self.region_data.columns:
            data_dict['city_tier'] = self.region_data['city_tier'].values[:self.n_regions]
        else:
            # 默认：前3个是一线城市，接下来5个是二线，其余三线
            tiers = np.ones(self.n_regions, dtype=int) * 3
            if self.n_regions > 8:
                tiers[:8] = 2
            if self.n_regions > 3:
                tiers[:3] = 1
            data_dict['户籍获取难度'] = tiers
        
        return data_dict
    
    def calculate_single_agent_utility(self,
                                     agent_data: Dict[str, Any],
                                     params: Dict[str, float],
                                     eta_i: Optional[float] = 0.0,
                                     nu_ij: Optional[np.ndarray] = None,
                                     xi_ij: Optional[np.ndarray] = None) -> np.ndarray:
        """
        为单个Agent计算所有地区的效用
        
        Args:
            agent_data: Agent数据字典
            params: 结构参数
            eta_i: 个体固定效应
            nu_ij: 个体-地区收入匹配
            xi_ij: 偏好匹配
            
        Returns:
            utilities: 效用数组 (n_regions,)
        """
        # 构建state_data（vectorized计算需要的输入格式）
        age = agent_data['age']
        prev_loc = agent_data['current_location']
        hukou_loc = agent_data['hukou_location']
        hometown_loc = agent_data.get('hometown_location', hukou_loc)  # 默认家乡=户籍地
        
        # Create arrays with correct shape for JIT compatibility
        # JIT函数期望2D数组，但单个Agent时可以简化为不使用JIT
        state_data_numba = {
            'age': np.array([[age]], dtype=np.float64),  # (1, 1)
            'prev_provcd_idx': np.array([[prev_loc]], dtype=np.int32),  # (1, 1)
            'hukou_prov_idx': np.array([[hukou_loc]], dtype=np.int32),  # (1, 1)
            'hometown_prov_idx': np.array([[hometown_loc]], dtype=np.int32),  # (1, 1)
        }
        
        # Agent类型
        agent_type = agent_data.get('agent_type', 0)
        
        # 扩展eta为(1, 1)形状
        adjusted_eta = np.array([[eta_i]], dtype=np.float64)
        
        # 扩展nu为(1, n_regions)形状
        if nu_ij is None:
            nu_ij_2d = np.zeros((1, self.n_regions), dtype=np.float64)
        else:
            nu_ij_2d = nu_ij.reshape(1, -1).astype(np.float64)
        
        # 扩展xi为(1, n_regions)形状
        if xi_ij is None:
            xi_ij_2d = np.zeros((1, self.n_regions), dtype=np.float64)
        else:
            xi_ij_2d = xi_ij.reshape(1, -1).astype(np.float64)
        
        # 调用vectorized效用函数（复用estimation模块）
        # Note: 由于JIT限制，直接调用可能会出错，这里提供两种备用方案
        try:
            utilities_matrix = calculate_flow_utility_vectorized(
                state_data=state_data_numba,
                region_data=self.region_data_dict,
                distance_matrix=self.distance_matrix.astype(np.float64),
                adjacency_matrix=self.adjacency_matrix.astype(np.int32),
                params=params,
                agent_type=agent_type,
                n_states=1,  # 单个Agent
                n_choices=self.n_regions,
                wage_predicted=None,  # 使用地区基础工资
                wage_reference=None,
                eta_i=adjusted_eta,
                nu_ij=nu_ij_2d,
                xi_ij=xi_ij_2d
            )
            # 返回形状为(n_regions,)的数组
            return utilities_matrix.flatten()
        except Exception as e:
            print(f"JIT编译错误，使用备用实现: {e}")
            return self._calculate_utility_fallback(
                agent_data, params, eta_i, nu_ij, xi_ij
            )
    
    def _calculate_utility_fallback(self,
                                  agent_data: Dict[str, Any],
                                  params: Dict[str, float],
                                  eta_i: float,
                                  nu_ij: Optional[np.ndarray],
                                  xi_ij: Optional[np.ndarray]) -> np.ndarray:
        """
        备用实现：不使用JIT，直接计算效用
        避免numba的兼容性问题
        """
        utilities = np.zeros(self.n_regions)
        
        # 从agent_data中提取信息
        current_loc = agent_data['current_location']
        hukou_loc = agent_data['hukou_location']
        age = agent_data['age']
        agent_type = agent_data.get('agent_type', 0)
        
        for j in range(self.n_regions):
            # 调用我们之前实现的完整Agent逻辑
            # 这个会在_complete_agent.py中实现
            utilities[j] = self._manual_utility_calculation(
                j, current_loc, hukou_loc, age, agent_type, params, eta_i
            )
        
        return utilities
    
    def _manual_utility_calculation(self,
                                  region_j: int,
                                  current_loc: int,
                                  hukou_loc: int,
                                  age: int,
                                  agent_type: int,
                                  params: Dict[str, float],
                                  eta_i: float) -> float:
        """手动计算单个地区的效用（不使用numba）"""
        # 这是一个简化的版本，完整的需要根据我们的Agent逻辑实现
        n_regions = self.n_regions
        
        # ===== 各组件效用 =====
        # 1. 收入效用
        alpha_w = params.get('alpha_w', 1.0)
        income_utility = alpha_w * np.log(self.region_data_dict['常住人口万'][region_j] + 1)
        
        # 2. 舒适度
        amenity_utility = 0
        for amenity in ['amenity_climate', 'amenity_health', 'amenity_education', 'amenity_public_services']:
            alpha_key = f"alpha_{amenity.replace('amenity_', '')}"
            alpha_val = params.get(alpha_key, 0.1)
            amenity_utility += alpha_val * self.region_data_dict.get(amenity, [0]*n_regions)[region_j]
        
        # 3. 家乡溢价
        alpha_home = params.get('alpha_home', 0.0)
        hometown_utility = alpha_home * (region_j == hukou_loc)
        
        # 4. 户籍惩罚
        hukou_penalty = 0
        if region_j != hukou_loc:
            # 简化的城市分级
            tier = 1 if region_j < 3 else (2 if region_j < 8 else 3)
            rho_base = params.get(f'rho_base_tier_{tier}', 1.0)
            
            rho_edu = params.get('rho_edu', 0.0) * self.region_data_dict.get('amenity_education', [0]*n_regions)[region_j]
            rho_health = params.get('rho_health', 0.0) * self.region_data_dict.get('amenity_health', [0]*n_regions)[region_j]
            rho_house = params.get('rho_house', 0.0) * (self.region_data_dict.get('房价收入比', [10]*n_regions)[region_j] / 10)
            
            hukou_penalty = rho_base + rho_edu + rho_health + rho_house
        
        # 5. 迁移成本
        migration_cost = 0
        if region_j != current_loc:
            gamma_0 = params.get(f'gamma_0_type_{agent_type}', 0.0)
            distance = self.distance_matrix[current_loc, region_j] if hasattr(self, 'distance_matrix') else 500
            
            gamma_1 = params.get('gamma_1', 0.0) * np.log(max(distance, 1))
            gamma_2 = -params.get('gamma_2', 0.0) * self.adjacency_matrix[current_loc, region_j] if hasattr(self, 'adjacency_matrix') else 0
            gamma_3 = params.get('gamma_3', 0.0) * (region_j == hukou_loc)  # 回流
            gamma_4 = params.get('gamma_4', 0.0) * age
            gamma_5 = -params.get('gamma_5', 0.0) * np.log(max(self.region_data_dict.get('常住人口万', [100]*n_regions)[region_j], 1))
            
            migration_cost = gamma_0 + gamma_1 + gamma_2 + gamma_3 + gamma_4 + gamma_5
        
        # 6. 个体固定效应
        ind_effect = eta_i
        
        # 总效用
        total_utility = (income_utility + amenity_utility + hometown_utility - 
                        hukou_penalty - migration_cost + ind_effect)
        
        return total_utility
    
    def load_distance_matrix(self, data_path: str = "data/processed/distance_matrix.xlsx") -> np.ndarray:
        """加载距离矩阵"""
        try:
            df = pd.read_excel(data_path, index_col=0)
            return df.values
        except FileNotFoundError:
            print(f"距离矩阵文件未找到: {data_path}")
            print("使用随机距离矩阵作为占位符")
            return self._create_random_matrix()
    
    def load_adjacency_matrix(self, data_path: str = "data/processed/adjacent_matrix.xlsx") -> np.ndarray:
        """加载邻接矩阵"""
        try:
            df = pd.read_excel(data_path, index_col=0)
            return df.values.astype(int)
        except FileNotFoundError:
            print(f"邻接矩阵文件未找到: {data_path}")
            print("使用随机邻接矩阵作为占位符")
            return self._create_random_adjacency()
    
    def load_region_data(self, data_path: str = "data/processed/geo.xlsx") -> pd.DataFrame:
        """加载地区特征数据"""
        try:
            df = pd.read_excel(data_path)
            return df
        except FileNotFoundError:
            print(f"地区数据文件未找到: {data_path}")
            print("使用随机生成的地区数据作为占位符")
            return self._create_random_region_data()
    
    def _create_random_matrix(self) -> np.ndarray:
        """创建随机距离矩阵"""
        np.random.seed(42)
        matrix = np.random.uniform(100, 2000, (self.n_regions, self.n_regions))
        np.fill_diagonal(matrix, 0)  # 对角线为0
        return matrix
    
    def _create_random_adjacency(self) -> np.ndarray:
        """创建随机邻接矩阵"""
        np.random.seed(42)
        matrix = (np.random.rand(self.n_regions, self.n_regions) > 0.7).astype(int)
        np.fill_diagonal(matrix, 0)  # 对角线为0
        return matrix
    
    def _create_random_region_data(self) -> pd.DataFrame:
        """创建随机的地区特征数据"""
        np.random.seed(42)
        
        data = {
            'amenity_climate': np.random.normal(0, 1, self.n_regions),
            'amenity_health': np.random.normal(0, 1, self.n_regions),
            'amenity_education': np.random.normal(0, 1, self.n_regions),
            'amenity_public_services': np.random.normal(0, 1, self.n_regions),
            'amenity_hazard': np.random.normal(0, 1, self.n_regions),
            '房价收入比': np.random.uniform(5, 30, self.n_regions),  # 5-30倍
            '常住人口万': np.random.uniform(100, 5000, self.n_regions),
            '户籍获取难度': np.random.randint(1, 4, self.n_regions)  # 1=容易, 2=中等, 3=困难
        }
        
        return pd.DataFrame(data)
    
    def get_region_characteristics(self, region_id: int) -> Dict[str, float]:
        """获取单个地区的特征（用于Agent.decision）"""
        return {k: v[region_id] for k, v in self.region_data_dict.items()}


def demo_utility_reuse():
    """演示效用函数复用"""
    print("\n" + "="*80)
    print("效用函数复用演示")
    print("="*80)
    
    n_regions = 10
    
    # 为region_data创建临时reuser
    temp_reuser = UtilityReuser(
        distance_matrix=np.random.uniform(100, 2000, (n_regions, n_regions)),
        adjacency_matrix=(np.random.rand(n_regions, n_regions) > 0.8).astype(int),
        region_data=pd.DataFrame(),  # 临时用空DataFrame
        n_regions=n_regions
    )
    
    # 创建真实的region_data
    region_df = temp_reuser._create_random_region_data()
    
    # 创建包装器
    reuser = UtilityReuser(
        distance_matrix=np.random.uniform(100, 2000, (n_regions, n_regions)),
        adjacency_matrix=(np.random.rand(n_regions, n_regions) > 0.8).astype(int),
        region_data=region_df,
        n_regions=n_regions
    )
    
    # Agent参数
    agent_data = {
        'age': 28,
        'current_location': 0,
        'hukou_location': 0,
        'hometown_location': 0,
        'agent_type': 0
    }
    
    # 模型参数
    params = {
        'alpha_w': 1.0,
        'alpha_home': 1.0,
        'alpha_climate': 0.1,
        'alpha_health': 0.1,
        'alpha_education': 0.1,
        'alpha_public_services': 0.1,
        'alpha_hazard': 0.1,
        'rho_base_tier_1': 2.0,
        'rho_base_tier_2': 1.0,
        'rho_base_tier_3': 0.5,
        'rho_edu': 0.3,
        'rho_health': 0.2,
        'rho_house': 0.4,
        'gamma_0_type_0': 0.5,
        'gamma_1': -0.15,
        'gamma_2': 0.3,
        'gamma_3': -0.35,
        'gamma_4': 0.02,
        'gamma_5': -0.1
    }
    
    # 计算效-utilities
    utilities = reuser.calculate_single_agent_utility(
        agent_data=agent_data,
        params=params,
        eta_i=0.5,
        nu_ij=None,
        xi_ij=None
    )
    
    print(f"Agent特征:")
    print(f"  年龄: {agent_data['age']}岁")
    print(f"  类型: {agent_data['agent_type']} (机会型)")
    print(f"  当前地: 省份{agent_data['current_location']}")
    print(f"  户籍地: 省份{agent_data['hukou_location']}")
    
    print(f"\n各地区效用值（前5个）:")
    for i in range(min(5, len(utilities))):
        print(f"  省份{i}: {utilities[i]:.3f}")
    
    print(f"\n效用统计:")
    print(f"  最高效用: {utilities.max():.3f} (省份{np.argmax(utilities)})")
    print(f"  最低效用: {utilities.min():.3f} (省份{np.argmin(utilities)})")
    print(f"  家乡效用: {utilities[agent_data['hukou_location']]:.3f}")
    
    return reuser, utilities


if __name__ == '__main__':
    demo_utility_reuse()