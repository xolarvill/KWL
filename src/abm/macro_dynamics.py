"""
ABM宏观动态模块
实现论文tex第1396-1420行的三个核心反馈循环
1. 工资动态方程 (eq:wage_dynamics_abm)
2. 房价动态方程 (eq:price_dynamics_abm) 
3. 公共服务动态方程 (eq:quality_dynamics_abm)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class RegionalState:
    """地区状态类"""
    region_id: int
    population: int = 1000000
    avg_wage: float = 50000.0
    housing_price: float = 15000.0
    public_services: float = 1.0  # 公共服务质量指数
    investment_base: float = 1.0  # 公共服务投资基数
    
    # 舒适度分项
    amenity_climate: float = 0.0
    amenity_health: float = 0.0
    amenity_education: float = 0.0
    amenity_public: float = 0.0


class MacroDynamics:
    """
    宏观经济动态模型
    捕捉微观决策到宏观环境的反馈效应
    """
    
    def __init__(self, n_regions: int, macro_params: Dict[str, float],
                 distance_matrix: np.ndarray = None,
                 adjacency_matrix: np.ndarray = None):
        """
        初始化宏观动态模型
        
        Args:
            n_regions: 地区数量（29个省份）
            macro_params: 宏观参数 {phi_w, phi_p, g_q}
            distance_matrix: 距离矩阵（如果None则随机生成）
            adjacency_matrix: 邻接矩阵（如果None则随机生成）
        """
        self.n_regions = n_regions
        self.macro_params = macro_params
        
        # 加载矩阵数据
        self.distance_matrix, self.adjacency_matrix = self._load_matrices(distance_matrix, adjacency_matrix)
        
        # 初始化地区状态
        self.regions = self._initialize_regions()
        
        # 外生增长率
        self.exogenous_wage_growth = 0.02  # g_t，全国平均工资增长率
        self.public_service_growth = macro_params.get('g_q', 0.02)  # g_q，公共服务投资增长率
    
    def _load_matrices(self, distance_matrix: np.ndarray = None, adjacency_matrix: np.ndarray = None) -> tuple:
        """加载距离和邻接矩阵（如果为None则随机生成）"""
        if distance_matrix is not None and adjacency_matrix is not None:
            return distance_matrix, adjacency_matrix
        
        print(f"  随机生成距离/邻接矩阵 ({self.n_regions}x{self.n_regions})...")
        
        # 随机生成距离矩阵
        dist = np.random.uniform(100, 2000, (self.n_regions, self.n_regions))
        np.fill_diagonal(dist, 0)
        
        # 随机生成邻接矩阵
        adj = (np.random.rand(self.n_regions, self.n_regions) > 0.75).astype(int)
        np.fill_diagonal(adj, 0)
        
        return dist, adj
        
    def _initialize_regions(self) -> List[RegionalState]:
        """初始化各地区状态（从数据加载）"""
        regions = []
        
        for region_id in range(self.n_regions):
            # 简化：使用随机初始值，实际应用中应从geo.xlsx加载
            region = RegionalState(
                region_id=region_id,
                population=np.random.randint(500000, 100000000),
                avg_wage=np.random.normal(60000, 15000),
                housing_price=np.random.normal(15000, 5000),
                public_services=np.random.normal(1.0, 0.2),
                investment_base=np.random.normal(1.0, 0.1),
                amenity_climate=np.random.normal(0, 1),
                amenity_health=np.random.normal(0, 1),
                amenity_education=np.random.normal(0, 1),
                amenity_public=np.random.normal(0, 1)
            )
            regions.append(region)
            
        return regions
    
    def update_regions(self, 
                      migration_decisions: Dict[int, int],
                      period: int) -> Dict[str, np.ndarray]:
        """
        根据迁移决策更新所有地区的宏观变量
        
        Args:
            migration_decisions: {agent_id: new_location} 迁移决策字典
            period: 当前时期
            
        Returns:
            更新后的宏观变量字典
        """
        # 计算净迁移流量
        net_flows = self._calculate_net_flows(migration_decisions)
        
        # 更新工资动态（tex方程1399）
        wage_changes = self._update_wages(net_flows)
        
        # 更新房价动态（tex方程1407）
        price_changes = self._update_housing_prices(net_flows)
        
        # 更新公共服务质量（tex方程1415）
        service_changes = self._update_public_services(net_flows, period)
        
        # 汇总变化
        updates = {
            'wage_changes': wage_changes,
            'price_changes': price_changes,
            'service_changes': service_changes,
            'net_flows': net_flows
        }
        
        return updates
    
    def _calculate_net_flows(self, migration_decisions: Dict[int, int]) -> np.ndarray:
        """
        计算各地区的净迁移流量
        
        Args:
            migration_decisions: {agent_id: new_location}
            
        Returns:
            net_flows: 净迁移流量数组 (n_regions,)
        """
        net_flows = np.zeros(self.n_regions)
        
        # 统计每个地区的净流入
        for new_location in migration_decisions.values():
            if 0 <= new_location < self.n_regions:
                net_flows[new_location] += 1
        
        # 减去净流出（需要知道原始位置）
        # 简化：假设所有agent都在移动，net_flows即为净流入
        
        return net_flows
    
    def _update_wages(self, net_flows: np.ndarray) -> np.ndarray:
        """
        工资动态方程（tex第1399行）
        Wage_{j,t+1} = Wage_{jt} * (1 + g_t) * (1 - φ_w * NetFlow_{jt} / Pop_{jt})
        
        Args:
            net_flows: 净迁移流量
            
        Returns:
            wage_changes: 工资变化率
        """
        phi_w = self.macro_params.get('phi_w', 0.08)  # 默认校准值
        
        wage_changes = np.zeros(self.n_regions)
        
        for j, region in enumerate(self.regions):
            # 避免除零
            if region.population > 0:
                # 计算相对净迁移率
                net_flow_rate = net_flows[j] / region.population
                
                # 工资动态方程（人口流入抑制工资增长）
                growth_factor = (1 + self.exogenous_wage_growth) * (1 - phi_w * net_flow_rate)
                
                # 更新工资
                old_wage = region.avg_wage
                region.avg_wage *= growth_factor
                
                # 记录变化率
                wage_changes[j] = (region.avg_wage - old_wage) / old_wage
        
        return wage_changes
    
    def _update_housing_prices(self, net_flows: np.ndarray) -> np.ndarray:
        """
        房价动态方程（tex第1407行）
        Price_{j,t+1} = Price_{jt} * (1 + φ_p * NetFlow_{jt} / Pop_{jt})
        
        Args:
            net_flows: 净迁移流量
            
        Returns:
            price_changes: 房价变化率
        """
        phi_p = self.macro_params.get('phi_p', 0.32)  # 默认校准值
        
        price_changes = np.zeros(self.n_regions)
        
        for j, region in enumerate(self.regions):
            # 避免除零
            if region.population > 0:
                # 计算相对净迁移率
                net_flow_rate = net_flows[j] / region.population
                
                # 房价动态方程（人口流入推高房价）
                growth_factor = 1 + phi_p * net_flow_rate
                
                # 更新房价
                old_price = region.housing_price
                region.housing_price *= growth_factor
                
                # 记录变化率
                price_changes[j] = (region.housing_price - old_price) / old_price
        
        return price_changes
    
    def _update_public_services(self, net_flows: np.ndarray, period: int) -> np.ndarray:
        """
        公共服务动态方程（tex第1415行）
        Q_{j,t+1} = Inv_j * (1 + g_q)^t / Pop_{j,t+1}
        
        Args:
            net_flows: 净迁移流量
            period: 当前时期
            
        Returns:
            service_changes: 公共服务质量变化
        """
        service_changes = np.zeros(self.n_regions)
        
        for j, region in enumerate(self.regions):
            # 更新人口（简化：假设净迁移直接改变人口）
            old_population = region.population
            region.population = max(1, int(region.population + net_flows[j]))  # 确保至少为1
            
            # 计算公共服务质量
            # Inv_j * (1 + g_q)^t / Pop_{t+1}
            investment_growth = (1 + self.public_service_growth) ** period
            region.public_services = (region.investment_base * investment_growth) / region.population
            
            # 记录变化率
            if old_population > 0:
                pop_change_rate = (region.population - old_population) / old_population
                service_changes[j] = pop_change_rate  # 简化：变化率与人口变化相关
        
        return service_changes
    
    def get_macro_variables(self) -> Dict[str, np.ndarray]:
        """获取当前宏观变量"""
        macro_vars = {
            'populations': np.array([r.population for r in self.regions]),
            'avg_wages': np.array([r.avg_wage for r in self.regions]),
            'housing_prices': np.array([r.housing_price for r in self.regions]),
            'public_services': np.array([r.public_services for r in self.regions]),
            'amenity_climate': np.array([r.amenity_climate for r in self.regions]),
            'amenity_health': np.array([r.amenity_health for r in self.regions]),
            'amenity_education': np.array([r.amenity_education for r in self.regions]),
            'amenity_public': np.array([r.amenity_public for r in self.regions])
        }
        
        return macro_vars
    
    def load_from_data(self, data_path: str):
        """
        从实际数据加载地区初始状态
        将从 geo.xlsx 加载真实数据
        """
        # TODO: 实现从geo.xlsx加载真实数据
        # 暂时使用随机初始化的简化版本
        pass


def demo_macro_dynamics():
    """演示宏观动态"""
    print("\n" + "="*60)
    print("ABM宏观动态演示")
    print("="*60)
    
    # 宏观参数（待校准）
    macro_params = {
        'phi_w': 0.08,  # 工资敏感度
        'phi_p': 0.32,  # 房价弹性
        'g_q': 0.02     # 公共服务增长率
    }
    
    # 初始化宏观动态模型
    n_regions = 29  # 29个省份
    macro_dynamics = MacroDynamics(n_regions, macro_params)
    
    # 获取初始状态
    initial_state = macro_dynamics.get_macro_variables()
    print(f"\n初始状态（前5个省份）:")
    print(f"  人口: {initial_state['populations'][:5]}")
    print(f"  工资: {initial_state['avg_wages'][:5]}")
    print(f"  房价: {initial_state['housing_prices'][:5]}")
    
    # 模拟迁移决策（示例：1000人从各省流向北京）
    print(f"\n模拟迁移: 1000人迁移到北京（省份0）...")
    migration_decisions = {}
    for i in range(1000):
        migration_decisions[i] = 0  # 都迁往省份0
    
    # 更新宏观变量
    updates = macro_dynamics.update_regions(migration_decisions, period=1)
    
    # 查看变化
    print(f"\n变化率:")
    print(f"  工资变化: {updates['wage_changes'][:5]}")
    print(f"  房价变化: {updates['price_changes'][:5]}")
    print(f"  公共服务变化: {updates['service_changes'][:5]}")
    
    # 获取更新后的状态
    updated_state = macro_dynamics.get_macro_variables()
    print(f"\n更新后状态（前5个省份）:")
    print(f"  人口: {updated_state['populations'][:5]}")
    print(f"  工资: {updated_state['avg_wages'][:5]}")
    print(f"  房价: {updated_state['housing_prices'][:5]}")
    
    return macro_dynamics


if __name__ == '__main__':
    macro_dynamics = demo_macro_dynamics()