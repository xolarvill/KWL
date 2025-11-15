"""
ABM Agent决策模块（轻量版）
复用estimation的效用函数，只保留Agent接口
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field
from src.abm.reuse_utility_wrapper import UtilityReuser


@dataclass
class AgentState:
    """Agent状态数据类"""
    agent_id: int
    agent_type: int
    age: int
    education: float
    hukou_location: int
    current_location: int
    eta_i: float
    
    # 动态状态
    visited_locations: Set[int] = field(default_factory=set)
    migration_count: int = 0
    is_hukou_local: bool = False
    
    def __post_init__(self):
        self.visited_locations.add(self.current_location)


class ABMAgent:
    """
    ABM Agent类 - 复用estimation效用函数
    """
    
    def __init__(self,
                 agent_id: int,
                 agent_type: int,
                 age: int,
                 education: float,
                 hukou_location: int,
                 current_location: int,
                 eta_i: float):
        """
        初始化Agent
        
        Args:
            agent_id: Agent唯一ID
            agent_type: 类型（0=机会型,1=稳定型,2=适应型）
            age: 年龄
            education: 教育年限
            hukou_location: 户籍省份索引
            current_location: 当前省份索引
            eta_i: 个体固定效应
        """
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            age=age,
            education=education,
            hukou_location=hukou_location,
            current_location=current_location,
            eta_i=eta_i
        )
        
        # Agent累计变量
        self.cumulative_utility: float = 0.0
        self.lifetime_migrations: int = 0
        
    def make_decision(self,
                     utility_reuser: UtilityReuser,
                     params: Dict[str, float],
                     macro_state: Dict[str, np.ndarray],
                     period: int) -> int:
        """
        Agent做出迁移决策
        
        Args:
            utility_reuser: 效用函数包装器
            params: 结构参数
            macro_state: 宏观状态
            period: 当前时期
            
        Returns:
            chosen_location: 选择的省份索引
        """
        # 准备Agent数据
        agent_data = self._prepare_agent_data()
        
        # 计算所有地区的效用（复用estimation模块）
        utilities = utility_reuser.calculate_single_agent_utility(
            agent_data=agent_data,
            params=params,
            eta_i=self.state.eta_i,
            nu_ij=None,  # 可扩展：个体-地区匹配
            xi_ij=None   # 可扩展：偏好匹配
        )
        
        # 保管当前位置的效用（用于记录）
        current_utility = utilities[self.state.current_location]
        self.cumulative_utility += current_utility
        
        # Softmax选择（多项Logit）
        probs = self._softmax(utilities, temperature=params.get('logit_scale', 1.0))
        chosen_location = np.random.choice(self.n_regions, p=probs)
        
        return chosen_location
    
    def update_state(self,
                    new_location: int,
                    new_wage: float,
                    period: int):
        """
        更新Agent状态（迁移完成后调用）
        
        Args:
            new_location: 新位置
            new_wage: 新位置工资
            period: 当前时期
        """
        old_location = self.state.current_location
        
        # 更新位置
        self.state.current_location = new_location
        self.state.visited_locations.add(new_location)
        
        # 年龄增长（迁移决策代表1年）
        self.state.age += 1
        
        # 统计迁移
        if new_location != old_location:
            self.state.migration_count += 1
            self.lifetime_migrations += 1
    
    def _prepare_agent_data(self) -> Dict[str, Any]:
        """准备Agent数据（供效用函数使用）"""
        return {
            'age': self.state.age,
            'current_location': self.state.current_location,
            'hukou_location': self.state.hukou_location,
            'hometown_location': self.state.hukou_location,  # 简化：家乡=户籍地
            'agent_type': self.state.agent_type,
            'education': self.state.education
        }
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax概率计算"""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))  # 数值稳定性
        return exp_x / np.sum(exp_x)
    
    def get_attribute(self, attr_name: str):
        """获取Agent属性"""
        return getattr(self.state, attr_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取Agent总结信息"""
        return {
            'agent_id': self.state.agent_id,
            'type': self.state.agent_type,
            'age': self.state.age,
            'education': self.state.education,
            'hukou_location': self.state.hukou_location,
            'current_location': self.state.current_location,
            'migration_count': self.state.migration_count,
            'visited_count': len(self.state.visited_locations),
            'lifetime_migrations': self.lifetime_migrations,
            'cumulative_utility': self.cumulative_utility
        }


class PopulationManager:
    """Agent群体管理器"""
    
    def __init__(self, population_df: pd.DataFrame, params: Dict[str, float]):
        """
        从DataFrame创建Agent群体
        
        Args:
            population_df: 合成人口DataFrame
            params: 结构参数
        """
        self.agents: List[ABMAgent] = []
        self.params = params
        self.n_agents = len(population_df)
        self.n_regions = params.get('n_choices', 29)
        
        # 创建所有Agent
        self._create_agents(population_df)
    
    def _create_agents(self, population_df: pd.DataFrame):
        """从DataFrame创建Agent"""
        for idx, row in population_df.iterrows():
            agent = ABMAgent(
                agent_id=row['agent_id'],
                agent_type=row['agent_type'],
                age=int(row['age']),
                education=float(row['education']),
                hukou_location=int(row['hukou_location']),
                current_location=int(row['current_location']),
                eta_i=float(row.get('eta_i', 0.0))
            )
            self.agents.append(agent)
    
    def simulate_period(self,
                       utility_reuser: UtilityReuser,
                       macro_state: Dict[str, np.ndarray],
                       period: int) -> Dict[str, Any]:
        """
        模拟一个完整时期
        
        Args:
            utility_reuser: 效用函数
            macro_state: 宏观状态
            period: 时期
            
        Returns:
            period_results: 时期模拟结果
        """
        migration_decisions = {}
        n_migrations = 0
        
        # 所有Agent并行做决策（实际可进一步优化并行化）
        for agent in self.agents:
            # 做决策
            new_location = agent.make_decision(
                utility_reuser=utility_reuser,
                params=self.params,
                macro_state=macro_state,
                period=period
            )
            
            # 记录决策
            if new_location != agent.state.current_location:
                migration_decisions[agent.state.agent_id] = {
                    'old_location': agent.state.current_location,
                    'new_location': new_location,
                    'agent': agent
                }
                n_migrations += 1
        
        # 执行迁移并更新Agent状态
        for agent_id, decision in migration_decisions.items():
            agent = decision['agent']
            old_loc = decision['old_location']
            new_loc = decision['new_location']
            
            # 更新Agent状态
            agent.update_state(
                new_location=new_loc,
                new_wage=macro_state['avg_wages'][new_loc],
                period=period
            )
        
        return {
            'migration_decisions': migration_decisions,
            'n_migrations': n_migrations,
            'migration_rate': n_migrations / self.n_agents
        }
    
    def get_regional_populations(self) -> np.ndarray:
        """获取各地区人口分布"""
        populations = np.zeros(self.n_regions)
        
        for agent in self.agents:
            loc = agent.state.current_location
            if 0 <= loc < self.n_regions:
                populations[loc] += 1
        
        return populations
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取群体统计信息"""
        total_migrations = sum(agent.state.migration_count for agent in self.agents)
        avg_migrations = total_migrations / self.n_agents
        
        avg_age = np.mean([agent.state.age for agent in self.agents])
        avg_education = np.mean([agent.state.education for agent in self.agents])
        
        # 按类型统计
        type_counts = {}
        for agent in self.agents:
            agent_type = agent.state.agent_type
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        return {
            'total_agents': self.n_agents,
            'avg_migrations': avg_migrations,
            'avg_age': avg_age,
            'avg_education': avg_education,
            'regional_populations': self.get_regional_populations(),
            'type_distribution': type_counts
        }
    
    def get_agent_by_id(self, agent_id: int) -> Optional[ABMAgent]:
        """通过ID获取Agent"""
        for agent in self.agents:
            if agent.state.agent_id == agent_id:
                return agent
        return None


def demo_agent_decision():
    """演示Agent决策流程"""
    print("\n" + "="*80)
    print("ABM Agent决策演示（复用estimation效用函数）")
    print("="*80)
    
    # 1. 创建宏观状态（10个省份）
    n_regions = 10
    macro_state = {
        'avg_wages': np.array([50000, 60000, 55000, 70000, 48000, 52000, 58000, 62000, 51000, 49000]),
        'amenity_climate': np.random.normal(0, 1, n_regions),
        'amenity_health': np.random.normal(0, 1, n_regions),
        'amenity_education': np.random.normal(0, 1, n_regions),
        'amenity_public': np.random.normal(0, 1, n_regions),
        'housing_prices': np.random.normal(15000, 3000, n_regions),
        'populations': np.random.randint(500000, 50000000, n_regions)
    }
    
    # 2. 创建UtilityReuser
    from src.abm.reuse_utility_wrapper import UtilityReuser
    
    reuser = UtilityReuser(
        distance_matrix=np.random.uniform(100, 2000, (n_regions, n_regions)),
        adjacency_matrix=(np.random.rand(n_regions, n_regions) > 0.8).astype(int),
        region_data=reuser._create_random_region_data() if 'reuser' in dir() else pd.DataFrame(),
        n_regions=n_regions
    )
    
    # 3. 创建参数
    params = {
        'alpha_w': 1.0, 'alpha_home': 1.0, 'alpha_climate': 0.1, 'alpha_health': 0.1,
        'alpha_education': 0.1, 'alpha_public_services': 0.1, 'alpha_hazard': 0.1,
        'rho_base_tier_1': 2.0, 'rho_base_tier_2': 1.0, 'rho_base_tier_3': 0.5,
        'rho_edu': 0.3, 'rho_health': 0.2, 'rho_house': 0.4,
        'gamma_0_type_0': 0.5, 'gamma_0_type_1': 1.5, 'gamma_0_type_2': 2.0,
        'gamma_1': -0.15, 'gamma_2': 0.3, 'gamma_3': -0.35, 'gamma_4': 0.02, 'gamma_5': -0.1
    }
    
    # 4. 创建单个Agent
    agent = ABMAgent(
        agent_id=0,
        agent_type=0,  # 机会型
        age=28,
        education=16,  # 大学本科
        hukou_location=0,
        current_location=0,
        eta_i=0.5
    )
    
    # 5. Agent做决策
    chosen_location = agent.make_decision(reuser, params, macro_state, period=0)
    
    print(f"Agent特征:")
    print(f"  ID: {agent.state.agent_id}")
    print(f"  类型: {agent.state.agent_type} (机会型)")
    print(f"  年龄: {agent.state.age}")
    print(f"  教育: {agent.state.education}年")
    print(f"  户籍: 省份{agent.state.hukou_location}")
    print(f"  当前: 省份{agent.state.current_location}")
    print(f"  个体效应: {agent.state.eta_i}")
    
    print(f"\n决策结果: 省份{chosen_location}")
    
    if chosen_location != agent.state.current_location:
        # 迁移
        agent.update_state(
            new_location=chosen_location,
            new_wage=macro_state['avg_wages'][chosen_location],
            period=0
        )
        print(f"  → 迁移！")
        print(f"  新位置: 省份{agent.state.current_location}")
        print(f"  迁移次数: {agent.state.migration_count}")
    else:
        print(f"  → 留守")
    
    # 6. 创建群体并模拟
    print(f"\n" + "="*40)
    print("创建Agent群体并模拟...")
    
    # 创建群体DataFrame
    population_df = pd.DataFrame({
        'agent_id': range(100),
        'agent_type': np.random.choice([0, 1, 2], 100, p=[0.33, 0.33, 0.34]),
        'age': np.random.randint(18, 60, 100),
        'education': np.random.choice([9, 12, 15, 16], 100),
        'hukou_location': np.random.randint(0, n_regions, 100),
        'current_location': np.random.randint(0, n_regions, 100),
        'eta_i': np.random.normal(0, 1, 100)
    })
    
    # 创建群体管理器
    manager = PopulationManager(population_df, params)
    
    # 模拟一个时期
    results = manager.simulate_period(reuser, macro_state, period=0)
    
    print(f"模拟结果:")
    print(f"  总Agent数: {manager.n_agents}")
    print(f"  迁移人数: {results['n_migrations']}")
    print(f"  迁移率: {results['migration_rate']:.3f}")
    
    # 统计数据
    stats = manager.get_summary_statistics()
    print(f"\n群体统计:")
    print(f"  平均年龄: {stats['avg_age']:.1f}")
    print(f"  平均教育: {stats['avg_education']:.1f}")
    print(f"  平均迁移: {stats['avg_migrations']:.2f}")
    print(f"  类型分布: {stats['type_distribution']}")
    
    return agent, manager, results


if __name__ == '__main__':
    agent, manager, results = demo_agent_decision()
    print("\n✓ Agent决策演示完成")