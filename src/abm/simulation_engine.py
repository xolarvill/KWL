"""
ABM模拟引擎 - 整合所有模块
协调人口合成、宏观动态、校准和验证
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import os
import pickle

# 添加路径
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.model_config import ModelConfig
from src.abm.synthetic_population import SyntheticPopulation
from src.abm.macro_dynamics import MacroDynamics
from src.abm.calibration import SimulationBasedCalibration
from src.abm.validation import ABMValidator


class ABMSimulationEngine:
    """
    ABM模拟主引擎
    整合所有组件：人口合成 → SMM校准 → 验证 → 政策模拟
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化模拟引擎
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.n_regions = config.n_choices  # 29个省份
        self.n_agents = 100000  # 目标人口规模
        
        # 数据加载状态
        self.population = None
        self.micro_params = None
        self.type_probabilities = None
        self.macro_params = None
        
        print(f"ABM模拟引擎初始化完成")
        print(f"  - 省份数量: {self.n_regions}")
        print(f"  - 目标人口规模: {self.n_agents:,}")
    
    def initialize_from_estimation(self, estimation_result_path: str = None):
        """
        从02估计结果初始化
        
        Args:
            estimation_result_path: 估计结果路径（如果None则使用Config初始值）
        """
        print("\n" + "="*60)
        print("从估计结果初始化ABM")
        print("="*60)
        
        if estimation_result_path and os.path.exists(estimation_result_path):
            # TODO: 加载真实估计结果
            print(f"加载估计结果: {estimation_result_path}")
            # self.micro_params = load_estimated_params(...)
            # self.type_probabilities = load_type_probabilities(...)
            pass
        else:
            print("使用ModelConfig初始值作为占位符")
            self.micro_params = self.config.get_initial_params(use_type_specific=True)
            self.type_probabilities = self.config.get_initial_type_probabilities()
        
        print(f"微观参数数量: {len(self.micro_params)}")
        print(f"类型概率: {self.type_probabilities}")
    
    def build_synthetic_population(self) -> pd.DataFrame:
        """
        构建合成人口（三步法）
        对应论文第1383-1390行
        """
        print("\n" + "="*60)
        print("步骤1: 构建合成人口")
        print("="*60)
        
        # 创建合成器
        synth = SyntheticPopulation(self.config)
        
        # 生成人口
        self.population = synth.create_population(self.type_probabilities)
        
        print(f"合成人口构建完成: {len(self.population):,}个代理人")
        print(f"平均教育水平: {self.population['education'].mean():.1f}")
        print(f"平均年龄: {self.population['age'].mean():.1f}")
        print(f"类型分布: {self.population['agent_type'].value_counts().sort_index().to_dict()}")
        
        return self.population
    
    def calibrate_macro_parameters(self, 
                                   target_moments: Dict[str, float] = None,
                                   save_path: str = None) -> Dict[str, Any]:
        """
        校准宏观参数（SMM）
        对应论文第1478-1491行
        
        Args:
            target_moments: 目标矩（如果None使用默认值）
            save_path: 校准结果保存路径
            
        Returns:
            calibration_results: 校准结果
        """
        print("\n" + "="*60)
        print("步骤2: SMM宏观参数校准")
        print("="*60)
        
        # 检查合成人口是否存在
        if self.population is None:
            raise ValueError("请先构建合成人口 (build_synthetic_population)")
        
        # 默认目标矩（来自论文表8）
        if target_moments is None:
            target_moments = {
                'population_gini': 0.67,
                'migration_rate_std': 0.024,
                'wage_migration_elasticity': 0.12,
                'housing_migration_elasticity': 0.35,
                'hukou_wage_premium': 0.15
            }
        
        print(f"目标矩: {target_moments}")
        
        # 创建校准器
        # ABM校准应始终使用29个省份，确保micro_params不包含n_choices以强制其使用默认值
        abm_micro_params = self.micro_params.copy()
        abm_micro_params.pop('n_choices', None) # 移除n_choices，让Calibration使用其内部的29默认值
        smm = SimulationBasedCalibration(
            synthetic_population=self.population,
            micro_params=abm_micro_params,
            target_moments=target_moments,
            n_periods=9  # 2010-2018
        )
        
        # 执行校准
        calibration_results = smm.calibrate(
            initial_guess=np.array([0.08, 0.32]),
            bounds=[(0.01, 0.5), (0.1, 1.0)]
        )
        
        # 保存最优参数
        self.macro_params = calibration_results['optimal_params']
        
        # 保存结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(calibration_results, f)
            print(f"\n校准结果已保存: {save_path}")
        
        # 打印校准报告
        self._print_calibration_report(calibration_results)
        
        return calibration_results
    
    def _print_calibration_report(self, calibration_results: Dict[str, Any]):
        """打印校准报告"""
        print("\n" + "="*60)
        print("校准报告")
        print("="*60)
        
        print(f"\n最优宏观参数:")
        for param, value in calibration_results['optimal_params'].items():
            print(f"  {param}: {value:.4f}")
        
        print(f"\n矩拟合质量:")
        print(f"  平均相对误差: {calibration_results['fit_metrics']['average_relative_error']:.4f}")
        
        print(f"\n目标矩 vs 模拟矩:")
        for moment in calibration_results['target_moments'].keys():
            target = calibration_results['target_moments'][moment]
            simulated = calibration_results['simulated_moments'][moment]
            error = abs(simulated - target) / max(abs(target), 1e-6)
            print(f"  {moment}:")
            print(f"    目标值: {target:.4f}")
            print(f"    模拟值: {simulated:.4f}")
            print(f"    相对误差: {error:.4f} ({error*100:.2f}%)")
    
    def validate_model(self, plot: bool = True) -> Dict[str, Any]:
        """
        验证模型涌现性质
        对应论文第1540-1602行
        
        Args:
            plot: 是否绘制验证图表
            
        Returns:
            validation_results: 验证结果
        """
        print("\n" + "="*60)
        print("步骤3: 模型验证")
        print("="*60)
        
        # 需要宏观参数和人口分布
        if self.macro_params is None or self.population is None:
            raise ValueError("请先完成校准 (calibrate_macro_parameters)")
        
        # 获取当前人口分布（简化：从合成人口统计）
        # TODO: 应基于完整ABM模拟后的最终分布
        regional_populations = np.zeros(self.n_regions)
        for location in self.population['current_location'].values:
            if 0 <= location < self.n_regions:
                regional_populations[location] += 1
        
        # 创建验证器
        validator = ABMValidator(regional_populations)
        
        # 运行完整验证
        validation_results = validator.run_full_validation(plot=plot)
        
        return validation_results
    
    def run_policy_simulation(self, 
                             policy_params: Dict[str, Any],
                             n_periods: int = 12) -> Dict[str, Any]:
        """
        运行政策反事实模拟
        对应论文第1604行及以后
        
        Args:
            policy_params: 政策参数（修改户籍惩罚等）
            n_periods: 模拟期数（默认12年，对应2019-2030）
            
        Returns:
            policy_results: 政策模拟结果
        """
        print("\n" + "="*60)
        print("步骤4: 政策反事实模拟")
        print("="*60)
        
        # 检查前置条件
        if self.population is None or self.macro_params is None:
            raise ValueError("请先完成人口合成和参数校准")
        
        print(f"政策参数: {policy_params}")
        print(f"模拟期数: {n_periods}年")
        
        # 更新微观参数（应用政策）
        policy_micro_params = self.micro_params.copy()
        for key, value in policy_params.items():
            if key in policy_micro_params:
                policy_micro_params[key] = value
        
        # 初始化宏观动态
        macro_dynamics = MacroDynamics(self.n_regions, self.macro_params)
        
        # 模拟政策效果（简化版本）
        # TODO: 接入完整ABM决策引擎
        simulation_results = self._simulate_policy_impact(
            macro_dynamics, policy_micro_params, n_periods
        )
        
        print(f"政策模拟完成")
        
        return simulation_results
    
    def _simulate_policy_impact(self, macro_dynamics: MacroDynamics, 
                               policy_micro_params: Dict[str, float],
                               n_periods: int) -> Dict[str, Any]:
        """
        模拟政策影响（简化版本）
        
        Args:
            macro_dynamics: 宏观动态模型
            policy_micro_params: 政策调整后的微观参数
            n_periods: 模拟期数
            
        Returns:
            simulation_results: 模拟结果
        """
        # 初始化跟踪变量
        population_history = []
        wage_history = []
        price_history = []
        
        # 初始状态
        initial_state = macro_dynamics.get_macro_variables()
        population_history.append(initial_state['populations'])
        wage_history.append(initial_state['avg_wages'])
        price_history.append(initial_state['housing_prices'])
        
        # 模拟多个时期
        for period in range(n_periods):
            # 模拟迁移决策（简化：随机+政策影响）
            migration_decisions = self._generate_migration_decisions(policy_micro_params)
            
            # 更新宏观变量
            updates = macro_dynamics.update_regions(migration_decisions, period)
            
            # 记录状态
            current_state = macro_dynamics.get_macro_variables()
            population_history.append(current_state['populations'])
            wage_history.append(current_state['avg_wages'])
            price_history.append(current_state['housing_prices'])
        
        # 计算政策效果
        baseline_final = population_history[0]  # 简化：用初始作为基准
        policy_final = population_history[-1]
        
        population_change = policy_final - baseline_final
        change_rate = population_change / (baseline_final + 1e-6)
        
        results = {
            'population_history': np.array(population_history),
            'wage_history': np.array(wage_history),
            'price_history': np.array(price_history),
            'final_population': policy_final,
            'population_change': population_change,
            'change_rate': change_rate
        }
        
        return results
    
    def _generate_migration_decisions(self, params: Dict[str, float]) -> Dict[int, int]:
        """
        生成迁移决策（简化版本）
        
        Args:
            params: 参数（包含政策影响）
            
        Returns:
            migration_decisions: 迁移决策字典
        """
        # 简化：随机生成一些迁移
        n_migrants = int(len(self.population) * 0.05)  # 5%的迁移率
        
        migration_decisions = {}
        for i in range(n_migrants):
            agent_id = np.random.randint(0, len(self.population))
            new_location = np.random.randint(0, self.n_regions)
            migration_decisions[agent_id] = new_location
        
        return migration_decisions
    
    def save_model(self, save_path: str = "results/abm/calibrated_model.pkl"):
        """保存校准后的完整模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_state = {
            'population': self.population,
            'micro_params': self.micro_params,
            'type_probabilities': self.type_probabilities,
            'macro_params': self.macro_params,
            'config': self.config
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"模型已保存: {save_path}")
    
    def load_model(self, load_path: str):
        """加载已校准的模型"""
        with open(load_path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.population = model_state['population']
        self.micro_params = model_state['micro_params']
        self.type_probabilities = model_state['type_probabilities']
        self.macro_params = model_state['macro_params']
        
        print(f"模型已加载: {load_path}")


def demo_abm_full_pipeline():
    """演示完整ABM流程"""
    print("\n" + "="*80)
    print("ABM完整流程演示")
    print("="*80)
    
    # 1. 初始化配置
    config = ModelConfig()
    
    # 2. 创建模拟引擎
    engine = ABMSimulationEngine(config)
    
    # 3. 从估计结果初始化（使用Config初始值作为占位符）
    engine.initialize_from_estimation()
    
    # 4. 构建合成人口
    population = engine.build_synthetic_population()
    
    # 5. 校准宏观参数
    calibration_results = engine.calibrate_macro_parameters(
        save_path="results/abm/calibration_results.pkl"
    )
    
    # 6. 验证模型
    validation_results = engine.validate_model(plot=False)  # 演示中不绘图
    
    # 7. 运行政策模拟示例
    policy_params = {
        'rho_base_tier_1': 0.0,  # 取消一线城市户籍限制
        'rho_edu': 0.0,           # 取消教育交互项
        'rho_health': 0.0,        # 取消医疗交互项
        'rho_house': 0.0          # 取消住房交互项
    }
    
    policy_results = engine.run_policy_simulation(policy_params, n_periods=12)
    
    # 8. 保存最终模型
    engine.save_model(save_path="results/abm/calibrated_model.pkl")
    
    # 打印总结
    print("\n" + "="*80)
    print("ABM流程完成总结")
    print("="*80)
    print(f"合成人口规模: {len(population):,}人")
    print(f"校准参数: {calibration_results['optimal_params']}")
    print(f"平均误差: {calibration_results['fit_metrics']['average_relative_error']:.4f}")
    print(f"Zipf验证: {'通过' if validation_results['zipf_law']['in_range'] else '未通过'}")
    print(f"胡焕庸线验证: {'通过' if validation_results['hu_line']['is_accurate'] else '未通过'}")
    print(f"政策人口变化率: {policy_results['change_rate'].mean():.4f}")
    
    return engine, calibration_results, validation_results, policy_results


if __name__ == '__main__':
    # 运行完整演示
    results = demo_abm_full_pipeline()
    print("\n✓ ABM完整流程演示完成")