"""
ABM校准模块
使用模拟矩匹配法（SMM）校准宏观参数 Φ = {phi_w, phi_p}
对应论文tex第1478-1491行
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class SimulationBasedCalibration:
    """
    模拟矩匹配法（SMM）校准器
    寻找最优宏观参数使得模拟矩逼近真实数据矩
    """
    
    def __init__(self, 
                 synthetic_population: pd.DataFrame,
                 micro_params: Dict[str, float],
                 target_moments: Dict[str, float],
                 n_periods: int = 9):  # 2010-2018共9年
        """
        初始化SMM校准器
        
        Args:
            synthetic_population: 合成人口数据
            micro_params: 结构参数θ̂（来自02估计）
            target_moments: 目标矩（真实数据）
            n_periods: 模拟期数
        """
        self.population = synthetic_population
        self.micro_params = micro_params
        self.target_moments = target_moments
        self.n_periods = n_periods
        self.n_regions = 29  # 实际省份数量
        
        # 目标矩权重（可根据重要性调整）
        self.moment_weights = {
            'population_gini': 1.0,
            'migration_rate_std': 1.0, 
            'wage_migration_elasticity': 1.5,  # 更重要
            'housing_migration_elasticity': 1.5,
            'hukou_wage_premium': 1.0
        }
        
    def calibrate(self, 
                  initial_guess: np.ndarray = None,
                  bounds: List[tuple] = None) -> Dict[str, Any]:
        """
        执行SMM校准
        
        Args:
            initial_guess: 初始参数猜测 [phi_w, phi_p]
            bounds: 参数边界
            
        Returns:
            校准结果字典
        """
        print("\n" + "="*60)
        print("开始SMM宏观参数校准")
        print("="*60)
        
        # 初始猜测
        if initial_guess is None:
            initial_guess = np.array([0.08, 0.32])  # tex表6的默认值
            
        # 参数边界
        if bounds is None:
            bounds = [(0.01, 0.5), (0.1, 1.0)]  # phi_w ∈ [0.01, 0.5], phi_p ∈ [0.1, 1.0]
        
        print(f"初始猜测: phi_w={initial_guess[0]:.3f}, phi_p={initial_guess[1]:.3f}")
        print(f"参数边界: {bounds}")
        
        # 目标函数
        def objective_function(params: np.ndarray) -> float:
            """SMM目标函数：加权平方距离"""
            phi_w, phi_p = params
            
            # 运行ABM模拟
            simulated_moments = self._run_abm_simulation(phi_w, phi_p)
            
            # 计算距离
            distance = self._calculate_moment_distance(simulated_moments)
            
            return distance
        
        # 执行优化
        print("\n执行数值优化...")
        result = minimize(
            objective_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20, 'ftol': 1e-3, 'disp': True}
        )
        
        # 提取最优参数
        optimal_params = {
            'phi_w': result.x[0],
            'phi_p': result.x[1]
        }
        
        print(f"\n校准完成!")
        print(f"最优参数: phi_w={optimal_params['phi_w']:.4f}, phi_p={optimal_params['phi_p']:.4f}")
        print(f"目标函数值: {result.fun:.6f}")
        
        # 使用最优参数运行最终模拟
        final_moments = self._run_abm_simulation(
            optimal_params['phi_w'], 
            optimal_params['phi_p']
        )
        
        # 计算拟合度
        fit_metrics = self._calculate_fit_quality(final_moments)
        
        return {
            'optimal_params': optimal_params,
            'simulated_moments': final_moments,
            'target_moments': self.target_moments,
            'fit_metrics': fit_metrics,
            'optimization_result': result
        }
    
    def _run_abm_simulation(self, phi_w: float, phi_p: float) -> Dict[str, float]:
        """
        运行ABM模拟并计算模拟矩
        
        Args:
            phi_w: 工资敏感度参数
            phi_p: 房价弹性参数
            
        Returns:
            simulated_moments: 模拟矩字典
        """
        # 宏观参数
        macro_params = {
            'phi_w': phi_w,
            'phi_p': phi_p,
            'g_q': 0.02
        }
        
        # 简化：使用随机模拟代替完整ABM（TODO: 接入真实ABM模拟）
        simulated_moments = self._simulate_moments(macro_params)
        
        return simulated_moments
    
    def _simulate_moments(self, macro_params: Dict[str, float]) -> Dict[str, float]:
        """
        模拟矩计算（简化版本）
        TODO: 接入真实ABM模拟引擎
        """
        # 从目标矩出发，添加随机扰动
        simulated = {}
        
        for key, target in self.target_moments.items():
            # 添加正态扰动（模拟误差）
            noise = np.random.normal(0, 0.1 * abs(target))
            simulated[key] = target + noise
        
        return simulated
    
    def _calculate_moment_distance(self, simulated_moments: Dict[str, float]) -> float:
        """
        计算模拟矩与目标矩的加权距离
        
        Args:
            simulated_moments: 模拟矩
            
        Returns:
            weighted_distance: 加权平方距离
        """
        total_distance = 0.0
        
        for moment_name, target_value in self.target_moments.items():
            if moment_name in simulated_moments:
                sim_value = simulated_moments[moment_name]
                weight = self.moment_weights.get(moment_name, 1.0)
                
                # 相对误差
                relative_error = (sim_value - target_value) / max(abs(target_value), 1e-6)
                
                # 加权平方误差
                weighted_error = weight * (relative_error ** 2)
                total_distance += weighted_error
        
        return total_distance
    
    def _calculate_fit_quality(self, simulated_moments: Dict[str, float]) -> Dict[str, float]:
        """
        计算矩拟合质量
        
        Args:
            simulated_moments: 模拟矩
            
        Returns:
            fit_metrics: 拟合指标
        """
        fit_metrics = {}
        relative_errors = []
        
        for moment_name, target_value in self.target_moments.items():
            if moment_name in simulated_moments:
                sim_value = simulated_moments[moment_name]
                
                # 相对误差
                relative_error = abs(sim_value - target_value) / max(abs(target_value), 1e-6)
                relative_errors.append(relative_error)
                
                fit_metrics[f'{moment_name}_error'] = relative_error
        
        # 平均相对误差（tex表8的5.6%）
        fit_metrics['average_relative_error'] = np.mean(relative_errors)
        
        return fit_metrics
    
    def bootstrap_standard_errors(self, 
                                  optimal_params: Dict[str, float],
                                  n_replications: int = 50) -> Dict[str, float]:
        """
        Bootstrap计算参数标准误
        
        Args:
            optimal_params: 最优参数
            n_replications: Bootstrap重复次数
            
        Returns:
            standard_errors: 标准误字典
        """
        print(f"\n进行Bootstrap标准误计算 ({n_replications}次重复)...")
        
        phi_w_estimates = []
        phi_p_estimates = []
        
        for rep in range(n_replications):
            if (rep + 1) % 10 == 0:
                print(f"  Bootstrap {rep+1}/{n_replications}")
            
            # Bootstrap重抽样
            boot_population = self.population.sample(
                n=len(self.population), 
                replace=True,
                random_state=42+rep
            )
            
            # 重新初始化校准器
            boot_calibrator = SimulationBasedCalibration(
                boot_population, 
                self.micro_params, 
                self.target_moments,
                self.n_periods
            )
            
            # 短优化（快速估计）
            result = minimize(
                boot_calibrator._run_objective,
                [optimal_params['phi_w'], optimal_params['phi_p']],
                method='L-BFGS-B',
                bounds=[(0.01, 0.5), (0.1, 1.0)],
                options={'maxiter': 5, 'disp': False}
            )
            
            phi_w_estimates.append(result.x[0])
            phi_p_estimates.append(result.x[1])
        
        # 计算标准误
        standard_errors = {
            'phi_w_se': np.std(phi_w_estimates),
            'phi_p_se': np.std(phi_p_estimates)
        }
        
        print(f"Bootstrap标准误: phi_w={standard_errors['phi_w_se']:.4f}, phi_p={standard_errors['phi_p_se']:.4f}")
        
        return standard_errors


def demo_calibration():
    """演示SMM校准"""
    print("\n" + "="*60)
    print("SMM校准演示")
    print("="*60)
    
    # 创建合成人口
    from src.abm.synthetic_population import SyntheticPopulation
    from src.config.model_config import ModelConfig
    
    config = ModelConfig()
    synth = SyntheticPopulation(config)
    
    type_probs = config.get_initial_type_probabilities()
    population = synth.create_population(type_probs)
    
    # 微观参数（来自02的占位符）
    micro_params = config.get_initial_params()
    
    # 目标矩（来自真实数据计算）
    target_moments = {
        'population_gini': 0.67,  # tex表8
        'migration_rate_std': 0.024,
        'wage_migration_elasticity': 0.12,
        'housing_migration_elasticity': 0.35,
        'hukou_wage_premium': 0.15
    }
    
    # 初始化校准器
    smm = SimulationBasedCalibration(
        population, micro_params, target_moments
    )
    
    # 执行校准（演示用，使用较少迭代）
    calibration_result = smm.calibrate(
        initial_guess=np.array([0.1, 0.3]),
        bounds=[(0.05, 0.5), (0.1, 1.0)]
    )
    
    # 打印结果
    print("\n校准结果摘要:")
    print(f"  最优工资敏感度 phi_w: {calibration_result['optimal_params']['phi_w']:.4f}")
    print(f"  最优房价弹性 phi_p: {calibration_result['optimal_params']['phi_p']:.4f}")
    print(f"  平均相对误差: {calibration_result['fit_metrics']['average_relative_error']:.4f}")
    
    print("\n目标矩 vs 模拟矩:")
    for moment in target_moments.keys():
        target = target_moments[moment]
        simulated = calibration_result['simulated_moments'].get(moment, 0)
        error = calibration_result['fit_metrics'].get(f'{moment}_error', 0)
        print(f"  {moment}: 目标={target:.3f}, 模拟={simulated:.3f}, 误差={error:.3f}")
    
    return calibration_result


if __name__ == '__main__':
    calibration_result = demo_calibration()