"""
诊断不同样本大小对EM算法的影响

比较小样本和大样本的收敛行为
"""
import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.data_handler.data_loader import DataLoader
from src.model.discrete_support import DiscreteSupportGenerator
from src.estimation.em_with_omega import run_em_algorithm_with_omega

def run_test(sample_size, test_name):
    """运行指定样本大小的测试"""
    print(f"\n{'='*50}")
    print(f"测试: {test_name} (样本大小: {sample_size})")
    print(f"{'='*50}")
    
    # 1. 加载配置
    config = ModelConfig()
    config.use_discrete_support = True
    config.em_max_iterations = 2  # 减少迭代次数用于测试
    config.lbfgsb_maxiter = 5    # 适中的优化迭代
    
    # 2. 加载数据
    print("[1] 加载数据...")
    data_loader = DataLoader(config)
    
    try:
        distance_matrix = data_loader.load_distance_matrix()
        adjacency_matrix = data_loader.load_adjacency_matrix()
        df_individual, state_space, transition_matrices, df_region = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None
    
    # 3. 抽取样本
    print(f"[2] 抽取 {sample_size} 个个体...")
    unique_individuals = df_individual['individual_id'].unique()[:sample_size]
    df_individual_sample = df_individual[df_individual['individual_id'].isin(unique_individuals)]
    
    print(f"  样本大小: {len(df_individual_sample)} 条观测，来自 {sample_size} 个个体")
    
    # 4. 创建支撑点生成器
    print("[3] 创建离散支撑点生成器...")
    support_gen = DiscreteSupportGenerator(
        n_eta_support=3,
        n_nu_support=3,
        n_xi_support=3,
        n_sigma_support=2,
        eta_range=(-1.0, 1.0),
        nu_range=(-0.5, 0.5),
        xi_range=(-0.5, 0.5),
        sigma_range=(0.5, 1.5)
    )
    
    # 5. 准备初始参数
    print("[4] 准备初始参数...")
    initial_params = config.get_initial_params(use_type_specific=True)
    initial_pi_k = np.array([0.4, 0.3, 0.3])
    
    # 6. 运行EM算法
    print("[5] 运行EM-with-ω算法...")
    
    try:
        results = run_em_algorithm_with_omega(
            observed_data=df_individual_sample,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=config.discount_factor,
            n_types=config.em_n_types,
            regions_df=df_region,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            support_generator=support_gen,
            prov_to_idx=data_loader.prov_to_idx,
            max_iterations=config.em_max_iterations,
            tolerance=config.em_tolerance,
            n_choices=config.n_choices,
            initial_params=initial_params,
            initial_pi_k=initial_pi_k,
            max_omega_per_individual=100,  # 统一设置
            use_simplified_omega=True,
            lbfgsb_maxiter=config.lbfgsb_maxiter
        )
        
        # 输出结果
        print(f"\n结果摘要 - {test_name}:")
        print(f"  收敛状态: {results['converged']}")
        print(f"  迭代次数: {results['n_iterations']}")
        print(f"  最终对数似然: {results['final_log_likelihood']:.4f}")
        
        # 检查参数是否更新
        estimated_params = results['structural_params']
        key_params = ['alpha_w', 'gamma_0_type_1', 'gamma_0_type_2']
        params_updated = False
        for param_name in key_params:
            if param_name in initial_params and param_name in estimated_params:
                initial_val = initial_params[param_name]
                final_val = estimated_params[param_name]
                if abs(final_val - initial_val) > 1e-6:
                    print(f"  {param_name}: {initial_val:.4f} -> {final_val:.4f}")
                    params_updated = True
        
        if not params_updated:
            print("  参数未更新!")
            
        return results
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("EM算法样本大小影响诊断")
    print("=" * 50)
    
    # 测试不同样本大小
    sample_sizes = [5, 10, 20, 50, 100]
    results = {}
    
    for size in sample_sizes:
        test_name = f"样本大小_{size}"
        results[size] = run_test(size, test_name)
    
    # 总结
    print(f"\n{'='*50}")
    print("诊断总结")
    print(f"{'='*50}")
    
    for size, result in results.items():
        if result:
            converged = result['converged']
            final_ll = result['final_log_likelihood']
            print(f"样本大小 {size:3d}: 收敛={converged}, 对数似然={final_ll:.2f}")
        else:
            print(f"样本大小 {size:3d}: 执行失败")

if __name__ == '__main__':
    main()