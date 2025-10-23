"""
综合测试EM算法修复效果

测试参数更新和目标函数优化是否正常工作
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

def main():
    """主测试函数"""
    print("=" * 60)
    print("综合测试EM算法修复效果")
    print("=" * 60)
    
    # 1. 加载配置
    config = ModelConfig()
    config.use_discrete_support = True
    config.em_max_iterations = 2  # 减少迭代次数用于测试
    config.lbfgsb_maxiter = 10    # 增加优化迭代次数
    
    # 2. 加载数据
    print("\n[1] 加载数据...")
    data_loader = DataLoader(config)
    
    try:
        distance_matrix = data_loader.load_distance_matrix()
        adjacency_matrix = data_loader.load_adjacency_matrix()
        df_individual, state_space, transition_matrices, df_region = \
            data_loader.create_estimation_dataset_and_state_space(simplified_state=True)
    except Exception as e:
        print(f"数据加载错误: {e}")
        return
    
    # 3. 抽取小样本
    SAMPLE_SIZE = 5
    print(f"\n[2] 抽取 {SAMPLE_SIZE} 个个体...")
    unique_individuals = df_individual['individual_id'].unique()[:SAMPLE_SIZE]
    df_individual_sample = df_individual[df_individual['individual_id'].isin(unique_individuals)]
    
    print(f"  样本大小: {len(df_individual_sample)} 条观测，来自 {SAMPLE_SIZE} 个个体")
    
    # 4. 创建支撑点生成器
    print("\n[3] 创建离散支撑点生成器...")
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
    print("\n[4] 准备初始参数...")
    initial_params = config.get_initial_params(use_type_specific=True)
    initial_pi_k = np.array([0.4, 0.3, 0.3])
    
    print(f"  参数总数: {len(initial_params)}")
    print(f"  初始类型概率: {initial_pi_k}")
    
    # 6. 运行EM算法
    print("\n[5] 运行EM-with-ω算法...")
    
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
            max_omega_per_individual=100,
            use_simplified_omega=True,
            lbfgsb_maxiter=config.lbfgsb_maxiter
        )
        
        # 7. 输出结果
        print("\n" + "=" * 60)
        print("结果摘要")
        print("=" * 60)
        
        print(f"\n收敛状态: {results['converged']}")
        print(f"迭代次数: {results['n_iterations']}")
        print(f"最终对数似然: {results['final_log_likelihood']:.4f}")
        
        print(f"\n最终类型概率:")
        for k, prob in enumerate(results['type_probabilities']):
            print(f"  类型 {k}: {prob:.4f}")
        
        print(f"\n关键结构参数:")
        estimated_params = results['structural_params']
        key_params = ['alpha_w', 'gamma_0_type_1', 'gamma_0_type_2', 'gamma_1', 'gamma_2']
        for param_name in key_params:
            if param_name in estimated_params:
                print(f"  {param_name}: {estimated_params[param_name]:.4f}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()