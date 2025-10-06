"""
测试inference模块的实现，包括数值Hessian和Bootstrap方法
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.estimation.em_nfxp import run_em_algorithm
from src.config.model_config import ModelConfig
from src.estimation.inference import estimate_mixture_model_standard_errors, bootstrap_standard_errors


def test_numerical_inference(sample_size: int = 100):
    """
    测试数值Hessian方法计算标准误
    """
    print("=" * 80)
    print("测试 1: 数值Hessian方法计算标准误")
    print("=" * 80)

    # 加载数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    df_individual, df_region, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)

    # 使用小样本
    unique_ids = df_individual['individual_id'].unique()[:sample_size]
    df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
    print(f"使用 {sample_size} 个个体进行测试，共 {len(df_individual)} 条观测")

    # 运行EM估计
    print("\n运行EM估计...")
    estimation_params = {
        "observed_data": df_individual,
        "regions_df": df_region,
        "state_space": state_space,
        "transition_matrices": transition_matrices,
        "distance_matrix": distance_matrix,
        "adjacency_matrix": adjacency_matrix,
        "beta": 0.95,
        "n_types": 3,
        "max_iterations": 3,  # 快速测试
        "tolerance": 1e-3,
        "n_choices": len(df_region['provcd'].unique()),
        "use_migration_behavior_init": True
    }
    results = run_em_algorithm(**estimation_params)
    estimated_params = results["structural_params"]

    # 计算标准误
    print("\n计算标准误（type_0_only方法）...")
    std_errors, t_stats, p_values = estimate_mixture_model_standard_errors(
        estimated_params=estimated_params,
        observed_data=df_individual,
        state_space=state_space,
        transition_matrices=transition_matrices,
        beta=0.95,
        regions_df=df_region,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        n_types=3,
        method="type_0_only"
    )

    # 输出结果
    print("\n标准误计算结果（前10个参数）：")
    print(f"{'参数名':<30} {'点估计':>12} {'标准误':>12} {'t统计量':>12} {'p值':>12}")
    print("-" * 80)
    param_names = [k for k in estimated_params.keys() if k != 'n_choices'][:10]
    for param_name in param_names:
        pe = estimated_params[param_name]
        se = std_errors.get(param_name, 'N/A')
        ts = t_stats.get(param_name, 'N/A')
        pv = p_values.get(param_name, 'N/A')

        if isinstance(se, str):
            print(f"{param_name:<30} {pe:>12.4f} {se:>12} {ts:>12} {pv:>12}")
        else:
            print(f"{param_name:<30} {pe:>12.4f} {se:>12.4f} {ts:>12.4f} {pv:>12.4f}")

    print("\n测试 1 完成！\n")
    return results


def test_bootstrap_inference(sample_size: int = 50, n_bootstrap: int = 5):
    """
    测试Bootstrap方法计算标准误
    """
    print("=" * 80)
    print(f"测试 2: Bootstrap方法计算标准误 ({n_bootstrap}次重复)")
    print("=" * 80)

    # 加载数据
    config = ModelConfig()
    data_loader = DataLoader(config)
    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    df_individual, df_region, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)

    # 使用小样本
    unique_ids = df_individual['individual_id'].unique()[:sample_size]
    df_individual = df_individual[df_individual['individual_id'].isin(unique_ids)]
    print(f"使用 {sample_size} 个个体进行测试，共 {len(df_individual)} 条观测")

    # 运行EM估计
    print("\n运行EM估计...")
    estimation_params = {
        "observed_data": df_individual,
        "regions_df": df_region,
        "state_space": state_space,
        "transition_matrices": transition_matrices,
        "distance_matrix": distance_matrix,
        "adjacency_matrix": adjacency_matrix,
        "beta": 0.95,
        "n_types": 3,
        "max_iterations": 3,  # 快速测试
        "tolerance": 1e-3,
        "n_choices": len(df_region['provcd'].unique()),
        "use_migration_behavior_init": True
    }
    results = run_em_algorithm(**estimation_params)
    estimated_params = results["structural_params"]
    posterior_probs = results["posterior_probs"]
    log_likelihood_matrix = results["log_likelihood_matrix"]
    type_probabilities = results["type_probabilities"]

    # 运行Bootstrap
    print(f"\n运行Bootstrap ({n_bootstrap}次重复)...")
    std_errors, conf_intervals, t_stats, p_values = bootstrap_standard_errors(
        estimated_params=estimated_params,
        posterior_probs=posterior_probs,
        log_likelihood_matrix=log_likelihood_matrix,
        type_probabilities=type_probabilities,
        observed_data=df_individual,
        state_space=state_space,
        transition_matrices=transition_matrices,
        beta=0.95,
        regions_df=df_region,
        distance_matrix=distance_matrix,
        adjacency_matrix=adjacency_matrix,
        n_types=3,
        n_bootstrap=n_bootstrap,
        max_em_iterations=2,  # 快速测试
        em_tolerance=1e-2,
        seed=42,
        n_jobs=1,  # 串行以便调试
        verbose=True
    )

    # 输出结果
    print("\nBootstrap结果（前10个参数）：")
    print(f"{'参数名':<30} {'点估计':>12} {'标准误':>12} {'95% CI下界':>12} {'95% CI上界':>12}")
    print("-" * 90)
    param_names = [k for k in estimated_params.keys() if k != 'n_choices'][:10]
    for param_name in param_names:
        pe = estimated_params[param_name]
        se = std_errors[param_name]
        ci = conf_intervals[param_name]
        print(f"{param_name:<30} {pe:>12.4f} {se:>12.4f} {ci[0]:>12.4f} {ci[1]:>12.4f}")

    print("\n测试 2 完成！\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="测试inference模块")
    parser.add_argument('--test', type=int, default=1, choices=[1, 2, 3],
                        help='选择测试: 1=数值Hessian, 2=Bootstrap, 3=两者都测试')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='测试样本大小（个体数量）')
    parser.add_argument('--n-bootstrap', type=int, default=5,
                        help='Bootstrap重复次数')
    args = parser.parse_args()

    if args.test in [1, 3]:
        test_numerical_inference(sample_size=args.sample_size)

    if args.test in [2, 3]:
        test_bootstrap_inference(sample_size=args.sample_size//2, n_bootstrap=args.n_bootstrap)

    print("所有测试完成！")
