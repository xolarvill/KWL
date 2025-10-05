"""
Type特定参数(gamma_0)自动搜索脚本

使用两阶段策略寻找最优的type迁移成本参数：
1. 粗粒度网格搜索：快速评估大量参数组合
2. 精细化优化：对候选参数进行完整评估

Usage:
    uv run python scripts/search_type_params.py --stage 1  # 粗搜索
    uv run python scripts/search_type_params.py --stage 2  # 精搜索
    uv run python scripts/search_type_params.py --quick    # 快速测试
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_handler.data_loader import DataLoader
from src.config.model_config import ModelConfig
from src.estimation.em_nfxp import run_em_algorithm
from src.estimation.param_search import (
    evaluate_type_separation,
    check_degeneracy,
    suggest_gamma_adjustment,
    evaluate_type_separation_for_low_migration_rate,
    check_degeneracy_for_low_migration_data,
    adaptive_type_identification_score
)
from src.model.likelihood import calculate_log_likelihood


def generate_gamma_grid(n_types: int = 3, granularity: str = 'sparse') -> list:
    """
    生成gamma_0参数网格

    Args:
        n_types: Type数量
        granularity: 'sparse'(稀疏,~20组合), 'full'(完整,~120组合), 'fine'(精细,~500组合)

    Returns:
        参数组合列表，每个元素为(gamma_0_type_0, gamma_0_type_1, gamma_0_type_2)
    """
    if granularity == 'sparse':
        # 稀疏网格：覆盖广但组合少 (~20组合, ~100分钟)
        base_values = [0.1, 1.0, 5.0]
    elif granularity == 'full':
        # 完整网格：原始粗粒度 (~120组合, ~10小时)
        base_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    elif granularity == 'fine':
        # 精细网格：密集采样 (~500组合, ~40小时)
        base_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    # 生成所有组合
    combinations = list(product(base_values, repeat=n_types))

    # 过滤掉没有区分度的组合（所有值相同）
    combinations = [c for c in combinations if len(set(c)) > 1]

    # 过滤掉差异过小的组合
    combinations = [
        c for c in combinations
        if max(c) / min(c) >= 1.5  # 最大值至少是最小值的1.5倍
    ]

    print(f"Generated {len(combinations)} parameter combinations ({granularity} granularity)")
    return combinations


def calculate_migration_rate(observed_data: pd.DataFrame) -> float:
    """
    计算数据中的迁移率
    """
    # 假设存在migration_flag列，标记是否发生迁移
    if 'mig_flag' in observed_data.columns:
        n_migrants = observed_data.groupby('individual_id')['mig_flag'].sum().sum()
        total_obs = len(observed_data)
        migration_rate = n_migrants / total_obs if total_obs > 0 else 0.0
    else:
        # 如果没有明确的迁移标志，可以根据位置变化计算
        # 计算每个个体的位置变化次数
        n_changes = 0
        n_total_obs = 0
        
        for individual_id, group in observed_data.groupby('individual_id'):
            if 'location' in group.columns and len(group) > 1:
                locations = group['location'].values
                changes = sum(1 for i in range(1, len(locations)) if locations[i] != locations[i-1])
                n_changes += changes
                n_total_obs += len(group) - 1
        
        migration_rate = n_changes / n_total_obs if n_total_obs > 0 else 0.0
    
    return migration_rate


def evaluate_single_combination_fast(
    gamma_combination: tuple,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: dict,
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    verbose: bool = False
) -> dict:
    """
    快速评估单个gamma_0参数组合（只运行E步）
    现在支持低迁移率数据的评估

    Returns:
        评估结果字典
    """
    n_types = len(gamma_combination)
    N = observed_data["individual_id"].nunique()

    # 计算迁移率
    migration_rate = calculate_migration_rate(observed_data)

    # 构建参数字典
    params = {
        "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0,
        "rho_base_tier_1": 1.0, "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1,
        **{f"gamma_0_type_{i}": gamma_combination[i] for i in range(n_types)},
        "gamma_1": -0.1, "gamma_2": 0.2, "gamma_3": -0.4,
        "gamma_4": 0.01, "gamma_5": -0.05,
        "alpha_climate": 0.1, "alpha_health": 0.1,
        "alpha_education": 0.1, "alpha_public_services": 0.1,
        "n_choices": 31
    }

    # 初始化type probabilities为均匀分布
    pi_k = np.ones(n_types) / n_types

    # 计算每个type的log-likelihood
    log_likelihood_matrix = np.zeros((N, n_types))

    for k in range(n_types):
        try:
            # 创建类型特定参数
            type_specific_params = params.copy()
            if f'gamma_0_type_{k}' in params:
                type_specific_params['gamma_0'] = params[f'gamma_0_type_{k}']
            
            # 调用calculate_log_likelihood with verbose=False
            log_lik_obs = -calculate_log_likelihood(
                params=type_specific_params,
                observed_data=observed_data,
                state_space=state_space,
                agent_type=int(k),
                beta=beta,
                transition_matrices=transition_matrices,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                verbose=False  # 关键：禁用打印
            )

            # Sum log-likelihoods for each individual
            observed_data_copy = observed_data.copy()
            observed_data_copy['log_lik_obs'] = log_lik_obs
            individual_log_lik = observed_data_copy.groupby("individual_id")['log_lik_obs'].sum()
            log_likelihood_matrix[:, k] = individual_log_lik.values

        except Exception as e:
            if verbose:
                print(f"  Error computing likelihood for type {k}: {e}")
            log_likelihood_matrix[:, k] = -1e10

    # 计算posterior probabilities (Bayes' rule in log space)
    pi_k_safe = np.maximum(pi_k, 1e-10)
    pi_k_safe = pi_k_safe / np.sum(pi_k_safe)

    log_pi_k = np.log(pi_k_safe)
    weighted_log_lik = log_likelihood_matrix + log_pi_k

    # Log-sum-exp trick for numerical stability
    max_log_lik = np.max(weighted_log_lik, axis=1, keepdims=True)
    log_marginal_lik = max_log_lik + np.log(
        np.sum(np.exp(weighted_log_lik - max_log_lik), axis=1, keepdims=True)
    )
    log_posterior = weighted_log_lik - log_marginal_lik

    # Compute posterior probabilities
    posterior_probs = np.exp(log_posterior)
    posterior_probs = np.maximum(posterior_probs, 0)
    row_sums = posterior_probs.sum(axis=1, keepdims=True)
    posterior_probs = posterior_probs / np.maximum(row_sums, 1e-10)

    # 计算total log-likelihood
    log_likelihood = np.sum(log_marginal_lik)

    # 计算评估指标（使用适应低迁移率的版本）
    n_params = len(params)
    n_obs = len(observed_data)

    metrics = evaluate_type_separation_for_low_migration_rate(
        posterior_probs, log_likelihood, n_params, n_obs, migration_rate
    )

    # 添加gamma值信息
    for i, gamma_val in enumerate(gamma_combination):
        metrics[f'gamma_0_type_{i}'] = gamma_val

    # 检查退化（使用适应低迁移率的版本）
    is_degenerate, deg_msg = check_degeneracy_for_low_migration_data(posterior_probs, migration_rate)
    metrics['is_degenerate'] = is_degenerate
    metrics['degeneracy_message'] = deg_msg

    # 添加迁移率信息
    metrics['migration_rate'] = migration_rate

    return metrics


def evaluate_single_combination(
    gamma_combination: tuple,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: dict,
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_em_iterations: int = 1,
    verbose: bool = False
) -> dict:
    """
    评估单个gamma_0参数组合（运行完整EM算法）
    现在支持低迁移率数据的评估

    Returns:
        评估结果字典
    """
    n_types = len(gamma_combination)

    # 计算迁移率
    migration_rate = calculate_migration_rate(observed_data)

    # 构建参数字典（支持多维度类型特定参数）
    initial_params = {
        "alpha_w": 1.0, "lambda": 2.0, "alpha_home": 1.0,
        "rho_base_tier_1": 1.0, "rho_edu": 0.1, "rho_health": 0.1, "rho_house": 0.1,
        **{f"gamma_0_type_{i}": gamma_combination[i] for i in range(n_types)},
        # 添加其他类型特定参数
        **{f"gamma_1_type_{i}": -0.1 for i in range(n_types)},  # 示例参数
        **{f"alpha_home_type_{i}": 1.0 for i in range(n_types)}, # 示例参数
        **{f"lambda_type_{i}": 2.0 for i in range(n_types)},    # 示例参数
        "gamma_1": -0.1, "gamma_2": 0.2, "gamma_3": -0.4,
        "gamma_4": 0.01, "gamma_5": -0.05,
        "alpha_climate": 0.1, "alpha_health": 0.1,
        "alpha_education": 0.1, "alpha_public_services": 0.1,
        "n_choices": 31
    }

    if not verbose:
        # 禁用打印输出
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = run_em_algorithm(
                observed_data=observed_data,
                state_space=state_space,
                transition_matrices=transition_matrices,
                beta=beta,
                n_types=n_types,
                regions_df=regions_df,
                distance_matrix=distance_matrix,
                adjacency_matrix=adjacency_matrix,
                max_iterations=n_em_iterations,
                n_choices=31,
                use_migration_behavior_init=True,  # 使用迁移行为初始化
            )
    else:
        result = run_em_algorithm(
            observed_data=observed_data,
            state_space=state_space,
            transition_matrices=transition_matrices,
            beta=beta,
            n_types=n_types,
            regions_df=regions_df,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            max_iterations=n_em_iterations,
            n_choices=31,
            use_migration_behavior_init=True,  # 使用迁移行为初始化
        )

    # 提取评估指标
    posterior_probs = result['posterior_probs']
    log_likelihood = result['final_log_likelihood']
    n_params = len(initial_params)
    n_obs = len(observed_data)

    metrics = evaluate_type_separation_for_low_migration_rate(
        posterior_probs, log_likelihood, n_params, n_obs, migration_rate
    )

    # 添加gamma值信息
    for i, gamma_val in enumerate(gamma_combination):
        metrics[f'gamma_0_type_{i}'] = gamma_val

    # 检查退化（使用适应低迁移率的版本）
    is_degenerate, deg_msg = check_degeneracy_for_low_migration_data(posterior_probs, migration_rate)
    metrics['is_degenerate'] = is_degenerate
    metrics['degeneracy_message'] = deg_msg

    # 添加迁移率信息
    metrics['migration_rate'] = migration_rate

    return metrics


def stage1_coarse_search(
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: dict,
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_types: int = 3,
    granularity: str = 'sparse'
) -> pd.DataFrame:
    """
    阶段1：粗粒度网格搜索
    现在支持低迁移率数据的评估

    Args:
        granularity: 'sparse'(稀疏,~20组合), 'full'(完整,~120组合), 'fine'(精细,~500组合)
    """
    print("="*80)
    print(f"Stage 1: Coarse Grid Search ({granularity} mode)")
    print("="*80)

    # 计算迁移率
    migration_rate = calculate_migration_rate(observed_data)
    print(f"Data migration rate: {migration_rate:.3f}")
    print("="*80)

    # 生成参数网格
    gamma_combinations = generate_gamma_grid(n_types, granularity=granularity)

    # 如果是full或fine模式且组合数过多，给出警告
    if len(gamma_combinations) > 50:
        print(f"\n⚠️  Warning: {len(gamma_combinations)} combinations detected.")
        print(f"   Estimated time: ~{len(gamma_combinations) * 5 / 60:.1f} hours")
        print(f"   Consider using --granularity sparse for faster search.\n")

    results = []

    for idx, gamma_comb in enumerate(gamma_combinations):
        print(f"\n[{idx+1}/{len(gamma_combinations)}] Testing gamma_0 = {gamma_comb}")

        try:
            # 使用快速评估函数（只运行E步）
            metrics = evaluate_single_combination_fast(
                gamma_comb,
                observed_data, state_space, transition_matrices,
                beta, regions_df, distance_matrix, adjacency_matrix,
                verbose=False
            )
            results.append(metrics)

            # 打印关键指标
            print(f"  Balance score: {metrics['balance_score']:.3f}, "
                  f"Type probs: [{metrics['type_0_prob']:.2f}, "
                  f"{metrics['type_1_prob']:.2f}, {metrics['type_2_prob']:.2f}], "
                  f"BIC: {metrics['bic']:.1f}, Migration rate: {migration_rate:.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # 转为DataFrame
    results_df = pd.DataFrame(results)

    # 检查是否有有效结果
    if len(results_df) == 0:
        print("\n❌ No valid results obtained. All combinations failed.")
        print("   This may indicate:")
        print("   1. Numerical issues with the current parameter ranges")
        print("   2. Data loading problems")
        print("   3. Bellman equation convergence failures")
        return results_df

    # 保存结果
    output_path = 'results/param_search_stage1.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Stage 1 results saved to: {output_path}")

    # 显示top 5
    print("\n" + "="*80)
    print(f"Top 5 Parameter Combinations (by balance score) - Migration rate: {migration_rate:.3f}")
    print("="*80)

    n_results = min(5, len(results_df))
    top5 = results_df.nlargest(n_results, 'balance_score')
    for idx, row in top5.iterrows():
        print(f"\n{idx+1}. gamma_0 = [{row['gamma_0_type_0']:.1f}, "
              f"{row['gamma_0_type_1']:.1f}, {row['gamma_0_type_2']:.1f}]")
        print(f"   Balance score: {row['balance_score']:.3f}")
        print(f"   Type probs: [{row['type_0_prob']:.2f}, "
              f"{row['type_1_prob']:.2f}, {row['type_2_prob']:.2f}]")
        print(f"   BIC: {row['bic']:.1f}, LogLik: {row['log_likelihood']:.1f}")
        print(f"   Status: {row['degeneracy_message']}")

    return results_df


def stage2_fine_optimization(
    results_stage1: pd.DataFrame,
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: dict,
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    top_k: int = 5
) -> pd.DataFrame:
    """
    阶段2：对top candidates进行精细评估
    现在支持低迁移率数据的评估
    """
    print("\n" + "="*80)
    print("Stage 2: Fine-grained Optimization")
    print("="*80)

    # 计算迁移率
    migration_rate = calculate_migration_rate(observed_data)
    print(f"Data migration rate: {migration_rate:.3f}")
    print("="*80)

    # 选择top K个候选
    candidates = results_stage1.nlargest(top_k, 'balance_score')

    results = []

    for idx, row in candidates.iterrows():
        gamma_comb = (
            row['gamma_0_type_0'],
            row['gamma_0_type_1'],
            row['gamma_0_type_2']
        )

        print(f"\n[{idx+1}/{len(candidates)}] Full evaluation of gamma_0 = {gamma_comb}")

        try:
            metrics = evaluate_single_combination(
                gamma_comb,
                observed_data, state_space, transition_matrices,
                beta, regions_df, distance_matrix, adjacency_matrix,
                n_em_iterations=5,  # 运行5次EM迭代
                verbose=True  # 显示详细信息
            )
            results.append(metrics)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    results_df = pd.DataFrame(results)

    # 保存结果
    output_path = 'results/param_search_stage2.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Stage 2 results saved to: {output_path}")

    # 选择最优参数
    print("\n" + "="*80)
    print(f"FINAL RECOMMENDATION - Migration rate: {migration_rate:.3f}")
    print("="*80)

    # 综合评分：balance_score + BIC归一化
    results_df['composite_score'] = (
        results_df['balance_score'] -
        (results_df['bic'] - results_df['bic'].min()) / (results_df['bic'].max() - results_df['bic'].min()) * 0.3
    )

    best = results_df.loc[results_df['composite_score'].idxmax()]

    print(f"\n🎯 Recommended gamma_0 values:")
    print(f"   gamma_0_type_0 = {best['gamma_0_type_0']:.2f}")
    print(f"   gamma_0_type_1 = {best['gamma_0_type_1']:.2f}")
    print(f"   gamma_0_type_2 = {best['gamma_0_type_2']:.2f}")
    print(f"\n📊 Performance:")
    print(f"   Balance score: {best['balance_score']:.3f}")
    print(f"   Type distribution: [{best['type_0_prob']:.2f}, "
          f"{best['type_1_prob']:.2f}, {best['type_2_prob']:.2f}]")
    print(f"   BIC: {best['bic']:.1f}")
    print(f"   Migration rate: {migration_rate:.3f}")
    print(f"   Status: {best['degeneracy_message']}")

    # 保存推荐参数到JSON
    recommendation = {
        'gamma_0_type_0': float(best['gamma_0_type_0']),
        'gamma_0_type_1': float(best['gamma_0_type_1']),
        'gamma_0_type_2': float(best['gamma_0_type_2']),
        'migration_rate': migration_rate,
        'metrics': {
            'balance_score': float(best['balance_score']),
            'type_probs': [float(best['type_0_prob']), float(best['type_1_prob']), float(best['type_2_prob'])],
            'bic': float(best['bic']),
            'log_likelihood': float(best['log_likelihood']),
            'migration_rate': migration_rate
        },
        'timestamp': datetime.now().isoformat()
    }

    with open('results/recommended_gamma_0.json', 'w') as f:
        json.dump(recommendation, f, indent=2)

    print(f"\n✓ Recommendation saved to: results/recommended_gamma_0.json")

    return results_df


def quick_test(
    observed_data: pd.DataFrame,
    state_space: pd.DataFrame,
    transition_matrices: dict,
    beta: float,
    regions_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    adjacency_matrix: np.ndarray
):
    """
    快速测试几个典型组合
    """
    print("="*80)
    print("Quick Test Mode - Extreme Differentiation Strategy")
    print("="*80)

    test_combinations = [
        # 原始测试（已知会退化）
        # (0.5, 2.0, 5.0),   # 低-中-高
        # (0.1, 1.0, 10.0),  # 极低-中-极高
        # (1.0, 3.0, 5.0),   # 中-高-极高
        # (0.3, 1.5, 4.0),   # 低-中-高（温和）

        # 新策略：极端差异化参数（400-1000倍差距）
        (0.05, 1.0, 20.0),   # 极低-中-极高 (400倍)
        (0.1, 2.0, 10.0),    # 低-中-高 (100倍)
        (0.01, 0.5, 15.0),   # 超低-中低-超高 (1500倍)
        (0.02, 1.0, 30.0),   # 超低-中-超超高 (1500倍)
    ]

    print("\nTesting extreme parameter differentiation to avoid type degeneracy...")
    print("Hypothesis: Need >100x difference between min and max gamma_0\n")

    for idx, gamma_comb in enumerate(test_combinations, 1):
        ratio = max(gamma_comb) / min(gamma_comb)
        print(f"\n[{idx}/{len(test_combinations)}] Testing gamma_0 = {gamma_comb}")
        print(f"  Differentiation ratio: {ratio:.0f}x")

        metrics = evaluate_single_combination_fast(
            gamma_comb,
            observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix,
            verbose=True
        )

        print(f"  ✓ Balance: {metrics['balance_score']:.3f}, "
              f"Types: [{metrics['type_0_prob']:.2f}, {metrics['type_1_prob']:.2f}, {metrics['type_2_prob']:.2f}]")

        if metrics['balance_score'] > 0.5:
            print(f"  🎯 SUCCESS! Found balanced distribution!")


def main():
    parser = argparse.ArgumentParser(description='Search for optimal type-specific parameters')
    parser.add_argument('--stage', type=int, choices=[1, 2], help='Search stage (1=coarse, 2=fine)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--granularity', type=str, choices=['sparse', 'full', 'fine'],
                       default='sparse',
                       help='Grid granularity: sparse(~20,1.7h), full(~120,10h), fine(~500,40h)')
    args = parser.parse_args()

    # 加载数据
    print("Loading data...")
    config = ModelConfig()
    data_loader = DataLoader(config)

    distance_matrix = data_loader.load_distance_matrix()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    observed_data, regions_df, state_space, transition_matrices = \
        data_loader.create_estimation_dataset_and_state_space(simplified_state=True)

    beta = config.discount_factor

    print(f"Data loaded: {len(observed_data)} observations, {len(state_space)} states\n")

    # 确保结果目录存在
    Path('results').mkdir(exist_ok=True)

    if args.quick:
        quick_test(
            observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix
        )

    elif args.stage == 1:
        stage1_coarse_search(
            observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix,
            granularity=args.granularity
        )

    elif args.stage == 2:
        # 加载stage 1结果
        if not os.path.exists('results/param_search_stage1.csv'):
            print("Error: Stage 1 results not found. Run --stage 1 first.")
            return

        results_stage1 = pd.read_csv('results/param_search_stage1.csv')
        stage2_fine_optimization(
            results_stage1,
            observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix
        )

    else:
        # 自动运行两个阶段
        print("Running both stages automatically...\n")

        results_stage1 = stage1_coarse_search(
            observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix,
            granularity=args.granularity
        )

        stage2_fine_optimization(
            results_stage1,
            observed_data, state_space, transition_matrices,
            beta, regions_df, distance_matrix, adjacency_matrix
        )


if __name__ == '__main__':
    main()
