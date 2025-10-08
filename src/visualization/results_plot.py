"""
可视化模块：结果图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from src.utils.example_generator import ExampleResultGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_estimation_results(
    params: Optional[Dict[str, float]] = None,
    std_errors: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制参数估计结果图表
    
    Args:
        params: 参数估计结果
        std_errors: 标准误
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or params is None:
        generator = ExampleResultGenerator()
        params = generator.example_params
        std_errors = generator.generate_example_standard_errors()
    
    # 按参数类型分组
    amenity_params = {k: v for k, v in params.items() if k.startswith('alpha_') and 'home' not in k}
    core_params = {k: v for k, v in params.items() if k in ['alpha_w', 'lambda', 'alpha_home']}
    migration_params = {k: v for k, v in params.items() if k.startswith('gamma')}
    hukou_params = {k: v for k, v in params.items() if k.startswith('rho')}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    
    # 地区舒适度参数
    ax1 = axes[0, 0]
    param_names = list(amenity_params.keys())
    values = [amenity_params[k] for k in param_names]
    # 确保误差值为非负
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)
    
    ax1.barh(range(len(param_names)), values, xerr=errors, alpha=0.7)
    ax1.set_yticks(range(len(param_names)))
    ax1.set_yticklabels(param_names)
    ax1.set_xlabel('参数值')

    ax1.grid(True, alpha=0.3)
    
    # 核心机制参数
    ax2 = axes[0, 1]
    param_names = list(core_params.keys())
    values = [core_params[k] for k in param_names]
    # 确保误差值为非负
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)
    
    ax2.barh(range(len(param_names)), values, xerr=errors, alpha=0.7, color='orange')
    ax2.set_yticks(range(len(param_names)))
    ax2.set_yticklabels(param_names)
    ax2.set_xlabel('参数值')

    ax2.grid(True, alpha=0.3)
    
    # 迁移成本参数
    ax3 = axes[1, 0]
    param_names = list(migration_params.keys())
    values = [migration_params[k] for k in param_names]
    # 确保误差值为非负
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)
    
    ax3.barh(range(len(param_names)), values, xerr=errors, alpha=0.7, color='green')
    ax3.set_yticks(range(len(param_names)))
    ax3.set_yticklabels(param_names)
    ax3.set_xlabel('参数值')

    ax3.grid(True, alpha=0.3)
    
    # 户籍惩罚参数
    ax4 = axes[1, 1]
    param_names = list(hukou_params.keys())
    values = [hukou_params[k] for k in param_names]
    # 确保误差值为非负
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)
    
    ax4.barh(range(len(param_names)), values, xerr=errors, alpha=0.7, color='red')
    ax4.set_yticks(range(len(param_names)))
    ax4.set_yticklabels(param_names)
    ax4.set_xlabel('参数值')

    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"参数估计结果图已保存到: {output_path}")
    
    return fig


def plot_model_fit_flow(
    actual_flow: Optional[np.array] = None,
    predicted_flow: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制模型拟合：实际与预测的迁移流量
    
    Args:
        actual_flow: 实际迁移流量
        predicted_flow: 预测迁移流量
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or actual_flow is None:
        generator = ExampleResultGenerator()
        df_flow, _, _ = generator.generate_example_predictions()
        actual_flow = df_flow['actual_flow']
        predicted_flow = df_flow['predicted_flow']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.scatter(np.log(actual_flow), np.log(predicted_flow), alpha=0.6)
    
    # 添加45度线
    min_val = min(np.log(actual_flow).min(), np.log(predicted_flow).min())
    max_val = max(np.log(actual_flow).max(), np.log(predicted_flow).max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='45度线')
    
    ax.set_xlabel('实际迁移流量 (对数)')
    ax.set_ylabel('预测迁移流量 (对数)')

    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"迁移流量拟合图已保存到: {output_path}")
    
    return fig


def plot_migration_by_groups(
    df_age: Optional[pd.DataFrame] = None,
    df_edu: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制关键群体的迁移率对比图
    
    Args:
        df_age: 按年龄段分组的数据
        df_edu: 按教育程度分组的数据
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or df_age is None:
        generator = ExampleResultGenerator()
        _, df_age, df_edu = generator.generate_example_predictions()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    
    # 按年龄段对比
    ax1 = axes[0]
    age_actual = df_age.groupby('group')['actual_rate'].mean()
    age_predicted = df_age.groupby('group')['predicted_rate'].mean()
    x = np.arange(len(age_actual))
    width = 0.35
    
    ax1.bar(x - width/2, age_actual.values, width, label='实际迁移率', alpha=0.8)
    ax1.bar(x + width/2, age_predicted.values, width, label='预测迁移率', alpha=0.8)
    ax1.set_xlabel('年龄段')
    ax1.set_ylabel('迁移率')

    ax1.set_xticks(x)
    ax1.set_xticklabels(age_actual.index)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 按学历对比
    ax2 = axes[1]
    edu_actual = df_edu.groupby('group')['actual_rate'].mean()
    edu_predicted = df_edu.groupby('group')['predicted_rate'].mean()
    x = np.arange(len(edu_actual))
    
    ax2.bar(x - width/2, edu_actual.values, width, label='实际迁移率', alpha=0.8)
    ax2.bar(x + width/2, edu_predicted.values, width, label='预测迁移率', alpha=0.8)
    ax2.set_xlabel('学历')
    ax2.set_ylabel('迁移率')

    ax2.set_xticks(x)
    ax2.set_xticklabels(edu_actual.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"分组迁移率对比图已保存到: {output_path}")
    
    return fig


def plot_abm_zipf_law(
    real_data: Optional[np.array] = None,
    sim_data: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制ABM模型的Zipf定律验证图
    
    Args:
        real_data: 真实数据
        sim_data: 模拟数据
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or real_data is None:
        generator = ExampleResultGenerator()
        abm_results = generator.generate_example_abm_results()
        # 使用排序后的人口数据来模拟Zipf分布
        real_data = sorted(abm_results['baseline_population'], reverse=True)
        sim_data = sorted(abm_results['policy_population'], reverse=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ranks = np.arange(1, len(real_data) + 1)
    
    ax.loglog(ranks, real_data, 'b-', label='真实数据', linewidth=2)
    ax.loglog(ranks, sim_data, 'r--', label='ABM模拟结果', linewidth=2)
    
    ax.set_xlabel('城市排名 (对数)')
    ax.set_ylabel('城市规模 (对数)')

    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Zipf定律图已保存到: {output_path}")
    
    return fig


def plot_policy_dynamic_impact(
    years: Optional[np.array] = None,
    baseline: Optional[np.array] = None,
    policy: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制政策动态影响图
    
    Args:
        years: 时间序列
        baseline: 基准路径
        policy: 政策路径
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or years is None:
        generator = ExampleResultGenerator()
        abm_results = generator.generate_example_abm_results()
        years = abm_results['time_series_years']
        baseline = abm_results['baseline_population']
        policy = abm_results['policy_population']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(years, baseline, label='基准路径', linewidth=2, marker='o')
    ax.plot(years, policy, label='反事实路径（2025年起全面放开户籍）', linewidth=2, marker='s')
    
    # 添加置信区间
    ax.fill_between(years, policy-5, policy+5, alpha=0.2, label='模拟置信区间')
    
    ax.set_xlabel('年份')
    ax.set_ylabel('一线城市总人口')

    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"政策动态影响图已保存到: {output_path}")
    
    return fig


def plot_feature_importance(
    feature_names: Optional[List[str]] = None,
    importance: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名称
        importance: 重要性值
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or feature_names is None:
        generator = ExampleResultGenerator()
        # 这里我们模拟特征重要性数据
        feature_names = ['年龄', '教育', '户籍地', '当前所在地', '收入', '健康', '公共服务', '气候']
        importance = np.random.uniform(0.05, 0.25, len(feature_names))
        importance = np.sort(importance)[::-1]  # 降序排列
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.barh(range(len(feature_names)), importance)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('重要性得分')

    ax.invert_yaxis()  # 重要性从高到低排列
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {output_path}")
    
    return fig


def plot_ml_performance(
    df_performance: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制机器学习性能对比图
    
    Args:
        df_performance: 性能数据框
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or df_performance is None:
        generator = ExampleResultGenerator()
        df_performance = generator.generate_example_ml_performance()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    
    metrics = ['RMSE', 'MAE', 'R²']
    colors = ['blue', 'orange', 'green']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        
        bars = ax.bar(df_performance['Model'], df_performance[metric], color=color, alpha=0.7)
    
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, df_performance[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"机器学习性能对比图已保存到: {output_path}")
    
    return fig


def plot_counterfactual_analysis(
    policy_results: Optional[Dict] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制反事实分析结果
    
    Args:
        policy_results: 政策结果字典
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or policy_results is None:
        generator = ExampleResultGenerator()
        abm_results = generator.generate_example_abm_results()
        policy_results = abm_results['policy_comparison']
    
    # 将字典转换为DataFrame
    policies = list(policy_results.keys())
    indicators = list(policy_results[policies[0]].keys())
    
    df = pd.DataFrame({
        indicator: [policy_results[policy][indicator] for policy in policies]
        for indicator in indicators
    })
    df.index = policies
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('政策情景')
    ax.set_ylabel('指标值')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"反事实分析结果图已保存到: {output_path}")
    
    return fig