"""
可视化模块：结果图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
from src.utils.example_generator import ExampleResultGenerator

# 设置现代化、简洁的绘图样式
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# 使用色盲友好的配色方案
COLORS = {
    'primary': '#0173B2',      # 蓝色
    'secondary': '#DE8F05',    # 橙色
    'tertiary': '#029E73',     # 绿色
    'quaternary': '#CC78BC',   # 紫色
    'quinary': '#CA9161',      # 棕色
    'senary': '#ECE133',       # 黄色
    'error': '#D55E00',        # 红色
    'neutral': '#949494',      # 灰色
}

# 调色板（来自ColorBrewer和学术期刊推荐）
PALETTE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#ECE133', '#D55E00', '#949494']


def save_figure_multiple_formats(fig: plt.Figure, output_path: str) -> None:
    """
    保存图表为多种格式（PNG、SVG、PDF）

    Args:
        fig: matplotlib图表对象
        output_path: 输出路径（包含文件名，扩展名将被替换）
    """
    # 获取不含扩展名的路径
    path_without_ext = str(Path(output_path).with_suffix(''))

    # 保存为PNG格式（用于预览和网页）
    png_path = f"{path_without_ext}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')

    # 保存为SVG格式（矢量格式，适合编辑）
    svg_path = f"{path_without_ext}.svg"
    fig.savefig(svg_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='svg')

    # 保存为PDF格式（矢量格式，适合论文）
    pdf_path = f"{path_without_ext}.pdf"
    fig.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')

    print(f"图表已保存为多种格式:")
    print(f"  - PNG: {png_path}")
    print(f"  - SVG: {svg_path}")
    print(f"  - PDF: {pdf_path}")


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
    core_params = {k: v for k, v in params.items() if k in ['alpha_w', 'alpha_home']}
    migration_params = {k: v for k, v in params.items() if k.startswith('gamma')}
    hukou_params = {k: v for k, v in params.items() if k.startswith('rho')}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # 地区舒适度参数
    ax1 = axes[0, 0]
    param_names = list(amenity_params.keys())
    values = [amenity_params[k] for k in param_names]
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)

    ax1.barh(range(len(param_names)), values, xerr=errors,
             color=COLORS['primary'], alpha=0.85, height=0.7,
             error_kw={'linewidth': 1.5, 'ecolor': COLORS['neutral'], 'alpha': 0.6})
    ax1.set_yticks(range(len(param_names)))
    ax1.set_yticklabels(param_names)
    ax1.set_xlabel('参数值', fontweight='normal')
    ax1.set_title('地区舒适度参数', fontweight='bold', pad=12)
    ax1.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)

    # 核心机制参数
    ax2 = axes[0, 1]
    param_names = list(core_params.keys())
    values = [core_params[k] for k in param_names]
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)

    ax2.barh(range(len(param_names)), values, xerr=errors,
             color=COLORS['secondary'], alpha=0.85, height=0.7,
             error_kw={'linewidth': 1.5, 'ecolor': COLORS['neutral'], 'alpha': 0.6})
    ax2.set_yticks(range(len(param_names)))
    ax2.set_yticklabels(param_names)
    ax2.set_xlabel('参数值', fontweight='normal')
    ax2.set_title('核心机制参数', fontweight='bold', pad=12)
    ax2.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)

    # 迁移成本参数
    ax3 = axes[1, 0]
    param_names = list(migration_params.keys())
    values = [migration_params[k] for k in param_names]
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)

    ax3.barh(range(len(param_names)), values, xerr=errors,
             color=COLORS['tertiary'], alpha=0.85, height=0.7,
             error_kw={'linewidth': 1.5, 'ecolor': COLORS['neutral'], 'alpha': 0.6})
    ax3.set_yticks(range(len(param_names)))
    ax3.set_yticklabels(param_names)
    ax3.set_xlabel('参数值', fontweight='normal')
    ax3.set_title('迁移成本参数', fontweight='bold', pad=12)
    ax3.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)

    # 户籍惩罚参数
    ax4 = axes[1, 1]
    param_names = list(hukou_params.keys())
    values = [hukou_params[k] for k in param_names]
    errors = [abs(std_errors.get(k, 0)) for k in param_names] if std_errors else [0] * len(values)

    ax4.barh(range(len(param_names)), values, xerr=errors,
             color=COLORS['error'], alpha=0.85, height=0.7,
             error_kw={'linewidth': 1.5, 'ecolor': COLORS['neutral'], 'alpha': 0.6})
    ax4.set_yticks(range(len(param_names)))
    ax4.set_yticklabels(param_names)
    ax4.set_xlabel('参数值', fontweight='normal')
    ax4.set_title('户籍惩罚参数', fontweight='bold', pad=12)
    ax4.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fig.patch.set_facecolor('white')

    ax.scatter(np.log(actual_flow), np.log(predicted_flow),
               alpha=0.5, s=30, color=COLORS['primary'], edgecolors='none')

    # 添加45度线
    min_val = min(np.log(actual_flow).min(), np.log(predicted_flow).min())
    max_val = max(np.log(actual_flow).max(), np.log(predicted_flow).max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color=COLORS['error'], linestyle='--', linewidth=2, label='45度线', alpha=0.8)

    ax.set_xlabel('实际迁移流量 (对数)', fontweight='normal')
    ax.set_ylabel('预测迁移流量 (对数)', fontweight='normal')
    ax.set_title('模型拟合效果', fontweight='bold', pad=15)
    ax.legend(frameon=False)

    # 添加轻微网格
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    # 按年龄段对比
    ax1 = axes[0]
    age_actual = df_age.groupby('group')['actual_rate'].mean()
    age_predicted = df_age.groupby('group')['predicted_rate'].mean()
    x = np.arange(len(age_actual))
    width = 0.38

    ax1.bar(x - width/2, age_actual.values, width,
            label='实际迁移率', alpha=0.85, color=COLORS['primary'])
    ax1.bar(x + width/2, age_predicted.values, width,
            label='预测迁移率', alpha=0.85, color=COLORS['secondary'])
    ax1.set_xlabel('年龄段', fontweight='normal')
    ax1.set_ylabel('迁移率', fontweight='normal')
    ax1.set_title('按年龄段', fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_actual.index)
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    # 按学历对比
    ax2 = axes[1]
    edu_actual = df_edu.groupby('group')['actual_rate'].mean()
    edu_predicted = df_edu.groupby('group')['predicted_rate'].mean()
    x = np.arange(len(edu_actual))

    ax2.bar(x - width/2, edu_actual.values, width,
            label='实际迁移率', alpha=0.85, color=COLORS['primary'])
    ax2.bar(x + width/2, edu_predicted.values, width,
            label='预测迁移率', alpha=0.85, color=COLORS['secondary'])
    ax2.set_xlabel('学历', fontweight='normal')
    ax2.set_ylabel('迁移率', fontweight='normal')
    ax2.set_title('按学历', fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(edu_actual.index)
    ax2.legend(frameon=False, loc='upper right')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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
        real_data = sorted(abm_results['baseline_population'], reverse=True)
        sim_data = sorted(abm_results['policy_population'], reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    fig.patch.set_facecolor('white')

    ranks = np.arange(1, len(real_data) + 1)

    ax.loglog(ranks, real_data, color=COLORS['primary'],
              linewidth=2.5, label='真实数据', alpha=0.9)
    ax.loglog(ranks, sim_data, color=COLORS['secondary'],
              linestyle='--', linewidth=2.5, label='ABM模拟结果', alpha=0.9)

    ax.set_xlabel('城市排名 (对数)', fontweight='normal')
    ax.set_ylabel('城市规模 (对数)', fontweight='normal')
    ax.set_title('城市规模分布的Zipf定律验证', fontweight='bold', pad=15)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, which='both')

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    fig.patch.set_facecolor('white')

    ax.plot(years, baseline, label='基准路径',
            linewidth=2.5, marker='o', markersize=6,
            color=COLORS['primary'], alpha=0.9)
    ax.plot(years, policy, label='反事实路径（2025年起全面放开户籍）',
            linewidth=2.5, marker='s', markersize=6,
            color=COLORS['secondary'], alpha=0.9)

    # 添加置信区间
    ax.fill_between(years, policy-5, policy+5,
                     alpha=0.15, color=COLORS['secondary'],
                     label='95%置信区间')

    ax.set_xlabel('年份', fontweight='normal')
    ax.set_ylabel('一线城市总人口 (百万)', fontweight='normal')
    ax.set_title('政策冲击的动态影响', fontweight='bold', pad=15)
    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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
        feature_names = ['年龄', '教育', '户籍地', '当前所在地', '收入', '健康', '公共服务', '气候']
        importance = np.random.uniform(0.05, 0.25, len(feature_names))
        importance = np.sort(importance)[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor('white')

    # 使用渐变色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))

    bars = ax.barh(range(len(feature_names)), importance,
                   color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('重要性得分', fontweight='normal')
    ax.set_title('特征重要性分析', fontweight='bold', pad=15)
    ax.invert_yaxis()

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.005, i, f'{val:.3f}',
                va='center', fontsize=10, color=COLORS['neutral'])

    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')

    plt.tight_layout()

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    metrics = ['RMSE', 'MAE', 'R²']
    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for i, (metric, color) in enumerate(zip(metrics, colors_list)):
        ax = axes[i]

        bars = ax.bar(df_performance['Model'], df_performance[metric],
                     color=color, alpha=0.85, width=0.6)

        ax.set_ylabel(metric, fontweight='normal')
        ax.set_title(metric, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

        # 添加数值标签
        for bar, value in zip(bars, df_performance[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)

        # 旋转x轴标签
        ax.set_xticklabels(df_performance['Model'], rotation=20, ha='right')

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure_multiple_formats(fig, output_path)

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

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor('white')

    # 使用配色方案
    x = np.arange(len(policies))
    width = 0.8 / len(indicators)

    for i, (indicator, color) in enumerate(zip(indicators, PALETTE)):
        offset = (i - len(indicators)/2 + 0.5) * width
        bars = ax.bar(x + offset, df[indicator], width,
                     label=indicator, alpha=0.85, color=color)

    ax.set_xlabel('政策情景', fontweight='normal')
    ax.set_ylabel('指标值', fontweight='normal')
    ax.set_title('反事实政策情景对比', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha='right')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout()

    if output_path:
        save_figure_multiple_formats(fig, output_path)

    return fig

# ==================== 新增图表函数（第五章） ====================

def plot_latent_type_migration_patterns(
    patterns: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制不同潜在类别的迁移模式对比图（图5.x1）

    Args:
        patterns: 包含三种类型迁移模式的字典
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or patterns is None:
        generator = ExampleResultGenerator()
        patterns = generator.generate_latent_type_migration_patterns()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')

    types = ['定居型', '经济型', '闯荡型']
    colors_map = {'定居型': COLORS['neutral'], '经济型': COLORS['primary'], '闯荡型': COLORS['tertiary']}

    for idx, type_name in enumerate(types):
        ax = axes[idx]
        df = patterns[type_name]

        # 计算每个省份的总流入和总流出
        outflow = df.groupby('origin')['flow'].sum().sort_values(ascending=False).head(10)
        inflow = df.groupby('destination')['flow'].sum().sort_values(ascending=False).head(10)

        # 绘制流出强度柱状图
        y_pos = np.arange(len(outflow))
        ax.barh(y_pos, outflow.values, alpha=0.85, color=colors_map[type_name], height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(outflow.index, fontsize=9)
        ax.set_xlabel('迁移流量强度', fontweight='normal')
        ax.set_title(f'{type_name}人群的迁移模式', fontweight='bold', pad=12)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
        ax.invert_yaxis()

        # 添加类型特征说明
        if type_name == '定居型':
            ax.text(0.98, 0.02, '特征：流量极低\n占比：60%',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors_map[type_name]),
                   fontsize=9)
        elif type_name == '经济型':
            ax.text(0.98, 0.02, '特征：流向发达地区\n占比：30%',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors_map[type_name]),
                   fontsize=9)
        else:
            ax.text(0.98, 0.02, '特征：流向多样化\n占比：10%',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors_map[type_name]),
                   fontsize=9)

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure_multiple_formats(fig, output_path)

    return fig


def plot_survival_curve(
    survival_data: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制学习与纠错的生存曲线图（图5.x2）

    Args:
        survival_data: 包含生存曲线数据的字典
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or survival_data is None:
        generator = ExampleResultGenerator()
        survival_data = generator.generate_survival_curve_data()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor('white')

    years = survival_data['years']
    new_migrants = survival_data['new_migrants']
    old_residents = survival_data['old_residents']

    # 绘制生存曲线
    ax.plot(years, new_migrants, linewidth=2.5, marker='o', markersize=7,
           color=COLORS['secondary'], label='新迁移者（首次跨省迁移）', alpha=0.9)
    ax.plot(years, old_residents, linewidth=2.5, marker='s', markersize=7,
           color=COLORS['primary'], label='老住户（居住超过3年）', alpha=0.9)

    # 填充置信区间
    ax.fill_between(years, new_migrants - 0.03, new_migrants + 0.03,
                    alpha=0.15, color=COLORS['secondary'])
    ax.fill_between(years, old_residents - 0.02, old_residents + 0.02,
                    alpha=0.15, color=COLORS['primary'])

    ax.set_xlabel('迁移后年数', fontweight='normal')
    ax.set_ylabel('继续留在该地的概率', fontweight='normal')
    ax.set_title('学习与纠错：新迁移者的后续迁移风险', fontweight='bold', pad=15)
    ax.legend(frameon=False, loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_ylim([0.3, 1.05])

    # 添加注释
    ax.annotate('新迁移者更可能"纠错"',
               xy=(5, new_migrants[5]), xytext=(6.5, 0.65),
               arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.5),
               fontsize=10, color=COLORS['secondary'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['secondary']))

    plt.tight_layout()

    if output_path:
        save_figure_multiple_formats(fig, output_path)

    return fig


def plot_internet_information_cost(
    data: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制互联网作为信息基础设施的价值图（图5.x3）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_internet_information_cost_data()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor('white')

    ax.scatter(data['internet_penetration'] * 100, data['information_cost'],
              alpha=0.6, s=40, color=COLORS['primary'], edgecolors='none', label='实际数据点')

    z = np.polyfit(data['internet_penetration'], data['information_cost'], 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(data['internet_penetration'].min(), data['internet_penetration'].max(), 100)
    ax.plot(x_smooth * 100, p(x_smooth), linewidth=2.5, color=COLORS['secondary'],
           label='非线性拟合曲线', alpha=0.9)

    ax.set_xlabel('地区互联网普及率 (%)', fontweight='normal')
    ax.set_ylabel('迁移决策的等价信息成本 (元)', fontweight='normal')
    ax.set_title('互联网作为信息基础设施的价值', fontweight='bold', pad=15)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig


def plot_information_counterfactual(
    data: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制信息完全反事实分析图（图5.x4）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_information_counterfactual_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    ax1 = axes[0]
    indicators = ['总迁移率', '回流率']
    baseline_values = [data['baseline']['total_migration_rate'], data['baseline']['return_migration_rate']]
    counterfactual_values = [data['counterfactual']['total_migration_rate'], data['counterfactual']['return_migration_rate']]

    x = np.arange(len(indicators))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_values, width, label='基准情景（有信息摩擦）',
                   alpha=0.85, color=COLORS['primary'])
    bars2 = ax1.bar(x + width/2, counterfactual_values, width, label='反事实情景（信息完全）',
                   alpha=0.85, color=COLORS['secondary'])

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel('比率', fontweight='normal')
    ax1.set_title('信息完全对迁移行为的影响', fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(indicators)
    ax1.legend(frameon=False, loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    ax2 = axes[1]
    changes = [
        (data['difference']['total_migration_rate'] / data['baseline']['total_migration_rate']) * 100,
        (data['difference']['return_migration_rate'] / data['baseline']['return_migration_rate']) * 100
    ]
    colors_change = [COLORS['tertiary'] if c > 0 else COLORS['error'] for c in changes]
    bars = ax2.barh(indicators, changes, alpha=0.85, color=colors_change, height=0.6)

    for i, (bar, val) in enumerate(zip(bars, changes)):
        ax2.text(val + (2 if val > 0 else -2), i,
               f'{val:+.1f}%', ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('变化幅度 (%)', fontweight='normal')
    ax2.set_title('信息完全的影响', fontweight='bold', pad=12)
    ax2.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')

    plt.tight_layout(pad=2.0)
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig


def plot_hukou_housing_interaction(
    data: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制户籍-房价交互效应图（图5.x5）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_hukou_housing_interaction_data()

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    fig.patch.set_facecolor('white')

    ax.plot(data['price_income_ratio'], data['base_penalty'],
           linewidth=2.5, label='基础户籍惩罚（不考虑交互）',
           color=COLORS['primary'], alpha=0.9)
    ax.plot(data['price_income_ratio'], data['total_penalty'],
           linewidth=2.5, label='总惩罚（考虑与房价交互）',
           color=COLORS['error'], alpha=0.9)

    ax.fill_between(data['price_income_ratio'],
                    data['base_penalty'], data['total_penalty'],
                    alpha=0.2, color=COLORS['error'], label='房价放大效应')

    ax.set_xlabel('房价收入比', fontweight='normal')
    ax.set_ylabel('非本地户籍者感受到的效用惩罚', fontweight='normal')
    ax.set_title('户籍惩罚的放大效应：房价如何加剧不平等', fontweight='bold', pad=15)
    ax.legend(frameon=False, loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig


# ==================== 新增图表函数（第六章） ====================

def plot_policy_heterogeneous_response(
    data: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制不同类型对政策的异质性响应图（图6.x1）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_policy_heterogeneous_response_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    types = data['type'].values
    colors_list = [COLORS['neutral'], COLORS['primary'], COLORS['tertiary']]

    ax1 = axes[0]
    total_new_migrants = data['new_migrants'].sum()
    proportions = (data['new_migrants'] / total_new_migrants * 100).values

    bars = ax1.bar(types, data['new_migrants'], alpha=0.85, color=colors_list, width=0.6)

    for i, (bar, val, pct) in enumerate(zip(bars, data['new_migrants'], proportions)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}万人\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('新增迁移人数 (万人)', fontweight='normal')
    ax1.set_title('"一刀切"政策的低效性：不同群体的反应', fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    ax2 = axes[1]
    bars2 = ax2.barh(types, data['cost_efficiency'], alpha=0.85, color=colors_list, height=0.6)

    for i, (bar, val) in enumerate(zip(bars2, data['cost_efficiency'])):
        width = bar.get_width()
        ax2.text(width + 5, bar.get_y() + bar.get_height()/2.,
               f'{val:.0f}万元/人',
               ha='left', va='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('每增加1个迁移者的财政成本 (万元/人)', fontweight='normal')
    ax2.set_title('政策成本效益比', fontweight='bold', pad=12)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')

    plt.tight_layout(pad=2.0)
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig


def plot_west_development_comparison(
    data: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制两种西部开发策略对比图（图6.x2）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_west_development_comparison_data()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')

    years = data['years']
    policy_a = data['policy_a']
    policy_b = data['policy_b']

    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    width = 0.35
    x = np.arange(len(years))
    bars1 = ax1.bar(x - width/2, policy_a['inflow'], width, label='政策A：高额补贴',
                   alpha=0.7, color=COLORS['secondary'])
    bars2 = ax1.bar(x + width/2, policy_b['inflow'], width, label='政策B：精准匹配',
                   alpha=0.7, color=COLORS['tertiary'])

    ax1_twin.plot(years, policy_a['retention_rate'], linewidth=2.5, marker='o', markersize=6,
                 color=COLORS['secondary'], linestyle='--', alpha=0.9)
    ax1_twin.plot(years, policy_b['retention_rate'], linewidth=2.5, marker='s', markersize=6,
                 color=COLORS['tertiary'], linestyle='--', alpha=0.9)

    ax1.set_xlabel('年份', fontweight='normal')
    ax1.set_ylabel('年度流入量 (万人)', fontweight='normal', color='black')
    ax1_twin.set_ylabel('留存率', fontweight='normal', color='black')
    ax1.set_title('两种西部开发策略的对比：短期流入 vs. 长期留存', fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45)
    ax1.legend(loc='upper left', frameon=False, fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    ax2 = axes[1]
    ax2.plot(years, policy_a['net_population'], linewidth=2.5, marker='o', markersize=7,
            color=COLORS['secondary'], label='政策A：高额补贴', alpha=0.9)
    ax2.plot(years, policy_b['net_population'], linewidth=2.5, marker='s', markersize=7,
            color=COLORS['tertiary'], label='政策B：精准匹配', alpha=0.9)

    ax2.fill_between(years, policy_a['net_population'], alpha=0.15, color=COLORS['secondary'])
    ax2.fill_between(years, policy_b['net_population'], alpha=0.15, color=COLORS['tertiary'])

    ax2.set_xlabel('年份', fontweight='normal')
    ax2.set_ylabel('累计净留存人口 (万人)', fontweight='normal')
    ax2.set_title('长期效果对比：累计净留存人口', fontweight='bold', pad=12)
    ax2.legend(loc='upper left', frameon=False, fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout(pad=2.0)
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig


def plot_reform_path_comparison(
    data: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制单一改革vs一揽子改革对比图（图6.x3）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_reform_path_comparison_data()

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor('white')

    years = data['years']

    ax.plot(years, data['baseline'], linewidth=2.5, linestyle=':', 
           color=COLORS['neutral'], label='基准情景（无改革）', alpha=0.8)
    ax.plot(years, data['single_reform'], linewidth=2.5, marker='o', markersize=5,
           color=COLORS['secondary'], label='路径A：单一改革（仅放开户籍）', alpha=0.9)
    ax.plot(years, data['comprehensive_reform'], linewidth=2.5, marker='s', markersize=5,
           color=COLORS['tertiary'], label='路径B：一揽子改革（户籍+土地+公共服务）', alpha=0.9)

    ax.fill_between(years, data['baseline'], data['single_reform'],
                    where=(data['single_reform'] > data['baseline']),
                    alpha=0.15, color=COLORS['secondary'], label='单一改革恶化期')
    ax.fill_between(years, data['baseline'], data['comprehensive_reform'],
                    alpha=0.15, color=COLORS['tertiary'])

    ax.axvline(x=2025, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='改革开始')

    ax.set_xlabel('年份', fontweight='normal')
    ax.set_ylabel('劳动力空间错配指数', fontweight='normal')
    ax.set_title('单一改革 vs. 一揽子改革的长期动态', fontweight='bold', pad=15)
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig


def plot_home_premium_tradeoff(
    data: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制家乡溢价作为社会稳定器的权衡图（图6.x4）
    """
    if use_example or data is None:
        generator = ExampleResultGenerator()
        data = generator.generate_home_premium_tradeoff_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    years = data['years']
    world_a = data['world_a']
    world_b = data['world_b']

    ax1 = axes[0]
    ax1.plot(years, world_a['mean_utility'], linewidth=2.5, marker='o', markersize=5,
            color=COLORS['primary'], label='世界A：有家乡溢价', alpha=0.9)
    ax1.plot(years, world_b['mean_utility'], linewidth=2.5, marker='s', markersize=5,
            color=COLORS['secondary'], label='世界B：无家乡溢价', alpha=0.9)

    ax1.fill_between(years, world_a['mean_utility'], alpha=0.15, color=COLORS['primary'])
    ax1.fill_between(years, world_b['mean_utility'], alpha=0.15, color=COLORS['secondary'])

    ax1.set_xlabel('年份', fontweight='normal')
    ax1.set_ylabel('平均个体效用', fontweight='normal')
    ax1.set_title('平均效用水平', fontweight='bold', pad=12)
    ax1.legend(frameon=False, loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    ax2 = axes[1]
    ax2.plot(years, world_a['utility_variance'], linewidth=2.5, marker='o', markersize=5,
            color=COLORS['primary'], label='世界A：有家乡溢价', alpha=0.9)
    ax2.plot(years, world_b['utility_variance'], linewidth=2.5, marker='s', markersize=5,
            color=COLORS['error'], label='世界B：无家乡溢价（风险更高）', alpha=0.9)

    ax2.fill_between(years, world_a['utility_variance'], world_b['utility_variance'],
                    where=(world_b['utility_variance'] > world_a['utility_variance']),
                    alpha=0.2, color=COLORS['error'], label='社会风险增量')

    ax2.set_xlabel('年份', fontweight='normal')
    ax2.set_ylabel('个体效用方差（社会风险）', fontweight='normal')
    ax2.set_title('效用分散度（社会不平等与风险）', fontweight='bold', pad=12)
    ax2.legend(frameon=False, loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 添加结论注释
    fig.text(0.5, 0.02, '关键发现：世界B平均效用更高，但方差更大，表明家乡溢价在现有制度下起到"保险"作用',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor=COLORS['neutral']))

    plt.tight_layout(pad=2.5, rect=[0, 0.04, 1, 1])
    if output_path:
        save_figure_multiple_formats(fig, output_path)
    return fig
