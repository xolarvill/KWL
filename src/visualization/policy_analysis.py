"""
政策分析图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from src.visualization.results_plot import plot_policy_dynamic_impact, plot_counterfactual_analysis

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
    'primary': '#0173B2',
    'secondary': '#DE8F05',
    'tertiary': '#029E73',
    'quaternary': '#CC78BC',
    'quinary': '#CA9161',
    'senary': '#ECE133',
    'error': '#D55E00',
    'neutral': '#949494',
}

PALETTE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#ECE133', '#D55E00', '#949494']


def plot_policy_comparison_efficiency(
    policy_scenarios: Optional[List[str]] = None,
    均衡度: Optional[np.array] = None,
    总效用: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制政策组合比较：引导人口流向的效率

    Args:
        policy_scenarios: 政策情景列表
        均衡度: 地区均衡度指标
        总效用: 总效用指标
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or policy_scenarios is None:
        policy_scenarios = ['仅放开一线城市户籍', '大力发展二线城市并放开户籍', '对迁往内陆省份的青年提供补贴']
        均衡度 = np.array([0.15, 0.35, 0.28])
        总效用 = np.array([0.08, 0.15, 0.12])

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor('white')

    # 使用不同颜色表示不同政策
    scatter = ax.scatter(均衡度, 总效用, s=250,
                        c=range(len(policy_scenarios)),
                        cmap='viridis', alpha=0.75,
                        edgecolors='white', linewidth=2)

    # 添加标签，优化位置
    for i, txt in enumerate(policy_scenarios):
        ax.annotate(txt, (均衡度[i], 总效用[i]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, ha='left',
                   bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('地区均衡度（人口基尼系数下降幅度）', fontweight='normal')
    ax.set_ylabel('全国总效用增幅', fontweight='normal')
    ax.set_title('政策效率前沿：均衡度与效用权衡', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 添加参考线
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', linewidth=0.8, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"政策组合效率比较图已保存到: {output_path}")

    return fig


def plot_policy_heterogeneous_effects(
    policy_labels: Optional[List[str]] = None,
    talent_loss: Optional[np.array] = None,
    city_crowding: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制政策的异质性效应——"一刀切" vs. "精准滴灌"

    Args:
        policy_labels: 政策标签
        talent_loss: 人才流失指标
        city_crowding: 城市拥挤指标
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or policy_labels is None:
        policy_labels = ['一刀切:\n全国范围内取消户籍限制', '精准滴灌:\n二线城市对高学历35岁以下开放']
        talent_loss = np.array([0.45, 0.25])
        city_crowding = np.array([0.65, 0.35])

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    fig.patch.set_facecolor('white')

    x = np.arange(len(policy_labels))
    width = 0.36

    bars1 = ax.bar(x - width/2, talent_loss, width,
                   label='中西部人才流失程度', alpha=0.85,
                   color=COLORS['error'])
    bars2 = ax.bar(x + width/2, city_crowding, width,
                   label='东部城市拥挤程度', alpha=0.85,
                   color=COLORS['tertiary'])

    ax.set_xlabel('政策类型', fontweight='normal')
    ax.set_ylabel('负面效应指标值', fontweight='normal')
    ax.set_title('政策异质性效应对比', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, fontsize=10)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"政策异质性效应图已保存到: {output_path}")

    return fig


def plot_unintended_consequences(
    years: Optional[np.array] = None,
    population: Optional[np.array] = None,
    wages: Optional[np.array] = None,
    output_path: Optional[str] = None,
    use_example: bool = True
) -> plt.Figure:
    """
    绘制政策的意外后果与反馈循环
    展示大规模补贴吸引劳动力迁往西部的政策效果

    Args:
        years: 时间序列
        population: 人口变化
        wages: 工资变化
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or years is None:
        years = np.arange(2020, 2031)
        base_pop = 100
        policy_impact_pop = np.concatenate([
            np.linspace(0, 15, 4),
            np.linspace(15, 12, 4),
            np.linspace(12, 8, 3)
        ])
        population = base_pop + policy_impact_pop

        base_wage = 50000
        policy_impact_wage = np.concatenate([
            np.linspace(0, -3000, 4),
            np.linspace(-3000, -5000, 4),
            np.linspace(-5000, -3000, 3)
        ])
        wages = base_wage + policy_impact_wage

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # 左侧Y轴：人口
    color1 = COLORS['primary']
    ax1.set_xlabel('年份', fontweight='normal')
    ax1.set_ylabel('西部地区人口数量 (百万)', color=color1, fontweight='normal')
    line1 = ax1.plot(years, population, label='人口数量',
                     color=color1, linewidth=2.5, marker='o',
                     markersize=7, alpha=0.9)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # 右侧Y轴：工资
    ax2 = ax1.twinx()
    color2 = COLORS['secondary']
    ax2.set_ylabel('平均工资 (元)', color=color2, fontweight='normal')
    line2 = ax2.plot(years, wages, label='平均工资',
                     color=color2, linewidth=2.5, marker='s',
                     markersize=7, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加政策实施标识
    policy_year = 2025
    ax1.axvline(x=policy_year, color=COLORS['neutral'],
                linestyle='--', alpha=0.6, linewidth=2, label='政策开始实施')

    # 添加阴影区域标识政策后时期
    ax1.axvspan(policy_year, years[-1], alpha=0.08,
                color=COLORS['neutral'], label='政策实施期')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', frameon=False)

    ax1.set_title('政策意外后果：劳动力供给增加对工资的反馈效应',
                 fontweight='bold', pad=15)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"政策意外后果图已保存到: {output_path}")

    return fig