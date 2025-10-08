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
        # 生成示例数据
        policy_scenarios = ['仅放开一线城市户籍', '大力发展二线城市并放开户籍', '对迁往内陆省份的青年提供补贴']
        均衡度 = np.array([0.15, 0.35, 0.28])  # 人口基尼系数的下降幅度
        总效用 = np.array([0.08, 0.15, 0.12])  # 总产出的增幅
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(均衡度, 总效用, s=100, c=range(len(policy_scenarios)), cmap='viridis', alpha=0.7)
    
    # 添加标签
    for i, txt in enumerate(policy_scenarios):
        ax.annotate(txt, (均衡度[i], 总效用[i]), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('地区均衡度 (人口基尼系数下降幅度)')
    ax.set_ylabel('全国总效用增幅')

    ax.grid(True, alpha=0.3)
    
    # 添加颜色条
    plt.colorbar(scatter)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    绘制政策的异质性效应——“一刀切” vs. “精准滴灌”
    
    Args:
        policy_labels: 政策标签
        talent_loss: 人才流失指标
        city_crowding: 城市拥挤指标
        output_path: 输出路径
        use_example: 是否使用示例数据
    """
    if use_example or policy_labels is None:
        # 生成示例数据
        policy_labels = ['一刀切: 全国范围内取消户籍限制', '精准滴灌: 二线城市对高学历35岁以下开放']
        talent_loss = np.array([0.45, 0.25])  # 中西部人才流失程度
        city_crowding = np.array([0.65, 0.35])  # 东部城市拥挤程度
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(policy_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, talent_loss, width, label='人才流失程度', alpha=0.8)
    bars2 = ax.bar(x + width/2, city_crowding, width, label='城市拥挤程度', alpha=0.8)
    
    ax.set_xlabel('政策类型')
    ax.set_ylabel('指标值')
    
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        # 生成示例数据
        years = np.arange(2020, 2031)
        # 模拟政策冲击前后的效果：人口先增后因工资下降而可能出现回流
        base_pop = 100
        # 确保数组长度与年份长度一致 (11年)
        policy_impact_pop = np.concatenate([
            np.linspace(0, 15, 4),  # 前3年增长
            np.linspace(15, 12, 4),  # 中期因拥挤效应增长放缓
            np.linspace(12, 8, 3)   # 后期可能因工资下降出现回流 (现在总数是4+4+3=11)
        ])
        population = base_pop + policy_impact_pop
        
        # 工资变化：因劳动力供给增加而下降，然后可能因回流而回升
        base_wage = 50000
        policy_impact_wage = np.concatenate([
            np.linspace(0, -3000, 4),  # 前3年下降
            np.linspace(-3000, -5000, 4),  # 中期进一步下降
            np.linspace(-5000, -3000, 3)   # 后期因回流而回升 (现在总数是4+4+3=11)
        ])
        wages = base_wage + policy_impact_wage
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('年份')
    ax1.set_ylabel('西部地区人口数量', color=color)
    line1 = ax1.plot(years, population, label='人口数量', color=color, linewidth=2, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('平均工资', color=color)
    line2 = ax2.plot(years, wages, label='平均工资', color=color, linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加政策实施标识
    ax1.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='政策开始实施')
    
    
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"政策意外后果图已保存到: {output_path}")
    
    return fig