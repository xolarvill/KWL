"""
生成示例输出结果的脚本
此脚本用于在参数估计完成前生成示例结果，以满足论文进度汇报要求
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.example_generator import generate_all_examples
from src.visualization.results_plot import (
    plot_estimation_results,
    plot_model_fit_flow,
    plot_migration_by_groups,
    plot_abm_zipf_law,
    plot_policy_dynamic_impact,
    plot_feature_importance,
    plot_ml_performance,
    plot_counterfactual_analysis
)
from src.visualization.policy_analysis import (
    plot_policy_comparison_efficiency,
    plot_policy_heterogeneous_effects,
    plot_unintended_consequences
)


def generate_example_outputs():
    """生成所有示例输出"""
    print("=" * 60)
    print("开始生成示例输出结果...")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/policy_figures", exist_ok=True)
    
    # 1. 生成基本统计结果
    print("\n1. 生成示例表格...")
    from src.utils.example_generator import (
        generate_all_examples, 
        create_example_heterogeneity_table,
        create_example_comparison_table, 
        create_example_goodness_of_fit_table,
        create_example_counterfactual_table,
        create_example_abm_calibration_table,
        create_example_out_of_sample_plot,
        create_example_hu_line_plot
    )
    generator = generate_all_examples()  # 这会返回生成器实例
    
    # 生成额外的表格
    create_example_heterogeneity_table(generator)
    create_example_comparison_table()
    create_example_goodness_of_fit_table()
    create_example_counterfactual_table()
    create_example_abm_calibration_table()
    
    # 2. 生成可视化图表
    print("\n2. 生成示例图表...")
    
    # 参数估计结果图
    plot_estimation_results(
        output_path="results/figures/estimation_results.png",
        use_example=True
    )
    
    # 模型拟合：实际与预测迁移流量
    plot_model_fit_flow(
        output_path="results/figures/model_fit_flow.png",
        use_example=True
    )
    
    # 分组迁移率对比图
    plot_migration_by_groups(
        output_path="results/figures/migration_by_groups.png",
        use_example=True
    )
    
    # ABM Zipf定律验证
    plot_abm_zipf_law(
        output_path="results/figures/abm_zipf_law.png",
        use_example=True
    )
    
    # 政策动态影响图
    plot_policy_dynamic_impact(
        output_path="results/figures/policy_dynamic_impact.png",
        use_example=True
    )
    
    # 特征重要性图
    plot_feature_importance(
        output_path="results/figures/feature_importance.png",
        use_example=True
    )
    
    # 机器学习性能对比
    plot_ml_performance(
        output_path="results/figures/ml_performance.png",
        use_example=True
    )
    
    # 反事实分析结果
    plot_counterfactual_analysis(
        output_path="results/figures/counterfactual_analysis.png",
        use_example=True
    )
    
    # 额外图表
    create_example_out_of_sample_plot(generator, "results/figures")
    create_example_hu_line_plot("results/figures")
    
    # 3. 生成政策分析图
    print("\n3. 生成政策分析图表...")
    
    # 政策组合效率比较
    plot_policy_comparison_efficiency(
        output_path="results/policy_figures/policy_efficiency.png",
        use_example=True
    )
    
    # 政策异质性效应
    plot_policy_heterogeneous_effects(
        output_path="results/policy_figures/policy_heterogeneous_effects.png",
        use_example=True
    )
    
    # 政策意外后果
    plot_unintended_consequences(
        output_path="results/policy_figures/unintended_consequences.png",
        use_example=True
    )
    
    print("\n" + "=" * 60)
    print("示例输出生成完成！")
    print("=" * 60)
    print("\n生成的文件包括:")
    print("表格文件 (results/tables/):")
    print("  - main_estimation_results.tex (结构参数估计结果)")
    print("  - model_fit_metrics.tex (模型拟合指标)")

    print("  - heterogeneity_results.tex (未观测异质性结果)")
    print("  - comparison_results.tex (研究成果对比)")
    print("  - goodness_of_fit.tex (模型拟合优度指标)")
    print("  - counterfactual_effects.tex (反事实模拟结果)")
    print("  - abm_calibration.tex (ABM校准目标矩)")
    print("\n基本图表 (results/figures/):")
    print("  - estimation_results.png (参数估计结果图)")
    print("  - model_fit_flow.png (迁移流量拟合图)")
    print("  - migration_by_groups.png (分组迁移率对比图)")
    print("  - abm_zipf_law.png (城市规模分布图)")
    print("  - policy_dynamic_impact.png (政策动态影响图)")
    print("  - feature_importance.png (特征重要性图)")
    print("  - ml_performance.png (机器学习性能对比图)")
    print("  - counterfactual_analysis.png (反事实分析结果图)")
    print("  - out_of_sample_validation.png (样本外预测检验图)")
    print("  - hu_line_emergence.png (胡焕庸线人口分布图)")
    print("\n政策分析图表 (results/policy_figures/):")
    print("  - policy_efficiency.png (政策组合效率比较)")
    print("  - policy_heterogeneous_effects.png (政策异质性效应)")
    print("  - unintended_consequences.png (政策意外后果)")
    print("\n现在你可以使用这些示例结果进行论文进度汇报！")


def generate_minimal_examples():
    """生成最小集示例（用于快速测试）"""
    print("生成最小集示例...")
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 只生成关键表格和图表
    from src.utils.example_generator import ExampleResultGenerator
    generator = ExampleResultGenerator()
    
    # 保存关键表格
    from src.utils.outreg2 import output_estimation_results, output_model_fit_results
    params = generator.example_params
    std_errors = generator.generate_example_standard_errors()
    t_stats = generator.generate_example_t_stats()
    p_values = generator.generate_example_p_values()
    metrics = generator.example_metrics
    

    
    # 生成关键图表
    plot_estimation_results(
        output_path="results/figures/estimation_results.png",
        use_example=True
    )
    
    plot_model_fit_flow(
        output_path="results/figures/model_fit_flow.png",
        use_example=True
    )
    
    print("最小集示例已生成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成示例输出结果")
    parser.add_argument('--minimal', action='store_true', 
                       help='生成最小集示例（仅关键结果）')
    
    args = parser.parse_args()
    
    if args.minimal:
        generate_minimal_examples()
    else:
        generate_example_outputs()