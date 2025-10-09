# 示例输出生成系统

## 概述
这个系统用于在参数估计完成前生成示例结果，以满足论文进度汇报要求。系统会自动生成符合论文要求的示例数据、表格和图表。

## 文件结构
- `src/utils/example_generator.py`: 示例数据生成器
- `src/visualization/results_plot.py`: 结果可视化模块
- `src/visualization/policy_analysis.py`: 政策分析可视化模块
- `scripts/generate_example_outputs.py`: 主生成脚本

## 运行方法
```bash
# 生成所有示例输出
uv run python scripts/generate_example_outputs.py

# 或者生成最小集示例（用于快速测试）
uv run python scripts/generate_example_outputs.py --minimal
```

## 生成的输出内容

### 表格文件 (results/tables/)
- `main_estimation_results.tex`: 结构参数估计结果
- `model_fit_metrics.tex`: 模型拟合指标
- `ml_performance.tex`: 机器学习性能结果

### 基本图表 (results/figures/)
- `estimation_results.png`: 参数估计结果图
- `model_fit_flow.png`: 迁移流量拟合图
- `migration_by_groups.png`: 分组迁移率对比图
- `abm_zipf_law.png`: 城市规模分布图
- `policy_dynamic_impact.png`: 政策动态影响图
- `feature_importance.png`: 特征重要性图
- `ml_performance.png`: 机器学习性能对比图
- `counterfactual_analysis.png`: 反事实分析结果图

### 政策分析图表 (results/policy_figures/)
- `policy_efficiency.png`: 政策组合效率比较
- `policy_heterogeneous_effects.png`: 政策异质性效应
- `unintended_consequences.png`: 政策意外后果

## 功能特点
1. **自动数据生成**: 当没有真实估计结果时，自动生成符合论文要求的示例数据
2. **完整可视化**: 覆盖论文中所有要求的图表类型
3. **标准格式输出**: 生成LaTeX表格格式，便于整合到论文中
4. **中文字体支持**: 所有图表均支持中文字体显示

## 使用场景
- 论文进度汇报
- 演示文稿准备
- 论文结构验证
- 代码功能测试

## 未来集成
当真实参数估计完成时，可以轻松切换到使用真实数据，可视化代码将保持不变。