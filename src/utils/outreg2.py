"""
该模块用于格式化输出估计结果为LaTeX表格
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import os


def output_estimation_results(
    params: Dict[str, float],
    std_errors: Dict[str, float],
    t_stats: Dict[str, float], 
    p_values: Dict[str, float],
    model_fit_metrics: Dict[str, float] = None,
    info_criteria: Dict[str, float] = None,
    output_path: str = "results/tables/estimation_results.tex",
    title: str = "结构参数估计结果"
) -> str:
    """
    输出估计结果到LaTeX表格文件
    
    Args:
        params: 估计参数
        std_errors: 标准误
        t_stats: t统计量
        p_values: p值
        model_fit_metrics: 模型拟合指标
        info_criteria: 信息准则
        output_path: 输出文件路径
        title: 表格标题
    
    Returns:
        str: 生成的LaTeX表格字符串
    """
    # 创建结果数据框
    param_names = list(params.keys())
    results_df = pd.DataFrame({
        'Parameter': param_names,
        'Coefficient': [params[name] for name in param_names],
        'Std. Error': [std_errors.get(name, np.nan) for name in param_names],
        't-statistic': [t_stats.get(name, np.nan) for name in param_names],
        'p-value': [p_values.get(name, np.nan) for name in param_names]
    })
    
    # 添加显著性星号
    results_df['Significance'] = results_df['p-value'].apply(_get_significance_stars)
    
    # 创建LaTeX表格
    latex_table = _create_latex_table(results_df, model_fit_metrics, info_criteria, title)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"估计结果已保存到: {output_path}")
    return latex_table


def _get_significance_stars(p_value: float) -> str:
    """根据p值确定显著性星号"""
    if pd.isna(p_value):
        return ""
    elif p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    else:
        return ""


def _create_latex_table(
    results_df: pd.DataFrame,
    model_fit_metrics: Dict[str, float] = None,
    info_criteria: Dict[str, float] = None,
    title: str = "结构参数估计结果"
) -> str:
    """创建LaTeX表格"""
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{title}}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Parameter & Coefficient & Std. Error & t-statistic & p-value \\\\
\\midrule
"""
    
    # 添加参数估计结果
    for _, row in results_df.iterrows():
        coef = f"{row['Coefficient']:.4f}"
        se = f"{row['Std. Error']:.4f}" if not pd.isna(row['Std. Error']) else ""
        t_stat = f"{row['t-statistic']:.3f}" if not pd.isna(row['t-statistic']) else ""
        p_val = f"{row['p-value']:.3f}" if not pd.isna(row['p-value']) else ""
        sig = row['Significance']
        
        latex += f"{row['Parameter']} & {coef}{sig} & {se} & {t_stat} & {p_val} \\\\\n"
    
    latex += "\\midrule\n"
    
    # 添加模型拟合指标
    if model_fit_metrics:
        latex += "\\multicolumn{5}{l}{\\textit{模型拟合指标:}} \\\\\n"
        for metric, value in model_fit_metrics.items():
            if isinstance(value, float):
                latex += f"\\multicolumn{{5}}{{l}}{{\\quad {metric}: {value:.4f}}} \\\\\n"
            else:
                latex += f"\\multicolumn{{5}}{{l}}{{\\quad {metric}: {value}}} \\\\\n"
    
    # 添加信息准则
    if info_criteria:
        latex += "\\multicolumn{5}{l}{\\textit{信息准则:}} \\\\\n"
        for criterion, value in info_criteria.items():
            latex += f"\\multicolumn{{5}}{{l}}{{\\quad {criterion}: {value:.2f}}} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex


def output_heterogeneity_results(
    support_points: Dict[str, List[float]],
    probabilities: Dict[str, List[float]],
    output_path: str = "results/tables/heterogeneity_results.tex"
) -> str:
    """
    输出未观测异质性结果
    
    Args:
        support_points: 支撑点值
        probabilities: 支撑点概率
        output_path: 输出路径
    
    Returns:
        str: LaTeX表格字符串
    """
    # 获取所有类型的名称
    all_types = set(list(support_points.keys()) + list(probabilities.keys()))
    
    rows = []
    for type_name in all_types:
        sp = support_points.get(type_name, [])
        probs = probabilities.get(type_name, [])
        
        max_len = max(len(sp), len(probs))
        for i in range(max_len):
            row = {
                'Type': type_name,
                'Index': i+1,
                'Support_Point': sp[i] if i < len(sp) else np.nan,
                'Probability': probs[i] if i < len(probs) else np.nan
            }
            rows.append(row)
    
    heterogeneity_df = pd.DataFrame(rows)
    
    # 创建LaTeX表格
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{未观测异质性分布的支撑点与概率}}
\\begin{{tabular}}{{lccc}}
\\toprule
Type & Index & Support Point & Probability \\\\
\\midrule
"""
    
    for _, row in heterogeneity_df.iterrows():
        sp = f"{row['Support_Point']:.4f}" if not pd.isna(row['Support_Point']) else ""
        prob = f"{row['Probability']:.4f}" if not pd.isna(row['Probability']) else ""
        latex += f"{row['Type']} & {row['Index']} & {sp} & {prob} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f"异质性结果已保存到: {output_path}")
    return latex


def output_model_fit_results(
    model_fit_metrics: Dict[str, float],
    output_path: str = "results/tables/model_fit_results.tex"
) -> str:
    """
    输出模型拟合检验结果
    
    Args:
        model_fit_metrics: 模型拟合指标
        output_path: 输出路径
    
    Returns:
        str: LaTeX表格字符串
    """
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{模型拟合度检验结果}}
\\begin{{tabular}}{{lc}}
\\toprule
指标 & 数值 \\\\
\\midrule
"""
    
    for metric, value in model_fit_metrics.items():
        if isinstance(value, float):
            latex += f"{metric} & {value:.4f} \\\\\n"
        else:
            latex += f"{metric} & {value} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f"模型拟合结果已保存到: {output_path}")
    return latex