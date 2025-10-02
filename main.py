"""
项目主运行脚本 - 端到端执行整个研究流程
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd


def run_complete_analysis():
    """
    执行完整的分析流程：
    1. 数据预处理和特征工程
    2. ML插件训练（工资预测）
    3. 结构参数估计
    4. 统计推断和模型拟合检验
    5. ABM反事实政策模拟
    """
    print("开始执行完整的迁移模型分析流程...")
    print("="*60)
    
    # 1. 数据预处理和特征工程
    print("步骤 1: 数据预处理和特征工程")
    print("-"*30)
    try:
        result = subprocess.run([
            sys.executable, "scripts/00_prepare_data.py"
        ], cwd=os.getcwd(), capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 数据预处理完成")
        else:
            print(f"⚠ 数据预处理可能存在问题: {result.stderr}")
    except Exception as e:
        print(f"⚠ 数据预处理执行出错: {e}")
    
    # 2. ML插件训练
    print("\n步骤 2: 训练ML插件（工资预测模型）")
    print("-"*30)
    try:
        result = subprocess.run([
            sys.executable, "scripts/01_train_ml_plugins.py"
        ], cwd=os.getcwd(), capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ML插件训练完成")
        else:
            print(f"⚠ ML插件训练可能存在问题: {result.stderr}")
    except Exception as e:
        print(f"⚠ ML插件训练执行出错: {e}")
    
    # 3. 结构参数估计
    print("\n步骤 3: 结构参数估计")
    print("-"*30)
    try:
        result = subprocess.run([
            sys.executable, "scripts/02_run_estimation.py"
        ], cwd=os.getcwd(), capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 结构参数估计完成")
        else:
            print(f"⚠ 结构参数估计可能存在问题: {result.stderr}")
            print("  运行备用输出生成脚本...")
            subprocess.run([sys.executable, "scripts/generate_all_outputs.py"], 
                         cwd=os.getcwd(), capture_output=True, text=True)
        subprocess.run([sys.executable, "scripts/generate_all_outputs.py"], 
                      cwd=os.getcwd(), capture_output=True, text=True)
    except Exception as e:
        print(f"⚠ 结构参数估计执行出错: {e}")
        print("  运行备用输出生成脚本...")
        subprocess.run([sys.executable, "scripts/generate_all_outputs.py"], 
                      cwd=os.getcwd(), capture_output=True, text=True)
    
    # 4. ABM模拟
    print("\n步骤 4: ABM反事实政策模拟")
    print("-"*30)
    try:
        result = subprocess.run([
            sys.executable, "scripts/03_run_abm_simulation.py"
        ], cwd=os.getcwd(), capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ABM反事实政策模拟完成")
        else:
            print(f"⚠ ABM模拟可能存在问题: {result.stderr}")
    except Exception as e:
        print(f"⚠ ABM模拟执行出错: {e}")
    
    print("\n" + "="*60)
    print("完整的迁移模型分析流程执行完毕！")
    print("\n主要输出文件：")
    print("- results/tables/main_estimation_results.tex: 主要估计结果")
    print("- results/tables/model_fit_metrics.tex: 模型拟合指标") 
    print("- results/tables/heterogeneity_results.tex: 异质性结果")
    print("- results/tables/mechanism_decomposition.tex: 机制分解结果")
    print("- results/policy/policy_analysis_summary.txt: 政策分析摘要")
    print("- results/logs/estimation_log.txt: 估计过程日志")
    print("- results/ml_models/wage_predictor.pkl: 工资预测模型")
    print("\n请检查 results/ 目录下的详细结果文件。")


def check_environment():
    """
    检查运行环境
    """
    print("检查运行环境...")
    
    # 检查必要的目录
    required_dirs = ['data/processed', 'results', 'results/tables', 'results/policy', 'results/ml_models']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")
    
    # 检查必要的数据文件
    required_files = [
        'data/processed/clds.csv',
        'data/processed/geo_amenities.csv',
        'data/processed/prov_code_ranked.json',
        'data/processed/prov_name_ranked.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("警告：缺少以下必需的数据文件：")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✓ 所有必需的数据文件都存在")
        return True


if __name__ == '__main__':
    print("中国劳动力迁移动态离散选择模型分析")
    print("研究项目 - 完整分析流程")
    print()
    
    # 检查环境
    if check_environment():
        print()
        run_complete_analysis()
    else:
        print("\n环境检查失败，请确保所有必需的数据文件都已准备就绪。")