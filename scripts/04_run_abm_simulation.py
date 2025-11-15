"""
ABM代理抽取与宏观参数校准主脚本
对应论文第1378-1538行：人口合成 + SMM校准 + 验证
"""
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.model_config import ModelConfig
from src.abm.simulation_engine import ABMSimulationEngine


def main():
    """
    ABM校准主函数
    """
    print("="*80)
    print("ABM代理抽取与宏观参数校准")
    print("="*80)
    
    # 步骤0: 初始化配置
    print("\n[0/4] 初始化配置...")
    config = ModelConfig()
    
    # 打印ABM配置
    abm_config = config.get_abm_config()
    print("ABM配置:")
    for key, value in list(abm_config.items())[:5]:
        print(f"  {key}: {value}")
    
    # 步骤1: 创建模拟引擎
    print("\n[1/4] 创建ABM模拟引擎...")
    engine = ABMSimulationEngine(config)
    
    # 步骤2: 从估计结果初始化（或使用Config初始值）
    print("\n[2/4] 初始化参数...")
    
    # 检查是否有02估计结果（真实路径）
    estimation_result_path = "results/estimation/structural_parameters.pkl"
    
    if os.path.exists(estimation_result_path):
        print(f"✓ 找到02估计结果: {estimation_result_path}")
        engine.initialize_from_estimation(estimation_result_path)
    else:
        print("! 02估计结果未找到，使用ModelConfig初始值作为占位符")
        engine.initialize_from_estimation()
    
    # 步骤3: 构建合成人口（三步法）
    print("\n[3/4] 构建合成人口（三步法）...")
    population = engine.build_synthetic_population()
    
    # 人口统计信息
    print("\n合成人口统计:")
    print(f"  总人口规模: {len(population):,}")
    print(f"  平均年龄: {population['age'].mean():.1f}岁")
    print(f"  教育水平: {population['education'].mean():.1f}年")
    print(f"  类型分布: {dict(population['agent_type'].value_counts().sort_index())}")
    
    # 人口地理分布
    location_counts = population['current_location'].value_counts().sort_index()
    top_5_locations = location_counts.nlargest(5)
    print(f"\n  人口最多的5个省份:")
    for loc, count in top_5_locations.items():
        print(f"    省份{loc}: {count:,}人 ({count/len(population)*100:.1f}%)")
    
    # 步骤4: SMM宏观参数校准
    print("\n[4/4] SMM宏观参数校准...")
    
    # 获取目标矩
    target_moments = config.get_target_moments()
    
    # 执行校准
    calibration_results = engine.calibrate_macro_parameters(
        target_moments=target_moments,
        save_path="results/abm/calibration_results.pkl"
    )
    
    # 步骤5: 模型验证（可选，需要完整模拟）
    run_validation = True
    if run_validation:
        print("\n[5/4] 模型验证（Zipf定律 + 胡焕庸线）...")
        validation_results = engine.validate_model(plot=True)
        
        # 打印验证报告
        print_validation_report(validation_results)
    
    # 步骤6: 保存完整模型
    print("\n[6/4] 保存校准后的模型...")
    engine.save_model(save_path="results/abm/calibrated_model.pkl")
    
    # 最终总结
    print("\n" + "="*80)
    print("ABM校准完成总结")
    print("="*80)
    print_final_report(engine, calibration_results, validation_results if run_validation else None)
    
    return engine, calibration_results, validation_results if run_validation else None


def print_validation_report(validation_results: Dict[str, Any]):
    """打印验证报告"""
    print("\n" + "-"*60)
    print("验证结果详细报告")
    print("-"*60)
    
    # Zipf定律结果
    zipf = validation_results['zipf_law']
    print(f"\n1. Zipf定律检验:")
    print(f"   - 幂指数 ζ: {zipf['zipf_exponent']:.4f} {'✓' if zipf['in_range'] else '✗'}")
    print(f"   - R²: {zipf['r_squared']:.4f}")
    print(f"   - 目标范围: [{zipf.get('target_range', (1.05, 1.11))[0]}, {zipf.get('target_range', (1.05, 1.11))[1]}]")
    print(f"   - 偏差: {zipf['deviation']:.4f}")
    
    # 胡焕庸线结果
    hu_line = validation_results['hu_line']
    print(f"\n2. 胡焕庸线检验:")
    print(f"   - 东南半壁占比: {hu_line['eastern_share']:.4f} ({hu_line['eastern_share']*100:.1f}%)")
    print(f"   - 目标占比: {hu_line['target_share']:.4f} ({hu_line['target_share']*100:.1f}%)")
    print(f"   - 相对误差: {hu_line['relative_error']:.4f} ({hu_line['relative_error']*100:.2f}%)")
    print(f"   - 容差: ±{hu_line.get('tolerance', 0.01)*100:.1f}%")
    print(f"   - 结果: {'通过' if hu_line['is_accurate'] else '未通过'}")
    
    # 总体评估
    overall = validation_results['overall_assessment']
    print(f"\n3. 总体评估:")
    print(f"   - Zipf定律: {'通过' if overall['zipf_pass'] else '未通过'}")
    print(f"   - 胡焕庸线: {'通过' if overall['hu_line_pass'] else '未通过'}")
    print(f"   - 综合结果: {overall['status']}")


def print_final_report(engine, calibration_results, validation_results):
    """打印最终报告"""
    print(f"\n1. 基础设置:")
    print(f"   - 省份数量: 29个")
    print(f"   - 合成人口: {len(engine.population):,}人")
    print(f"   - 代理人类型: {engine.config.em_n_types}类")
    
    print(f"\n2. 宏观参数校准:")
    for param, value in calibration_results['optimal_params'].items():
        print(f"   - {param}: {value:.4f}")
    
    print(f"\n3. 矩拟合质量:")
    print(f"   - 平均相对误差: {calibration_results['fit_metrics']['average_relative_error']:.4f}")
    print(f"   - 总目标矩数量: {len(calibration_results['target_moments'])}")
    
    if validation_results:
        print(f"\n4. 模型验证:")
        overall = validation_results['overall_assessment']
        print(f"   - Zipf定律: {'涌现成功' if overall['zipf_pass'] else '涌现失败'}")
        print(f"   - 胡焕庸线: {'拟合成功' if overall['hu_line_pass'] else '拟合失败'}")
        print(f"   - 综合评估: {overall['status']}")
    else:
        print(f"\n4. 模型验证: 跳过（需要完整模拟）")
    
    print(f"\n5. 输出文件:")
    print(f"   - 校准结果: results/abm/calibration_results.pkl")
    print(f"   - 完整模型: results/abm/calibrated_model.pkl")
    print(f"   - 验证图表: results/figures/zipf_law_validation.png")
    print(f"   - 人口分布: results/figures/population_distribution_hu_line.png")


if __name__ == '__main__':
    try:
        # 运行ABM校准流程
        main()
        
        print("\n✓ ABM校准完成!")
        print("\n下一步: 运行 scripts/05_policy_counterfactual.py 进行政策反事实分析")
        
    except Exception as e:
        print(f"\n✗ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)