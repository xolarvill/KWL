import torch
import numpy as np
import pandas as pd
import os
import time

from config.model_config import ModelConfig
from data.data_loader import DataLoader
from model.dynamic_model import DynamicModel
from estimation.model_estimator import ModelEstimator

def main():
    print("启动劳动力迁移动态离散选择模型估计...")
    start_time = time.time()
    
    # 1. 加载配置
    config = ModelConfig()
    print(f"配置加载完成，使用子样本组: {config.subsample_group}")
    
    # 2. 加载数据
    print("开始加载数据...")
    data_loader = DataLoader(config)
    
    # 3. 创建动态模型（统一接口）
    print("初始化动态模型...")
    dynamic_model = DynamicModel.from_data_loader(config, data_loader)
    
    # 4. 创建模型估计器
    print("初始化模型估计器...")
    estimator = ModelEstimator(
        model=dynamic_model,
        config=config,
        output_dir="outputs"
    )
    
    # 5. 执行参数估计
    print("开始执行参数估计...")
    estimation_results = estimator.estimate()
    
    # 6. 计算标准误差和p值
    print("计算标准误差和p值...")
    std_errors = estimator.compute_standard_errors()
    p_values = estimator.compute_p_values(std_errors)
    
    # 7. 保存结果
    estimator.save_results()
    
    # 8. 生成报告
    report = estimator.generate_report(output_format=config.output_language)
    
    # 保存报告到文件
    report_dir = os.path.join("outputs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"estimation_report_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存至: {report_file}")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"参数估计完成，总耗时: {total_time:.2f}秒")
    
    return estimation_results

if __name__ == "__main__":
    main()