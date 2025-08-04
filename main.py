import torch
import numpy as np
import pandas as pd
import os
import time

from config.model_config import ModelConfig
from data_handling.data_loader import DataLoader
from model.dynamic_model import DynamicModel
from estimation.model_estimator import ModelEstimator

def main():
    """主执行函数"""
    print("启动劳动力迁移动态离散选择模型估计...")
    start_time = time.time()
    
    # 1. 加载配置
    config = ModelConfig()
    print(f"配置加载完成，使用子样本组: {config.subsample_group}")
    
    # 2. 加载数据
    print("开始加载数据...")
    data_loader = DataLoader(config)
    
    # 3. 创建动态模型
    print("初始化动态模型...")
    # 注意：由于模型估计的复杂性，第一次运行可能需要较长时间进行即时编译和数据处理
    dynamic_model = DynamicModel.from_data_loader(config, data_loader)
    
    # 4. 创建模型估计器
    print("初始化模型估计器...")
    estimator = ModelEstimator(
        model=dynamic_model,
        config=config
    )
    
    # 5. 执行参数估计
    print("开始执行参数估计...")
    estimation_results = estimator.estimate()
    
    # 6. 保存结果 (包含参数、标准误、p值等)
    print("正在保存估计结果...")
    estimator.save_results()
    
    # 7. 生成并保存报告
    print("正在生成并保存报告...")
    estimator.generate_report(output_format=config.output_language, save_to_file=True)

if __name__ == "__main__":
    main()