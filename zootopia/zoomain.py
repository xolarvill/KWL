# main.py
import torch
from pathlib import Path
import argparse


from config.model_config import ModelConfig
from data.data_loader import DataLoader
from models.dynamic_model import DynamicChoiceModel
from optimization.estimator import ModelEstimator
from utils.output_formatter import ResultFormatter

def main(args):
    # 加载配置
    config = ModelConfig(args)
    
    # 数据准备
    ## 创建数据加载器
    data_loader = DataLoader(config) 
    ## 直接调用加载函数，它会使用配置中的路径
    individual_data = data_loader.load_individual_data(args.individual_data_path)
    regional_data = data_loader.load_regional_data(args.regional_data_path)
    adjacency_matrix = data_loader.load_adjacency_matrix(args.adjacency_matrix_path)
    
    # 初始化模型
    model = DynamicChoiceModel(config)
    
    # 参数估计
    estimator = ModelEstimator(model, merged_data, config)
    estimated_params = estimator.estimate()
    std_errors = estimator.compute_standard_errors()
    p_values = estimator.compute_p_values()
    
    # 格式化结果
    formatter = ResultFormatter(config)
    results = formatter.format_parameters(estimated_params, std_errors, p_values)
    
    # 导出结果
    formatter.export_to_stata_format(results, args.output_path)
    
    print("Parameter estimation completed successfully.")
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Discrete Choice Model for Labor Flow Analysis")
    parser.add_argument("--individual_data_path", type=str, required=True, help="Path to individual panel data")
    parser.add_argument("--regional_data_path", type=str, required=True, help="Path to regional characteristics data")
    parser.add_argument("--adjacency_matrix_path", type=str, required=True, help="Path to regional adjacency matrix")
    parser.add_argument("--output_path", type=str, default="./results/parameters.txt", help="Path for saving estimation results")
    parser.add_argument("--n_tau_types", type=int, default=3, help="Number of heterogeneity types")
    parser.add_argument("--n_support_points", type=int, default=5, help="Number of support points for discretization")
    # 添加更多命令行参数...
    
    args = parser.parse_args()
    main(args)