from time import time
from config.model_config import ModelConfig
from data.data_loader import DataLoader
from model.likelihood_aggregator import LikelihoodAggregator
from estimation.model_estimator import ModelEstimator
from utils.output_formatter import ResultFormatter

def main():
    '''
    主函数，用于参数估计
    '''
    # 加载配置，实例化ModelConfig
    config = ModelConfig()
    
    # 数据准备
    ## 创建数据加载器
    data_loader = DataLoader(config) 
    ## 直接调用加载函数，它会使用配置中的路径
    individual_data = data_loader.load_individual_data()
    regional_data = data_loader.load_regional_data()
    adjacency_matrix = data_loader.load_adjacency_matrix()
    
    # 初始化模型
    model = LikelihoodAggregator(config)
    
    # 参数估计
    estimator = ModelEstimator(model, individual_data, regional_data, adjacency_matrix, config) # 实例化优化器，传入优化配置、样本似然函数模型和所有数据
    estimated_params = estimator.estimate() 
    std_errors = estimator.compute_standard_errors() 
    p_values = estimator.compute_p_values() 
    
    # 格式化结果
    formatter = ResultFormatter(config) # 实例化结果格式化器
    results = formatter.format_parameters(estimated_params, std_errors, p_values)
    
    # 导出结果
    formatter.export_to_stata_format(results, config.output_dir)
    
    print("Parameter estimation completed successfully.")
    print(f"Results saved to {config.output_dir}")

if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")