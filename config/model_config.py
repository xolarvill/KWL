import torch

class ModelConfig:
    def __init__(self, args):
        # 模型结构参数
        self.n_regions = 31  # 假设31个省级单位
        self.n_time_periods = 7  # 2010-2022年
        self.n_tau_types = args.n_tau_types  # 异质性类型数量
        self.n_support_points = args.n_support_points  # 支撑点数量
        
        # 效用函数参数
        self.discount_factor = 0.95  # 贴现因子
        self.terminal_period = 65  # 退休年龄
        
        # 优化参数
        self.learning_rate = 0.1
        self.max_iter = 100
        self.tolerance = 1e-6
        
        # 计算资源参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_jobs = -1  # 使用所有可用CPU核心
        
        # 值迭代参数
        self.value_iteration_max_iter = 1000
        self.value_iteration_tolerance = 1e-6