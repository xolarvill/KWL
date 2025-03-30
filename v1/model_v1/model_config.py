import torch

class ModelConfig:
    def __init__(self, args=None):
        # 模型结构参数
        self.n_regions = 31  # 假设31个省级单位
        self.n_time_periods = 7  # 2010-2022年
        self.n_tau_types = 3 if args is None else args.n_tau_types  # 异质性类型数量
        self.n_support_points = 3 if args is None else args.n_support_points  # 支撑点数量
        
        # 效用函数参数
        self.discount_factor = 0.95  # 贴现因子
        self.terminal_period = 65  # 退休年龄
        self.max_age = 60  # 最大年龄
        
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
        
        # 参数初始值
        self.alpha0_ini = 0.8  # 经济效益参数
        self.alpha1_ini = 0.8  # 房价参数
        self.alpha2_ini = 0.8  # 天气参数
        self.alpha3_ini = 0.8  # 教育参数
        self.alpha4_ini = 0.8  # 健康参数
        self.alpha5_ini = 0.8  # 交通参数
        self.alphaH_ini = 0.1  # 恋家溢价参数
        self.aplhaP_ini = 0.1  # 户籍制度障碍参数
        self.xi_ini = 0.1  # 地区偏好参数
        self.zeta_ini = 0.1  # 系统性误差参数
        
        # 工资方程参数
        self.r1_ini = 0.8  # 年龄一次项参数
        self.r2_ini = -0.1  # 年龄二次项参数
        self.rt_ini = 0.05  # 时间项参数
        
        # 迁移成本参数
        self.gammaF_ini = 0.8  # 迁移摩擦参数
        self.gamma0_ini = 0.5  # 异质性摩擦参数
        self.gamma1_ini = -0.1  # 距离衰减系数
        self.gamma2_ini = 0.5  # 邻近省份折扣
        self.gamma3_ini = 0.8  # 先前省份折扣
        self.gamma4_ini = 0.05  # 年龄对迁移成本的影响
        self.gamma5_ini = 0.8  # 城市规模对迁移成本的影响
        
        # 支撑点初始值
        self.eta_support_1_ini = -1.0  # 个体固定效应支撑点1
        self.eta_support_2_ini = 0.0   # 个体固定效应支撑点2
        self.eta_support_3_ini = 1.0   # 个体固定效应支撑点3
        self.nu_support_1_ini = -0.5   # 个体-地区匹配效应支撑点1
        self.nu_support_2_ini = 0.0    # 个体-地区匹配效应支撑点2
        self.nu_support_3_ini = 0.5    # 个体-地区匹配效应支撑点3
        self.xi_support_1_ini = -0.5   # 地区偏好效应支撑点1
        self.xi_support_2_ini = 0.0    # 地区偏好效应支撑点2
        self.xi_support_3_ini = 0.5    # 地区偏好效应支撑点3
        self.sigmavarepsilon_support_1_ini = 0.3  # 暂态效应方差支撑点1
        self.sigmavarepsilon_support_2_ini = 0.5  # 暂态效应方差支撑点2
        self.sigmavarepsilon_support_3_ini = 0.7  # 暂态效应方差支撑点3
        
        # 类型概率初始值
        self.pi_tau_1_ini = 0.33  # 类型1概率
        self.pi_tau_2_ini = 0.33  # 类型2概率
        self.pi_tau_3_ini = 0.34  # 类型3概率