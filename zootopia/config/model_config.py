# 管理和存储模型的各种配置参数
import torch

class ModelConfig:
    '''
    Note: 
    1: 如果不想要从头重新处理数据，使用#号注释掉等号后面的路径，并且添加None作为空路径。为了方便调试，建议使用None。
    2: 人群分割条件可以根据实际需求进行调整。其中1为年龄小于30岁，2为年龄大于等于30岁，3为年龄大于等于65岁。
    3: 涉及参数、函数等模型修改时，需要同时修改其他py文件中的对应部分。
    '''
    def __init__(self):
        # 数据路径参数
        self.individual_data_path: str = None #'path/to/default/individual_data.dta'
        self.regional_data_path: str = None #'path/to/default/regional_data.xlsx'
        self.adjacency_matrix_path: str = 'file/adjacent.xlsx'
        self.linguistic_data_path: str = 'file/linguistic.json'
        self.provcd_rank: str = 'file/provcd_rank.json'
        
        # 动态生成不同人群的切割条件
        # 限制subsample_group只能为1,2,3
        self.subsample_group: int = 1
        if not isinstance(subsample_group, int) or subsample_group not in [1, 2, 3]:
            raise ValueError('subsample_group必须为1、2或3')
        
        # 外生参数
        self.discount_factor: float = 0.95  # 贴现因子，迁移一般是考虑久远的影响，所以此处取0.95
        self.n_regions: int = 31  # 地区数量
        
        # 未知变量相关参数
        self.n_eta_support_points: int = 3  # 个体固定效应支撑点数量
        self.n_sigmavarepsilon_support_points: int = 3  # 暂态效应方差支撑点数量
        self.eta_support_1_ini: float = -1.0  # 个体固定效应支撑点1
        self.eta_support_2_ini: float = 0.0  # 个体固定效应支撑点2
        self.eta_support_3_ini: float = 1.0  # 个体固定效应支撑点3
        self.nu_support_1_ini: float = -0.5  # 个体-地区匹配效应支撑点1
        self.nu_support_2_ini: float = 0.0  # 个体-地区匹配效应支撑点2
        self.nu_support_3_ini: float = 0.5  # 个体-地区匹配效应支撑点3
        self.xi_support_1_ini: float = -0.5  # 地区偏好效应支撑点1
        self.xi_support_2_ini: float = 0.0  # 地区偏好效应支撑点2
        self.xi_support_3_ini: float = 0.5  # 地区偏好效应支撑点3
        self.sigmavarepsilon_support_1_ini: float = -0.5  # 暂态效应方差支撑点1
        self.sigmavarepsilon_support_2_ini: float = 0.0  # 暂态效应方差支撑点2
        self.sigmavarepsilon_support_3_ini: float = 0.5  # 暂态效应方差支撑点3
        
        # torch.nn.Parameter初始待估参数
        ## u
        self.alpha0_ini: float = 0.8 # wage income parameter 
        self.alpha1_ini: float = 0.8 # houseprice
        self.alpha2_ini: float = 0.8 # weather = hazard + temperature + air quality + water supply
        self.alpha3_ini: float = 0.8 # education 
        self.alpha4_ini: float = 0.8 # health
        self.alpha5_ini: float = 0.8 # traffic = public transportation + road service
        self.alphaH_ini: float = 0.1 # home premium parameter
        self.alphaP_ini: float = 0.1 # hukou penalty parameter

        ## wage
        self.r1_ini: float = 0.8 # 年龄一次项参数
        self.r2_ini: float = 0.8 # 年龄二次项参数
        self.rt_ini: float = 0.8 # 时间参数
        
        ## 迁移成本参数（gamma系列）
        self.gammaF_ini: float = 0.8 # mirgration friction parameter
        self.gamma0_ini: float = 0.5 # heterogenous friction parameter 
        self.gamma1_ini: float = -0.1 # 距离衰减系数
        self.gamma2_ini: float = 0.5 # 邻近省份折扣
        self.gamma3_ini: float = 0.8 # 先前省份折扣
        self.gamma4_ini: float = 0.05 # 年龄对迁移成本的影响
        self.gamma5_ini: float = 0.8 # 更大的城市更便宜
        
        ## 异质性群体概率
        self.pi1_ini: float = 0.8
        self.pi2_ini: float = 0.8
        self.pi3_ini: float = 0.8
        
        # 优化参数
        self.learning_rate: float = 0.1
        self.max_iter: int = 100
        self.tolerance: int = 1e-6
        
        # 计算资源参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_jobs = -1  # 使用所有可用CPU核心
        
        # 值迭代参数
        self.max_iter = 1000
        self.tolerance = 1e-6
        self.terminal_period: int = 65  # 退休年龄
        
        # 输出参数
        self.output_language: str = 'LaTeX'
        self.output_file: str = 'tex'
        self.output_style: str = 'Booktab'
        self.output_dir: str = 'outputs_logs/outputs'