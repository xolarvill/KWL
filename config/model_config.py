# 管理和存储模型的各种配置参数

from typing import List

class ModelConfig:
    '''
    Note: 
    1: 如果不想要从头重新处理数据，使用#号注释掉等号后面的路径，并且添加None作为空路径。为了方便调试，建议使用None。
    2: 人群分割条件可以根据实际需求进行调整。其中1为年龄小于30岁，2为年龄大于等于30岁，3为年龄大于等于65岁。
    3: 涉及参数、函数等模型修改时，需要同时修改其他py文件中的对应部分。
    4: 人群种类tau影响的变量有gamma0_tau1、gamma0_tau2、gamma0_tau3，config对其赋值的为gamma0_tau1_ini、gamma0_tau2_ini、gamma0_tau3_ini。对应的概率为pi_tau1、pi_tau2、pi_tau3。
    '''
    def __init__(self):
        # 数据路径参数
        self.individual_data_path: str | None = None #'path/to/default/individual_data.dta'
        self.regional_data_path: str | None = None #'path/to/default/regional_data.xlsx'
        
        self.prov_code_ranked_path: str = 'data/prov_code_ranked.json'
        self.prov_name_ranked_path: str = 'data/prov_name_ranked.json'
        
        self.adjacency_matrix_path: str = 'data/adjacent.xlsx' # 邻接矩阵
        
        self.prov_language_data_path: str = 'data/prov_language_data.csv' # 省份代表性语言
        self.linguistic_data_path: str = 'data/linguistic.json' # 语言谱系树
        self.linguistic_matrix_path: str = 'data/linguistic_matrix.csv' # 语言亲疏度矩阵，越大越疏远
        
        self.distance_matrix_path: str = 'data/distance_matrix.csv' # 物理距离矩阵
        
        
        # 动态生成不同人群的切割条件
        # 限制subsample_group只能为1,2,3
        self.subsample_group: int = 0
        if not isinstance(self.subsample_group, int) or self.subsample_group not in [0, 1, 2, 3]:
            raise ValueError('subsample_group必须为0、1、2或3，0为不需要进行分割')
        
        # 外生参数
        self.discount_factor: float = 0.95  # 贴现因子，迁移一般是考虑久远的影响，所以此处取0.95
        self.n_regions: int = 31  # 地区数量
        self.n_period: int = 7 # 时期数量
        self.age_min: int = 18  # 最小年龄
        self.age_max: int = 65  # 最大年龄
        
        # 未知变量相关参数
        ## 给定支撑点数量
        self.n_nu_support_points: int = 5  # 个体-地区匹配效应支撑点数量
        self.n_xi_support_points: int = 5  # 地区偏好效应支撑点数量
        self.n_eta_support_points: int = 7  # 个体固定效应支撑点数量
        self.n_sigmavarepsilon_support_points: int = 3  # 暂态效应方差支撑点数量
        
        ## 由均匀分布假设对应的各自取值概率
        self.prob_nu_support_points: List[float] = [1/self.n_nu_support_points] * self.n_nu_support_points  # 个体-地区匹配效应支撑点概率
        self.prob_xi_support_points: List[float] = [1/self.n_xi_support_points] * self.n_xi_support_points  # 地区偏好效应支撑点概率
        self.prob_eta_support_points: List[float] = [1/self.n_eta_support_points] * self.n_eta_support_points  # 个体固定效应支撑点概率
        self.prob_sigmavarepsilon_support_points: List[float] = [1/self.n_sigmavarepsilon_support_points] * self.n_sigmavarepsilon_support_points  # 暂态效应方差支撑点概率
        
        ## 代估支撑点值
        self.nu_support_1_ini: float = 0.3  # 个体-地区匹配效应支撑点1
        self.nu_support_2_ini: float = 0.6  # 个体-地区匹配效应支撑点2
        
        self.xi_support_1_ini: float = 0.4  # 地区偏好效应支撑点1
        self.xi_support_2_ini: float = 0.7  # 地区偏好效应支撑点2
        
        self.eta_support_1_ini: float = 1.0  # 个体固定效应支撑点1
        self.eta_support_2_ini: float = 2.0  # 个体固定效应支撑点2
        self.eta_support_3_ini: float = 3.0  # 个体固定效应支撑点3

        self.sigmavarepsilon_support_1_ini: float = 0.2  # 暂态效应方差支撑点1
        self.sigmavarepsilon_support_2_ini: float = 0.4  # 暂态效应方差支撑点2
        self.sigmavarepsilon_support_3_ini: float = 0.6  # 暂态效应方差支撑点3
        self.sigmavarepsilon_support_4_ini: float = 0.8  # 暂态效应方差支撑点4
        
        # 异质性群体
        """
        Note:
        按照定义sum_{tau} pi_{tau} = 1，所以pi_1_ini + pi_2_ini + pi_3_ini = 1，只需设置其中两个即可。
        """
        ## 给定种类数量
        self.n_tau_types: int = 3  # 迁移类型数量
        self.tau: List[int] = [1, 2, 3]
        ## 代估种类概率值
        self.pi_1_ini: float = 0.3
        self.pi_2_ini: float = 0.4
        
        # 效用函数待估参数
        ## u
        self.alpha0_ini: float = 0.8 # wage income parameter 
        self.alpha1_ini: float = 0.8 # houseprice
        self.alpha2_ini: float = 0.8 # environment = hazard + temperature + air quality + water supply
        self.alpha3_ini: float = 0.8 # education 
        self.alpha4_ini: float = 0.8 # health
        self.alpha5_ini: float = 0.8 # business
        self.alpha6_ini: float = 0.3 # cultural: linguistic
        self.alpha7_ini: float = 0.5 # public goods
        self.alphaH_ini: float = 0.1 # home premium parameter
        self.alphaP_ini: float = 0.1 # hukou penalty parameter

        ## wage
        self.r1_ini: float = 0.8 # 年龄一次项参数
        self.r2_ini: float = 0.8 # 年龄二次项参数
        self.rt_ini: float = 0.8 # 时间参数
        
        ## 迁移成本参数（gamma系列）
        self.gamma0_tau1_ini: float = 0.3 # 第一类群体的迁移成本截距
        self.gamma0_tau2_ini: float = 0.4 # 第二类群体的迁移成本截距
        self.gamma0_tau3_ini: float = 0.3 # 第三类群体的迁移成本截距
        self.gamma1_ini: float = -0.1 # 距离衰减系数
        self.gamma2_ini: float = -0.2 # 邻近省份折扣
        self.gamma3_ini: float = -0.4 # 先前省份折扣
        self.gamma4_ini: float = 0.5 # 年龄对迁移成本的影响
        self.gamma5_ini: float = -0.8 # 更大的城市更便宜
        
        
        
        # 优化参数
        self.max_iter: int = 1000
        self.tolerance: float = 1e-6
        
        # 输出参数
        self.output_language: str = 'LaTeX'
        self.output_file: str = 'tex'
        self.output_style: str = 'Booktab'
        self.base_dir: str = 'logs_outputs'
        self.logs_dir: str = 'logs_outputs/logs'
        self.outputs_dir: str = 'logs_outputs/outputs'
        