# 管理和存储模型的各种配置参数
from enum import StrEnum
from typing import List, Optional
from dataclasses import dataclass, field
import json

class OutputLanguage(StrEnum):
    LATEX = 'LaTeX'
    MARKDOWN = 'Markdown'
    HTML = 'HTML'
    
class OutputFileFormat(StrEnum):
    LATEX = '.tex'
    MARKDOWN = '.md'
    PLAINTEXT = '.txt'
    
class OutputStyle(StrEnum):
    BOOKTAB = "Booktab"
    PLAIN = "Plain"
    GRID = "Grid"

@dataclass
class ModelConfig:
    '''
    Note: 
    1: 如果不想要从头重新处理数据，使用#号注释掉等号后面的路径，并且添加None作为空路径。为了方便调试，建议使用None。
    2: 人群分割条件可以根据实际需求进行调整。其中1为年龄小于30岁，2为年龄大于等于30岁，3为年龄大于等于65岁。
    3: 涉及参数、函数等模型修改时，需要同时修改其他py文件中的对应部分。
    4: 人群种类tau影响的变量有gamma0_tau1、gamma0_tau2、gamma0_tau3，config对其赋值的为gamma0_tau1_ini、gamma0_tau2_ini、gamma0_tau3_ini。对应的概率为pi_tau1、pi_tau2、pi_tau3。
    '''

    # 数据路径参数
    individual_data_path: str = 'data/processed/cfps10_22mc.dta'
    regional_data_path: str = 'data/processed/geo.xlsx'

    prov_code_ranked_path: str = 'data/processed/prov_code_ranked.json'
    prov_name_ranked_path: str = 'data/processed/prov_name_ranked.json'
    adjacency_matrix_path: str = 'data/processed/adjacent.xlsx'  # 邻接矩阵
    prov_language_data_path: str = 'data/processed/prov_language_data.csv'  # 省份代表性语言
    linguistic_data_path: str = 'data/processed/linguistic.json'  # 语言谱系树
    linguistic_matrix_path: str = 'data/processed/linguistic_matrix.csv'  # 语言亲疏度矩阵，越大越疏远
    distance_matrix_path: str = 'data/processed/distance_matrix.csv'  # 物理距离矩阵

    # 外生参数
    discount_factor: float = 0.95  # 贴现因子
    n_regions: int = 31  # 地区数量
    n_period: int = 7  # 时期数量
    age_min: int = 18  # 最小年龄
    age_max: int = 65  # 最大年龄

    # 未知变量相关参数 - 支撑点数量
    n_nu_support_points: int = 5  # 个体-地区匹配效应
    n_xi_support_points: int = 5  # 地区偏好效应
    n_eta_support_points: int = 7  # 个体固定效应
    n_sigmavarepsilon_support_points: int = 3  # 暂态效应方差

    # 支撑点概率（由均匀分布假设）
    prob_nu_support_points: List[float] = field(init=False)
    prob_xi_support_points: List[float] = field(init=False)
    prob_eta_support_points: List[float] = field(init=False)
    prob_sigmavarepsilon_support_points: List[float] = field(init=False)

    def __post_init__(self):
        """初始化那些标记为field(init=False)的字段"""
        # 假设支撑点概率为均匀分布
        self.prob_nu_support_points = [1.0 / self.n_nu_support_points] * self.n_nu_support_points
        self.prob_xi_support_points = [1.0 / self.n_xi_support_points] * self.n_xi_support_points
        self.prob_eta_support_points = [1.0 / self.n_eta_support_points] * self.n_eta_support_points
        self.prob_sigmavarepsilon_support_points = [1.0 / self.n_sigmavarepsilon_support_points] * self.n_sigmavarepsilon_support_points
        
        
    # 代估支撑点值（初始值）
    nu_support_1_ini: float = 0.3
    nu_support_2_ini: float = 0.6

    xi_support_1_ini: float = 0.4
    xi_support_2_ini: float = 0.7

    eta_support_1_ini: float = 1.0
    eta_support_2_ini: float = 2.0
    eta_support_3_ini: float = 3.0

    sigmavarepsilon_support_1_ini: float = 0.2
    sigmavarepsilon_support_2_ini: float = 0.4
    sigmavarepsilon_support_3_ini: float = 0.6
    sigmavarepsilon_support_4_ini: float = 0.8

    # 异质性群体
    n_tau_types: int = 3  # 迁移类型数量
    tau: List[int] = field(default_factory=lambda: [1, 2, 3])

    pi_1_ini: float = 0.3
    pi_2_ini: float = 0.4
    # pi_3_ini = 1 - pi_1_ini - pi_2_ini （可由外部计算）

    # 效用函数待估参数
    alpha0_ini: float = 0.8  # wage income
    alpha1_ini: float = 0.8  # houseprice
    alpha2_ini: float = 0.8  # environment
    alpha3_ini: float = 0.8  # education
    alpha4_ini: float = 0.8  # health
    alpha5_ini: float = 0.8  # business
    alpha6_ini: float = 0.3  # cultural: linguistic
    alpha7_ini: float = 0.5  # public goods
    alphaH_ini: float = 0.1  # home premium
    alphaP_ini: float = 0.1  # hukou penalty

    # wage 参数
    r1_ini: float = 0.8  # 年龄一次项
    r2_ini: float = 0.8  # 年龄二次项
    rt_ini: float = 0.8  # 时间项

    # 迁移成本参数（gamma）
    gamma0_tau1_ini: float = 0.3
    gamma0_tau2_ini: float = 0.4
    gamma0_tau3_ini: float = 0.3
    gamma1_ini: float = -0.1  # 距离衰减
    gamma2_ini: float = -0.2  # 邻近折扣
    gamma3_ini: float = -0.4  # 先前省份折扣
    gamma4_ini: float = 0.5   # 年龄影响
    gamma5_ini: float = -0.8  # 城市规模影响

    # 优化参数
    max_iter: int = 1000
    tolerance: float = 1e-6

    # 输出参数
    output_language: OutputLanguage = OutputLanguage.LATEX
    output_file: OutputFileFormat = OutputFileFormat.LATEX
    output_style: OutputStyle = OutputStyle.BOOKTAB
    base_dir: str = 'logs_outputs'
    logs_dir: str = 'logs_outputs/logs'
    outputs_dir: str = 'logs_outputs/outputs'


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == '__main__':
    config = ModelConfig()
    print(config)