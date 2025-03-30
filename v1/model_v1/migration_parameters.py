import torch
import numpy as np
from torch import Tensor
from typing import Dict, List
from model_config import ModelConfig

class MigrationParameters:
    """封装所有待估参数，支持自动微分"""
    def __init__(self, config: ModelConfig):
        self.config = config
        # 使用torch.nn.Parameter存储非离散化的待估参数
        ## u(x,j)
        self.alpha0 = torch.nn.Parameter(torch.tensor(self.config.alpha0_ini)) # 经济效益参数 
        self.alpha1 = torch.nn.Parameter(torch.tensor(self.config.alpha1_ini)) # 房价参数
        self.alpha2 = torch.nn.Parameter(torch.tensor(self.config.alpha2_ini)) # 天气参数
        self.alpha3 = torch.nn.Parameter(torch.tensor(self.config.alpha3_ini)) # 教育参数
        self.alpha4 = torch.nn.Parameter(torch.tensor(self.config.alpha4_ini)) # 健康参数
        self.alpha5 = torch.nn.Parameter(torch.tensor(self.config.alpha5_ini)) # 交通参数
        self.alphaH = torch.nn.Parameter(torch.tensor(self.config.alphaH_ini)) # 恋家溢价参数
        self.alphaP = torch.nn.Parameter(torch.tensor(self.config.aplhaP_ini)) # 户籍制度障碍参数
        self.xi = torch.nn.Parameter(torch.tensor(self.config.xi_ini)) # 地区偏好参数
        self.zeta = torch.nn.Parameter(torch.tensor(self.config.zeta_ini)) # 系统性误差参数
        
        ## wage
        self.r1 = torch.nn.Parameter(torch.tensor(self.config.r1_ini)) # 年龄一次项参数
        self.r2 = torch.nn.Parameter(torch.tensor(self.config.r2_ini)) # 年龄二次项参数
        self.rt = torch.nn.Parameter(torch.tensor(self.config.rt_ini)) # 时间项参数
        
        ## 迁移成本参数（gamma系列）
        self.gammaF = torch.nn.Parameter(torch.tensor(self.config.gammaF_ini)) # 迁移摩擦参数
        self.gamma0 = torch.nn.Parameter(torch.tensor(self.config.gamma0_ini)) # 异质性摩擦参数 
        self.gamma1 = torch.nn.Parameter(torch.tensor(self.config.gamma1_ini)) # 距离衰减系数
        self.gamma2 = torch.nn.Parameter(torch.tensor(self.config.gamma2_ini)) # 邻近省份折扣
        self.gamma3 = torch.nn.Parameter(torch.tensor(self.config.gamma3_ini)) # 先前省份折扣
        self.gamma4 = torch.nn.Parameter(torch.tensor(self.config.gamma4_ini)) # 年龄对迁移成本的影响
        self.gamma5 = torch.nn.Parameter(torch.tensor(self.config.gamma5_ini)) # 城市规模对迁移成本的影响
        
        ## 支撑点参数
        # 个体固定效应支撑点
        self.eta_support = torch.tensor([
            self.config.eta_support_1_ini,
            self.config.eta_support_2_ini,
            self.config.eta_support_3_ini
        ])
        
        # 个体-地区匹配效应支撑点
        self.nu_support = torch.tensor([
            self.config.nu_support_1_ini,
            self.config.nu_support_2_ini,
            self.config.nu_support_3_ini
        ])
        
        # 地区偏好效应支撑点
        self.xi_support = torch.tensor([
            self.config.xi_support_1_ini,
            self.config.xi_support_2_ini,
            self.config.xi_support_3_ini
        ])
        
        # 暂态效应方差支撑点
        self.sigmavarepsilon_support = torch.tensor([
            self.config.sigmavarepsilon_support_1_ini,
            self.config.sigmavarepsilon_support_2_ini,
            self.config.sigmavarepsilon_support_3_ini
        ])
        
        # 类型概率
        self.pi_tau = torch.tensor([
            self.config.pi_tau_1_ini,
            self.config.pi_tau_2_ini,
            self.config.pi_tau_3_ini
        ])
        
    def to_dict(self) -> Dict[str, Tensor]:
        """转换为参数字典，用于数值计算"""
        return {name: param.data for name, param in self.named_parameters()}

    def named_parameters(self):
        """获取所有可训练参数"""
        return self._parameters.items()
    
    def parameters(self):
        """获取所有可训练参数值"""
        return self._parameters.values()