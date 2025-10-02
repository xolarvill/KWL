import torch
import numpy as np
from torch import Tensor, nn
from typing import Dict, List
from config import ModelConfig

class MigrationParameters(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 使用torch.nn.Parameter存储非离散化的待估参数
        ## u(x,j)
        self.alpha0 = torch.nn.Parameter(torch.tensor(self.config.alpha0_ini)) # wage income parameter 
        self.alpha1 = torch.nn.Parameter(torch.tensor(self.config.alpha1_ini)) # houseprice
        self.alpha2 = torch.nn.Parameter(torch.tensor(self.config.alpha2_ini)) # environment = hazard + temperature + air quality + water supply
        self.alpha3 = torch.nn.Parameter(torch.tensor(self.config.alpha3_ini)) # education 
        self.alpha4 = torch.nn.Parameter(torch.tensor(self.config.alpha4_ini)) # health
        self.alpha5 = torch.nn.Parameter(torch.tensor(self.config.alpha5_ini)) # business
        self.alpha6 = torch.nn.Parameter(torch.tensor(self.config.alpha6_ini)) # language
        self.alpha7 = torch.nn.Parameter(torch.tensor(self.config.alpha7_ini)) # public goods
        self.alphaH = torch.nn.Parameter(torch.tensor(self.config.alphaH_ini)) # home premium parameter
        self.alphaP = torch.nn.Parameter(torch.tensor(self.config.alphaP_ini)) # hukou penalty parameter
        self.xi = torch.nn.Parameter(torch.tensor(self.config.xi_ini)) # random permanent component
        self.zeta = torch.nn.Parameter(torch.tensor(self.config.zeta_ini)) # exogenous shock
        
        ## wage
        self.r1 = torch.nn.Parameter(torch.tensor(self.config.r1_ini)) # 年龄一次项参数
        self.r2 = torch.nn.Parameter(torch.tensor(self.config.r2_ini)) # 年龄二次项参数
        self.rt = torch.nn.Parameter(torch.tensor(self.config.rt_ini)) # 时间项项参数
        
        ## 迁移成本参数（gamma系列）
        self.gammaF = torch.nn.Parameter(torch.tensor(self.config.gammaF_ini)) # mirgration friction parameter
        self.gamma0 = torch.nn.Parameter(torch.tensor(self.config.gamma0_ini)) # heterogenous friction parameter 
        self.gamma1 = torch.nn.Parameter(torch.tensor(self.config.gamma1_ini)) # 距离衰减系数
        self.gamma2 = torch.nn.Parameter(torch.tensor(self.config.gamma2_ini)) # 邻近省份折扣
        self.gamma3 = torch.nn.Parameter(torch.tensor(self.config.gamma3_ini)) # 先前省份折扣
        self.gamma4 = torch.nn.Parameter(torch.tensor(self.config.gamma4_ini)) # 年龄对迁移成本的影响
        self.gamma5 = torch.nn.Parameter(torch.tensor(self.config.gamma5_ini)) # 更大的城市更便宜
        
        # 支撑点离散化用tensor向量表示
        
        ## 个体固定效应支撑点
        self.eta_support = torch.nn.Parameter(torch.tensor([
            - self.config.eta_support_3_ini,
            - self.config.eta_support_2_ini,
            - self.config.eta_support_1_ini,
            0,
            self.config.eta_support_1_ini,
            self.config.eta_support_2_ini,
            self.config.eta_support_3_ini
        ]))
        
        ## 个体-地区匹配的效应支撑点
        self.nu_support = torch.nn.Parameter(torch.tensor([
            - self.config.nu_support_2_ini,
            - self.config.nu_support_1_ini,
            0,
            self.config.nu_support_1_ini,
            self.config.nu_support_2_ini,
        ]))
        
        ## 地区偏好效应的支撑点
        self.xi_support = torch.nn.Parameter(torch.tensor([
            - self.config.xi_support_2_ini,
            - self.config.xi_support_1_ini,
            0,
            self.config.xi_support_1_ini,
            self.config.xi_support_2_ini
        ]))
        
        ## 暂态效应方差的支撑点
        self.sigmavarepsilon_support = torch.nn.Parameter(torch.tensor([
            self.config.sigmavarepsilon_support_1_ini,
            self.config.sigmavarepsilon_support_2_ini,
            self.config.sigmavarepsilon_support_3_ini,
            self.config.sigmavarepsilon_support_4_ini
        ]))
        
        # 异质性群体类型概率 - 使用未归一化的参数，在使用时进行softmax归一化
        self.pi_tau_raw = torch.nn.Parameter(torch.tensor([
            self.config.pi_tau_1_ini,
            self.config.pi_tau_2_ini,
            1 - self.config.pi_tau_1_ini - self.config.pi_tau_2_ini
        ]))
        
    def to_dict(self) -> Dict[str, Tensor]:
        # 转换为参数字典，用于数值计算
        return {name: param.data for name, param in self.named_parameters()}

    def named_parameters(self) -> Dict[str, Tensor]:
        # 获取所有可训练参数
        return dict(self.named_parameters())

    def parameters(self):
        # 获取所有可训练参数值
        return self._parameters.values()

    @property
    def pi_tau(self):
        """
        返回归一化后的类型概率
        使用数值稳定性处理的softmax归一化，确保在优化过程中概率值始终为正且和为1
        """
        # 减去最大值以增强数值稳定性，避免指数运算时的溢出问题
        shifted_raw = self.pi_tau_raw - torch.max(self.pi_tau_raw)
        return torch.nn.functional.softmax(shifted_raw, dim=0)