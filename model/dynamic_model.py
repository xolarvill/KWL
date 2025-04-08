import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from joblib import Parallel, delayed

from config.model_config import ModelConfig
from model.migration_parameters import MigrationParameters
from model.individual_likelihood import DynamicProgramming, IndividualLikelihood

class DynamicModel(nn.Module):
    """动态离散选择模型的统一接口
    
    该类整合了DynamicProgramming、IndividualLikelihood，
    提供了一个统一的接口来进行模型估计。
    
    参数:
        config (ModelConfig): 模型配置
        individual_data (pd.DataFrame): 个体面板数据
        regional_data (pd.DataFrame): 地区特征数据
        adjacency_matrix (np.ndarray): 地区临近矩阵
        distance_matrix (np.ndarray): 地区距离矩阵
        linguistic_matrix (np.ndarray, optional): 语言相似度矩阵
    """
    def __init__(self, 
                 config: ModelConfig,
                 individual_data: pd.DataFrame,
                 regional_data: pd.DataFrame,
                 adjacency_matrix: np.ndarray,
                 distance_matrix: np.ndarray,
                 linguistic_matrix: Optional[np.ndarray] = None):
        super().__init__()
        self.config = config
        self.individual_data = individual_data
        self.regional_data = regional_data
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = distance_matrix
        self.linguistic_matrix = linguistic_matrix if linguistic_matrix is not None else np.zeros_like(distance_matrix)
        
        # 获取所有个体ID和省份
        self.all_pids = individual_data['pid'].unique().tolist()
        self.all_provinces = regional_data.index.tolist()
        self.N = len(self.all_pids)
        
        # 初始化参数
        self.params = MigrationParameters(config)
        
        # 创建动态规划求解器
        self.dp = self._create_dynamic_programming()
        
        # 预计算期望价值函数
        self.dp.calculate_ev(self.all_provinces)
        
        # 缓存个体似然计算结果
        self.individual_likelihoods_cache = {}
        
    def _create_dynamic_programming(self) -> DynamicProgramming:
        """创建动态规划求解器"""
        return DynamicProgramming(
            config=self.config,
            params=self.params,
            geo_data=self.regional_data,
            distance_matrix=self.distance_matrix,
            adjacency_matrix=self.adjacency_matrix,
            linguistic_matrix=self.linguistic_matrix
        )
    
    def forward(self) -> torch.Tensor:
        """前向传播，计算总体对数似然"""
        return self.log_likelihood()
    
    def log_likelihood(self) -> torch.Tensor:
        """计算总体对数似然
        
        根据theory.md中的公式：
        Λ(θ) = Σ_i log(Σ_τ π_τ L_i(θ_τ))
        """
        # 计算所有个体的似然
        individual_likelihoods = self._compute_individual_likelihoods()
        
        # 计算对数似然总和
        log_lik = torch.sum(torch.log(torch.stack(individual_likelihoods) + 1e-10))
        
        return log_lik
    
    def _compute_individual_likelihoods(self, n_jobs: int = -1) -> List[torch.Tensor]:
        """并行计算所有个体的似然函数
        
        参数:
            n_jobs (int): 并行作业数，-1表示使用所有可用核心
            
        返回:
            List[torch.Tensor]: 所有个体的似然列表
        """
        # 使用joblib并行计算
        if n_jobs != 1:
            likelihoods = Parallel(n_jobs=n_jobs)(
                delayed(self._compute_single_likelihood)(pid) 
                for pid in self.all_pids
            )
        else:
            # 串行计算（用于调试）
            likelihoods = [self._compute_single_likelihood(pid) for pid in self.all_pids]
            
        return likelihoods
    
    def _compute_single_likelihood(self, pid: int) -> torch.Tensor:
        """计算单个个体的似然函数
        
        根据theory.md中的公式：
        L_i(θ_τ) = Σ_ω∈Ω(N_i) (Π_t ψ_it λ_it) / (n_ν n_ε n_ξ (n_ν)^N_i)
        
        参数:
            pid (int): 个体ID
            
        返回:
            torch.Tensor: 个体似然值
        """
        # 检查缓存
        if pid in self.individual_likelihoods_cache:
            return self.individual_likelihoods_cache[pid]
        
        try:
            # 计算个体在所有类型下的似然总和
            total_lik = torch.tensor(0.0, requires_grad=True)
            
            # 遍历所有可能的类型tau
            for tau in range(self.config.n_tau_types):
                # 计算类型为tau的个体似然
                tau_lik = self._compute_tau_likelihood(pid, tau)
                
                # 加权求和（使用类型概率）
                total_lik += self.params.pi_tau[tau] * tau_lik
            
            # 缓存结果
            self.individual_likelihoods_cache[pid] = total_lik
            return total_lik
            
        except Exception as e:
            print(f"计算个体 {pid} 的似然函数时出错: {str(e)}")
            return torch.tensor(1e-10, requires_grad=True)  # 返回一个很小的值，避免log(0)
    
    def _compute_tau_likelihood(self, pid: int, tau: int) -> torch.Tensor:
        """计算特定类型tau下的个体似然
        
        参数:
            pid (int): 个体ID
            tau (int): 类型索引
            
        返回:
            torch.Tensor: 类型为tau的个体似然
        """
        tau_lik = torch.tensor(0.0, requires_grad=True)
        
        # 遍历所有可能的支撑点组合
        for eta_idx in range(self.config.n_eta_support_points):
            for nu_idx in range(self.config.n_nu_support_points):
                for xi_idx in range(self.config.n_xi_support_points):
                    for sigma_idx in range(self.config.n_sigmavarepsilon_support_points):
                        # 创建个体似然计算器
                        individual_likelihood = IndividualLikelihood(
                            pid=pid,
                            data=self.individual_data,
                            dp=self.dp,
                            params=self.params,
                            config=self.config
                        )
                        
                        # 计算该组合下的似然贡献
                        omega_lik = individual_likelihood._calculate_omega_likelihood(
                            tau, eta_idx, nu_idx, xi_idx, sigma_idx
                        )
                        
                        # 计算该组合的概率权重
                        omega_prob = (
                            self.config.prob_eta_support_points[eta_idx] *
                            self.config.prob_nu_support_points[nu_idx] *
                            self.config.prob_xi_support_points[xi_idx] *
                            self.config.prob_sigmavarepsilon_support_points[sigma_idx]
                        )
                        
                        # 加权求和
                        tau_lik += omega_prob * omega_lik
        
        return tau_lik
    
    def parameters(self):
        """返回模型参数"""
        return self.params.parameters()
    
    def named_parameters(self):
        """返回命名参数"""
        return self.params.named_parameters()
    
    def get_dynamic_programming(self) -> DynamicProgramming:
        """获取动态规划求解器"""
        return self.dp
    
    def get_params(self) -> MigrationParameters:
        """获取模型参数"""
        return self.params
    
    def get_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.config
    
    def clear_cache(self):
        """清除缓存"""
        self.individual_likelihoods_cache = {}
    
    @classmethod
    def from_data_loader(cls, config: ModelConfig, data_loader):
        """从数据加载器创建模型"""
        individual_data = data_loader.load_individual_data()
        regional_data = data_loader.load_regional_data()
        adjacency_matrix = data_loader.load_adjacency_matrix()
        
        # 尝试加载语言相似度矩阵
        try:
            linguistic_matrix = data_loader.load_linguistic_matrix()
        except (AttributeError, FileNotFoundError):
            print("未找到语言相似度矩阵，使用零矩阵代替")
            linguistic_matrix = np.zeros((len(regional_data), len(regional_data)))
        
        # 尝试加载距离矩阵
        try:
            distance_matrix = data_loader.load_distance_matrix()
        except (AttributeError, FileNotFoundError):
            print("未找到距离矩阵，使用邻接矩阵生成简单距离矩阵")
            # 简单距离矩阵：邻接为1，非邻接为2，对角线为0
            distance_matrix = np.where(adjacency_matrix > 0, 1, 2)
            np.fill_diagonal(distance_matrix, 0)
        
        return cls(
            config=config,
            individual_data=individual_data,
            regional_data=regional_data,
            adjacency_matrix=adjacency_matrix,
            distance_matrix=distance_matrix,
            linguistic_matrix=linguistic_matrix
        )