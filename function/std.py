import pandas as pd
import numpy as np
import torch
from scipy.linalg import inv
from scipy.stats import norm
from llh_individual_ds import MigrationParameters
from llh_log_sample_ds import TotalLogLikelihood

def compute_hessian(total_log_likelihood: TotalLogLikelihood, params: MigrationParameters) -> np.ndarray:
    """通过自动微分计算Hessian矩阵"""
    from torch.autograd.functional import hessian
    
    def log_lik_func(params_tensor: torch.Tensor) -> torch.Tensor:
        return total_log_likelihood(params_tensor)
    
    params_tensor = torch.cat([p.flatten() for p in params.parameters()])
    H = hessian(log_lik_func, params_tensor).detach().numpy()
    return H

class ParameterResults:
    """计算参数的标准误、p值，并按学术格式输出"""
    def __init__(self, params: MigrationParameters, hessian: np.ndarray):
        self.params = params
        self.hessian = hessian
        self.std_errors = None
        self.p_values = None
        self.significance = None

    def calculate_statistics(self) -> None:
        """计算标准误和p值"""
        # 计算协方差矩阵（Hessian的逆）
        cov_matrix = inv(-self.hessian)  # 负号因优化最小化负对数似然
        
        # 计算标准误
        self.std_errors = np.sqrt(np.diag(cov_matrix))
        
        # 计算z统计量和p值（假设渐近正态分布）
        param_values = np.array([p.detach().numpy() for p in self.params.parameters()])
        z_scores = param_values / self.std_errors
        self.p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # 双尾检验

        # 显著性标记
        self.significance = [
            '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            for p in self.p_values
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """生成结果DataFrame"""
        df = pd.DataFrame({
            'Parameter': [name for name in self.params.named_parameters().keys()],
            'Estimate': [p.item() for p in self.params.parameters()],
            'Std. Error': self.std_errors,
            'P-value': self.p_values,
            'Significance': self.significance
        })
        return df

    def save_to_file(self, filename: str = 'std/results.tex', format: str = 'latex') -> None:
        """保存为文件（支持LaTeX/CSV）"""
        df = self.to_dataframe()
        if format == 'latex':
            latex_str = df.to_latex(
                index=False,
                float_format="%.4f",
                columns=['Parameter', 'Estimate', 'Std. Error', 'Significance'],
                header=['Parameter', 'Estimate', 'Std. Error', ''],
                escape=False
            ).replace('Significance', '')
            with open(filename, 'w') as f:
                f.write(latex_str)
        elif format == 'csv':
            df.to_csv(filename, index=False)
            

# example

# def estimate_parameters(...) -> ParameterResults:  # 修改返回类型
# ...原有参数估计逻辑...
    
# 计算Hessian矩阵
# hessian = compute_hessian(total_log_likelihood, params)
    
# 生成结果对象
# results = ParameterResults(params, hessian)
# results.calculate_statistics()
# return results

# 示例调用
# results = estimate_parameters(data, geo_data, adjmatrix, dismatrix)
# results.save_to_file('std/results.tex', format='latex')