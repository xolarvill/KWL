import pandas as pd
import numpy as np
import torch
from scipy.optimize import minimize
from llh_individual_ds import MigrationParameters
from llh_log_sample_ds import TotalLogLikelihood
from std import compute_hessian, ParameterResults

# 优化器配置
def estimate_parameters(
    data: pd.DataFrame,
    geo_data: pd.DataFrame,
    adjmatrix: np.ndarray,
    dismatrix: np.ndarray,
    max_iter: int = 200
) -> MigrationParameters:
    # 初始化参数和总体似然函数
    params = MigrationParameters()
    all_provinces = geo_data['provcd'].unique().tolist()
    total_log_likelihood = TotalLogLikelihood(
        all_pids=data['pid'].unique().tolist(),
        data=data,
        params=params,
        all_provinces=all_provinces,
        n_jobs=-1
    )
    
    # 使用SciPy的L-BFGS优化器（支持大规模参数）
    initial_params = torch.cat([p.flatten() for p in params.parameters()]).detach().numpy()
    
    result = minimize(
        fun=lambda x: total_log_likelihood(torch.tensor(x, requires_grad=True)).item(),
        x0=initial_params,
        method='L-BFGS-B',
        jac=lambda x: torch.autograd.grad(total_log_likelihood(torch.tensor(x)), torch.tensor(x))[0].numpy(),
        options={'maxiter': max_iter}
    )
    
    # 更新最终参数
    with torch.no_grad():
        for i, param in enumerate(params.parameters()):
            param.copy_(torch.tensor(result.x[i]))

    # 计算Hessian矩阵
    hessian = compute_hessian(total_log_likelihood, params)
        
    # 生成结果对象
    results = ParameterResults(params, hessian)
    results.calculate_statistics()
    return results

# 避免未定义
total_log_likelihood = []

# 验证梯度正确性
def gradient_check(params: MigrationParameters, epsilon=1e-5):
    """数值梯度验证"""
    base_loss = total_log_likelihood(params.parameters())
    gradients = []
    for param in params.parameters():
        grad = []
        for i in range(param.numel()):
            original = param.data.flatten()[i].item()
            param.data.flatten()[i] += epsilon
            loss_plus = total_log_likelihood(params.parameters())
            param.data.flatten()[i] = original - epsilon
            loss_minus = total_log_likelihood(params.parameters())
            param.data.flatten()[i] = original
            grad.append((loss_plus - loss_minus) / (2 * epsilon))
        gradients.append(torch.tensor(grad))
    return gradients


# example
if __name__ == '__main__':
    # 加载数据
    data = pd.read_csv('individual_data.csv')
    geo_data = pd.read_csv('geo_data.csv')
    adjmatrix = np.load('adjmatrix.npy')
    dismatrix = np.load('dismatrix.npy')

    # 参数估计
    estimated_params = estimate_parameters(data, geo_data, adjmatrix, dismatrix)
    print("Estimated Parameters:", estimated_params.to_dict())