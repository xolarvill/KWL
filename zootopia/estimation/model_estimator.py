# model_estimator.py
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from functools import partial
import joblib
import time
import os
from torch.autograd import grad

from model.likelihood_aggregator import LikelihoodAggregator
from config.model_config import ModelConfig

class ModelEstimator:
    """模型参数估计器，使用L-BFGS优化器和PyTorch自动微分"""
    def __init__(self, 
                 model: LikelihoodAggregator, 
                 config: ModelConfig,
                 output_dir: str = None):
        self.model = model
        self.config = config
        self.output_dir = output_dir or getattr(config, 'output_dir', 'outputs')
        
        # 确保输出目录存在
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 设置优化器，使用L-BFGS
        self.optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=getattr(config, 'learning_rate', 0.1),
            max_iter=config.max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=getattr(config, 'tolerance', 1e-6),
            tolerance_change=getattr(config, 'tolerance', 1e-6)
        )
        
        # 记录优化过程
        self.loss_history = []
        self.param_history = []
        self.start_time = None
        
    def objective_function(self):
        """定义优化目标函数（负对数似然）"""
        # 清除之前的梯度
        self.optimizer.zero_grad()
        
        # 计算负对数似然
        log_likelihood = self.model.log_likelihood()
        loss = -log_likelihood  # 最小化负对数似然等价于最大化对数似然
        
        # 计算梯度
        loss.backward()
        
        # 记录当前损失值
        self.loss_history.append(loss.item())
        
        # 记录当前参数值
        current_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        self.param_history.append(current_params)
        
        # 打印当前迭代信息
        elapsed_time = time.time() - self.start_time
        print(f"Iteration {len(self.loss_history)}, Loss: {loss.item():.6f}, Time: {elapsed_time:.2f}s")
        
        return loss
    
    def estimate(self, max_iterations: int = None):
        """执行参数估计"""
        print("开始参数估计...")
        self.start_time = time.time()
        
        # 设置最大迭代次数
        max_iter = max_iterations or self.config.max_iter
        
        # 定义闭包函数用于优化
        def closure():
            return self.objective_function()
        
        # 执行优化
        try:
            self.optimizer.step(closure)
        except Exception as e:
            print(f"优化过程中出错: {str(e)}")
        
        # 计算总耗时
        total_time = time.time() - self.start_time
        print(f"参数估计完成，总耗时: {total_time:.2f}秒")
        
        # 返回估计结果
        return {
            'parameters': {name: param.data for name, param in self.model.named_parameters()},
            'loss_history': self.loss_history,
            'total_time': total_time
        }
    
    def compute_standard_errors(self):
        """计算标准误差（基于Hessian矩阵的逆）"""
        print("计算标准误差...")
        
        # 获取参数和对数似然函数
        params = list(self.model.parameters())
        log_likelihood = self.model.log_likelihood()
        
        # 计算Hessian矩阵（使用自动微分）
        hessian = self._compute_hessian(log_likelihood, params)
        
        # 计算Hessian矩阵的逆（协方差矩阵）
        try:
            # 使用numpy的线性代数库计算逆矩阵
            hessian_np = hessian.detach().numpy()
            covariance = np.linalg.inv(hessian_np)
            
            # 从协方差矩阵对角线提取标准误差
            std_errors = np.sqrt(np.diag(covariance))
            
            # 将标准误差与参数名称关联
            param_names = [name for name, _ in self.model.named_parameters()]
            std_error_dict = {name: std_errors[i] for i, name in enumerate(param_names)}
            
            return std_error_dict
        except np.linalg.LinAlgError:
            print("警告: Hessian矩阵不可逆，无法计算标准误差")
            return None
    
    def _compute_hessian(self, log_likelihood, params):
        """计算Hessian矩阵"""
        # 计算一阶导数（梯度）
        grads = grad(log_likelihood, params, create_graph=True)
        
        # 初始化Hessian矩阵
        n_params = sum(p.numel() for p in params)
        hessian = torch.zeros((n_params, n_params))
        
        # 计算二阶导数
        for i, g in enumerate(grads):
            # 对每个梯度元素计算二阶导数
            for j, param in enumerate(params):
                if i <= j:  # 利用Hessian矩阵的对称性
                    second_deriv = grad(g, param, retain_graph=True)[0]
                    hessian[i, j] = second_deriv
                    hessian[j, i] = second_deriv  # 对称性
        
        return hessian
    
    def compute_p_values(self, std_errors=None):
        """计算p值（基于正态分布假设）"""
        if std_errors is None:
            std_errors = self.compute_standard_errors()
            if std_errors is None:
                return None
        
        # 获取参数估计值
        params = {name: param.data.item() for name, param in self.model.named_parameters()}
        
        # 计算z统计量和p值
        from scipy import stats
        p_values = {}
        for name, value in params.items():
            if name in std_errors:
                z_stat = value / std_errors[name]
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # 双尾检验
                p_values[name] = p_value
        
        return p_values
    
    def save_results(self, file_name=None):
        """保存估计结果"""
        if not self.output_dir:
            print("未指定输出目录，无法保存结果")
            return
        
        # 默认文件名
        if file_name is None:
            file_name = f"estimation_results_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # 完整文件路径
        file_path = os.path.join(self.output_dir, file_name)
        
        # 获取估计结果
        params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        std_errors = self.compute_standard_errors()
        p_values = self.compute_p_values(std_errors)
        
        # 整理结果
        results = {
            'parameters': params,
            'standard_errors': std_errors,
            'p_values': p_values,
            'loss_history': self.loss_history,
            'param_history': self.param_history
        }
        
        # 保存结果
        try:
            joblib.dump(results, file_path)
            print(f"估计结果已保存至: {file_path}")
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
    
    def generate_report(self, output_format='latex'):
        """生成估计报告"""
        # 获取参数估计值、标准误差和p值
        params = {name: param.data.item() for name, param in self.model.named_parameters()}
        std_errors = self.compute_standard_errors()
        p_values = self.compute_p_values(std_errors)
        
        if output_format.lower() == 'latex':
            # 生成LaTeX表格
            latex_table = "\\begin{table}[htbp]\n"
            latex_table += "\\centering\n"
            latex_table += "\\caption{参数估计结果}\n"
            latex_table += "\\begin{tabular}{lccc}\n"
            latex_table += "\\hline\n"
            latex_table += "参数 & 估计值 & 标准误 & p值 \\\\\n"
            latex_table += "\\hline\n"
            
            for name in sorted(params.keys()):
                value = params[name]
                se = std_errors.get(name, float('nan')) if std_errors else float('nan')
                p = p_values.get(name, float('nan')) if p_values else float('nan')
                
                # 格式化输出
                latex_table += f"{name} & {value:.4f} & {se:.4f} & {p:.4f} \\\\\n"
            
            latex_table += "\\hline\n"
            latex_table += "\\end{tabular}\n"
            latex_table += "\\end{table}"
            
            return latex_table
        else:
            # 生成简单文本报告
            report = "参数估计结果:\n"
            report += "="*50 + "\n"
            report += f"{'参数':<20} {'估计值':<10} {'标准误':<10} {'p值':<10}\n"
            report += "-"*50 + "\n"
            
            for name in sorted(params.keys()):
                value = params[name]
                se = std_errors.get(name, float('nan')) if std_errors else float('nan')
                p = p_values.get(name, float('nan')) if p_values else float('nan')
                
                # 格式化输出
                report += f"{name:<20} {value:<10.4f} {se:<10.4f} {p:<10.4f}\n"
            
            return report
    
    def parallel_computation(self, func, data_list, n_jobs=-1):
        """并行计算函数"""
        # 使用joblib进行并行计算
        return joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(func)(item) for item in data_list
        )