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
        
        # 从config获取输出路径
        self.base_dir = getattr(config, 'base_dir', 'logs_outputs')
        self.logs_dir = getattr(config, 'logs_dir', os.path.join(self.base_dir, 'logs'))
        self.output_dir = getattr(config, 'outputs_dir', os.path.join(self.base_dir, 'outputs')) if output_dir is None else output_dir
        
        # 确保输出目录存在
        self._ensure_directories_exist()
        
        # 设置日志记录
        self._setup_logging()
        
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
        
    def _ensure_directories_exist(self):
        """确保输出目录存在，如果不存在则创建"""
        try:
            for directory in [self.base_dir, self.logs_dir, self.output_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"已创建目录: {directory}")
        except Exception as e:
            error_msg = f"创建目录时出错: {str(e)}"
            print(error_msg)
            raise IOError(error_msg)
    
    def _setup_logging(self):
        """设置日志记录"""
        import logging
        log_file = os.path.join(self.logs_dir, f"estimation_log_{time.strftime('%Y%m%d')}.log")
        
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ModelEstimator')
        self.logger.info("初始化模型估计器完成")
        
    def objective_function(self):
        """定义优化目标函数（负对数似然）"""
        # 清除之前的梯度
        self.optimizer.zero_grad()
        
        try:
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
            
            # 记录当前迭代信息
            elapsed_time = time.time() - self.start_time
            log_msg = f"Iteration {len(self.loss_history)}, Loss: {loss.item():.6f}, Time: {elapsed_time:.2f}s"
            self.logger.info(log_msg)
            print(log_msg)  # 同时在控制台显示
            
            return loss
        except Exception as e:
            error_msg = f"计算目标函数时出错: {str(e)}"
            self.logger.error(error_msg)
            # 返回一个默认的高损失值，以便优化器可以继续
            return torch.tensor(1e10, requires_grad=True)
    
    def estimate(self):
        """执行参数估计"""
        self.logger.info("开始参数估计...")
        self.start_time = time.time()
        
        # 设置最大迭代次数
        max_iter = self.config.max_iter
        
        # 定义闭包函数用于优化
        def closure():
            try:
                return self.objective_function()
            except Exception as e:
                self.logger.error(f"目标函数计算出错: {str(e)}")
                # 返回一个默认的高损失值，以便优化器可以继续
                return torch.tensor(1e10, requires_grad=True)
        
        # 执行优化
        try:
            self.optimizer.step(closure)
        except Exception as e:
            self.logger.error(f"优化过程中出错: {str(e)}")
        
        # 计算总耗时
        total_time = time.time() - self.start_time
        self.logger.info(f"参数估计完成，总耗时: {total_time:.2f}秒")
        
        # 返回估计结果
        return {
            'parameters': {name: param.data for name, param in self.model.named_parameters()},
            'loss_history': self.loss_history,
            'total_time': total_time
        }
    
    def compute_standard_errors(self):
        """计算标准误差（基于Hessian矩阵的逆）"""
        self.logger.info("计算标准误差...")
        
        # 获取参数和对数似然函数
        params = list(self.model.parameters())
        log_likelihood = self.model.log_likelihood()
        
        # 计算Hessian矩阵（使用自动微分）
        try:
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
            except np.linalg.LinAlgError as e:
                self.logger.warning(f"警告: Hessian矩阵不可逆，无法计算标准误差: {str(e)}")
                return None
        except Exception as e:
            self.logger.error(f"计算Hessian矩阵时出错: {str(e)}")
            return None
    
    def _compute_hessian(self, log_likelihood, params):
        """计算Hessian矩阵"""
        try:
            # 计算一阶导数（梯度）
            grads = grad(log_likelihood, params, create_graph=True)
            
            # 初始化Hessian矩阵
            n_params = sum(p.numel() for p in params)
            hessian = torch.zeros((n_params, n_params))
            
            # 计算二阶导数
            for i, g in enumerate(grads):
                # 对每个梯度元素计算二阶导数
                for j, param in enumerate(params):
                    try:
                        if i <= j:  # 利用Hessian矩阵的对称性
                            second_deriv = grad(g, param, retain_graph=True)[0]
                            hessian[i, j] = second_deriv
                            hessian[j, i] = second_deriv  # 对称性
                    except Exception as e:
                        self.logger.warning(f"计算参数 {i},{j} 的二阶导数时出错: {str(e)}")
                        # 对于计算失败的元素，使用0填充
                        hessian[i, j] = 0.0
                        hessian[j, i] = 0.0
            
            return hessian
        except Exception as e:
            self.logger.error(f"计算Hessian矩阵时出错: {str(e)}")
            raise
    
    def compute_p_values(self, std_errors=None):
        """计算p值（基于正态分布假设）"""
        self.logger.info("计算p值...")
        
        if std_errors is None:
            std_errors = self.compute_standard_errors()
            if std_errors is None:
                self.logger.warning("无法计算p值，因为标准误差计算失败")
                return {}
        
        # 获取参数估计值
        params = {name: param.data.item() for name, param in self.model.named_parameters()}
        
        # 计算z统计量和p值
        from scipy import stats
        p_values = {}
        for name, value in params.items():
            try:
                if name in std_errors and std_errors[name] > 0:
                    z_stat = value / std_errors[name]
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # 双尾检验
                    p_values[name] = p_value
                else:
                    self.logger.warning(f"参数 {name} 的标准误差无效或不存在，无法计算p值")
                    p_values[name] = float('nan')
            except Exception as e:
                self.logger.warning(f"计算参数 {name} 的p值时出错: {str(e)}")
                p_values[name] = float('nan')
        
        return p_values
    
    def save_results(self, file_name=None, file_format='pkl'):
        """保存估计结果"""
        self.logger.info("保存估计结果...")
        
        # 确保输出目录存在
        self._ensure_directories_exist()
        
        # 默认文件名（使用当天日期）
        if file_name is None:
            date_str = time.strftime('%Y%m%d')
            file_name = f"estimation_results_{date_str}.{file_format}"
        
        # 完整文件路径
        file_path = os.path.join(self.output_dir, file_name)
        
        # 获取估计结果
        try:
            params = {name: param.data.clone() for name, param in self.model.named_parameters()}
            std_errors = self.compute_standard_errors()
            p_values = self.compute_p_values(std_errors)
            
            # 整理结果
            results = {
                'parameters': params,
                'standard_errors': std_errors if std_errors else {},
                'p_values': p_values if p_values else {},
                'loss_history': self.loss_history,
                'param_history': self.param_history
            }
            
            # 保存结果
            try:
                joblib.dump(results, file_path)
                self.logger.info(f"估计结果已保存至: {file_path}")
                return file_path
            except Exception as e:
                error_msg = f"保存结果时出错: {str(e)}"
                self.logger.error(error_msg)
                return None
        except Exception as e:
            self.logger.error(f"准备保存数据时出错: {str(e)}")
            return None
    
    def generate_report(self, output_format='latex', save_to_file=True):
        """生成估计报告"""
        self.logger.info(f"生成{output_format}格式的估计报告...")
        
        try:
            # 获取参数估计值、标准误差和p值
            params = {name: param.data.item() for name, param in self.model.named_parameters()}
            std_errors = self.compute_standard_errors() or {}
            p_values = self.compute_p_values(std_errors) or {}
            
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
                    try:
                        value = params[name]
                        se = std_errors.get(name, float('nan'))
                        p = p_values.get(name, float('nan'))
                        
                        # 格式化输出
                        latex_table += f"{name} & {value:.4f} & {se:.4f} & {p:.4f} \\\\\n"
                    except Exception as e:
                        self.logger.warning(f"处理参数 {name} 时出错: {str(e)}")
                        latex_table += f"{name} & {'ERROR'} & {'ERROR'} & {'ERROR'} \\\\\n"
                
                latex_table += "\\hline\n"
                latex_table += "\\end{tabular}\n"
                latex_table += "\\end{table}"
                
                # 保存到文件
                if save_to_file:
                    try:
                        date_str = time.strftime('%Y%m%d')
                        file_name = f"outputs_{date_str}.tex"
                        file_path = os.path.join(self.output_dir, file_name)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(latex_table)
                        
                        self.logger.info(f"LaTeX报告已保存至: {file_path}")
                    except Exception as e:
                        self.logger.error(f"保存LaTeX报告时出错: {str(e)}")
                
                return latex_table
            else:
                # 生成简单文本报告
                report = "参数估计结果:\n"
                report += "="*50 + "\n"
                report += f"{'参数':<20} {'估计值':<10} {'标准误':<10} {'p值':<10}\n"
                report += "-"*50 + "\n"
                
                for name in sorted(params.keys()):
                    try:
                        value = params[name]
                        se = std_errors.get(name, float('nan'))
                        p = p_values.get(name, float('nan'))
                        
                        # 格式化输出
                        report += f"{name:<20} {value:<10.4f} {se:<10.4f} {p:<10.4f}\n"
                    except Exception as e:
                        self.logger.warning(f"处理参数 {name} 时出错: {str(e)}")
                        report += f"{name:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}\n"
                
                # 保存到文件
                if save_to_file:
                    try:
                        date_str = time.strftime('%Y%m%d')
                        file_name = f"outputs_{date_str}.txt"
                        file_path = os.path.join(self.output_dir, file_name)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(report)
                        
                        self.logger.info(f"文本报告已保存至: {file_path}")
                    except Exception as e:
                        self.logger.error(f"保存文本报告时出错: {str(e)}")
                
                return report
        except Exception as e:
            error_msg = f"生成报告时出错: {str(e)}"
            self.logger.error(error_msg)
            return f"生成报告失败: {str(e)}"
    
    def parallel_computation(self, func, data_list, n_jobs=-1):
        """并行计算函数"""
        # 使用joblib进行并行计算
        return joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(func)(item) for item in data_list
        )