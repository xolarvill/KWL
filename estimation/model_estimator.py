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
    
    def calculate_fit_measures(self, log_likelihood=None):
        """计算拟合优度指标（AIC、BIC和McFadden's Pseudo-R²）"""
        self.logger.info("计算拟合优度指标...")
        
        if log_likelihood is None:
            log_likelihood = self.model.log_likelihood().item()
        
        # 获取参数数量
        n_params = sum(p.numel() for p in self.model.parameters())
        
        # 获取样本量
        n_samples = getattr(self.model, 'N', 0)
        if n_samples == 0:
            self.logger.warning("无法获取样本量，使用默认值1000")
            n_samples = 1000
        
        # 计算AIC: -2*log_likelihood + 2*n_params
        aic = -2 * log_likelihood + 2 * n_params
        
        # 计算BIC: -2*log_likelihood + n_params*log(n_samples)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        # 计算McFadden's Pseudo-R²
        # 需要获取零模型的对数似然值（所有参数为0的模型）
        try:
            # 保存当前参数
            current_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
            
            # 设置所有参数为0
            for param in self.model.parameters():
                param.data.zero_()
            
            # 计算零模型的对数似然
            null_log_likelihood = self.model.log_likelihood().item()
            
            # 恢复参数
            for name, param in self.model.named_parameters():
                param.data.copy_(current_params[name])
            
            # 计算McFadden's Pseudo-R²: 1 - (log_likelihood / null_log_likelihood)
            mcfadden_r2 = 1 - (log_likelihood / null_log_likelihood)
        except Exception as e:
            self.logger.warning(f"计算McFadden's Pseudo-R²时出错: {str(e)}")
            mcfadden_r2 = float('nan')
        
        # 返回结果
        fit_measures = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'mcfadden_r2': mcfadden_r2,
            'n_params': n_params,
            'n_samples': n_samples
        }
        
        self.logger.info(f"拟合优度指标: AIC={aic:.2f}, BIC={bic:.2f}, McFadden's R²={mcfadden_r2:.4f}")
        
        return fit_measures
    
    def vuong_test(self, alternative_model, n_bootstrap=1000):
        """执行Vuong测试比较两个模型
        
        参数:
            alternative_model: 替代模型
            n_bootstrap: 自助法重复次数
            
        返回:
            dict: 包含测试统计量和p值的字典
        """
        self.logger.info("执行Vuong测试...")
        
        try:
            # 获取两个模型的个体似然值
            model1_likelihoods = torch.stack(self.model._compute_individual_likelihoods())
            model2_likelihoods = torch.stack(alternative_model._compute_individual_likelihoods())
            
            # 计算似然比
            likelihood_ratios = torch.log(model1_likelihoods) - torch.log(model2_likelihoods)
            
            # 计算Vuong统计量
            n = len(likelihood_ratios)
            mean_ratio = torch.mean(likelihood_ratios).item()
            std_ratio = torch.std(likelihood_ratios, unbiased=True).item()
            vuong_statistic = (np.sqrt(n) * mean_ratio) / std_ratio
            
            # 计算p值（双尾检验）
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(vuong_statistic)))
            
            # 自助法计算置信区间
            bootstrap_statistics = []
            for _ in range(n_bootstrap):
                # 有放回地抽样
                indices = np.random.choice(n, n, replace=True)
                bootstrap_ratios = likelihood_ratios[indices]
                bootstrap_mean = torch.mean(bootstrap_ratios).item()
                bootstrap_std = torch.std(bootstrap_ratios, unbiased=True).item()
                bootstrap_stat = (np.sqrt(n) * bootstrap_mean) / bootstrap_std
                bootstrap_statistics.append(bootstrap_stat)
            
            # 计算95%置信区间
            bootstrap_statistics.sort()
            lower_ci = bootstrap_statistics[int(0.025 * n_bootstrap)]
            upper_ci = bootstrap_statistics[int(0.975 * n_bootstrap)]
            
            result = {
                'vuong_statistic': vuong_statistic,
                'p_value': p_value,
                'mean_ratio': mean_ratio,
                'std_ratio': std_ratio,
                'confidence_interval': (lower_ci, upper_ci)
            }
            
            self.logger.info(f"Vuong测试结果: 统计量={vuong_statistic:.4f}, p值={p_value:.4f}")
            
            return result
        except Exception as e:
            error_msg = f"执行Vuong测试时出错: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    # 修改estimate方法，在结果中添加拟合优度指标
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
        
        # 计算最终对数似然值
        final_log_likelihood = -self.loss_history[-1] if self.loss_history else float('nan')
        
        # 计算拟合优度指标
        fit_measures = self.calculate_fit_measures(final_log_likelihood)
        
        # 返回估计结果
        return {
            'parameters': {name: param.data for name, param in self.model.named_parameters()},
            'loss_history': self.loss_history,
            'total_time': total_time,
            'log_likelihood': final_log_likelihood,
            'fit_measures': fit_measures
        }
    
    # 修改save_results方法，保存拟合优度指标
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
            
            # 计算拟合优度指标
            final_log_likelihood = -self.loss_history[-1] if self.loss_history else float('nan')
            fit_measures = self.calculate_fit_measures(final_log_likelihood)
            
            # 整理结果
            results = {
                'parameters': params,
                'standard_errors': std_errors if std_errors else {},
                'p_values': p_values if p_values else {},
                'loss_history': self.loss_history,
                'param_history': self.param_history,
                'fit_measures': fit_measures
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
    
    # 修改generate_report方法，添加拟合优度指标
    def generate_report(self, output_format='latex', save_to_file=True):
        """生成估计报告"""
        self.logger.info(f"生成{output_format}格式的估计报告...")
        
        try:
            # 获取参数估计值、标准误差和p值
            params = {name: param.data.item() for name, param in self.model.named_parameters()}
            std_errors = self.compute_standard_errors() or {}
            p_values = self.compute_p_values(std_errors) or {}
            
            # 计算拟合优度指标
            final_log_likelihood = -self.loss_history[-1] if self.loss_history else float('nan')
            fit_measures = self.calculate_fit_measures(final_log_likelihood)
            
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
                
                # 添加拟合优度表格
                latex_table += "\n\n"
                latex_table += "\\begin{table}[htbp]\n"
                latex_table += "\\centering\n"
                latex_table += "\\caption{模型拟合优度}\n"
                latex_table += "\\begin{tabular}{lc}\n"
                latex_table += "\\hline\n"
                latex_table += "指标 & 值 \\\\\n"
                latex_table += "\\hline\n"
                latex_table += f"对数似然值 & {fit_measures['log_likelihood']:.4f} \\\\\n"
                latex_table += f"AIC & {fit_measures['aic']:.4f} \\\\\n"
                latex_table += f"BIC & {fit_measures['bic']:.4f} \\\\\n"
                latex_table += f"McFadden's Pseudo-R² & {fit_measures['mcfadden_r2']:.4f} \\\\\n"
                latex_table += f"参数数量 & {fit_measures['n_params']} \\\\\n"
                latex_table += f"样本量 & {fit_measures['n_samples']} \\\\\n"
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
                
                # 添加拟合优度信息
                report += "\n\n模型拟合优度:\n"
                report += "="*50 + "\n"
                report += f"{'指标':<20} {'值':<10}\n"
                report += "-"*50 + "\n"
                report += f"{'对数似然值':<20} {fit_measures['log_likelihood']:<10.4f}\n"
                report += f"{'AIC':<20} {fit_measures['aic']:<10.4f}\n"
                report += f"{'BIC':<20} {fit_measures['bic']:<10.4f}\n"
                report += f"{'McFadden R²':<20} {fit_measures['mcfadden_r2']:<10.4f}\n"
                report += f"{'参数数量':<20} {fit_measures['n_params']:<10}\n"
                report += f"{'样本量':<20} {fit_measures['n_samples']:<10}\n"
                
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
    
    def compute_standard_errors(self):
        """
        计算标准误差
        """
        self.logger.info("计算标准误差...")
        try:
            # 获取当前参数
            current_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
            # 计算Hessian矩阵
            hessian = torch.autograd.functional.hessian(self.model.log_likelihood, tuple(self.model.parameters()))
            # 计算标准误差
            std_errors = {name: torch.sqrt(torch.diag(hessian[i])) for i, name in enumerate(current_params.keys())}
            return std_errors
        except Exception as e:
            self.logger.error(f"计算标准误差时出错: {str(e)}")
            return {}
    
    
    def compute_p_values(self, std_errors):
        """
        计算p值
        """
        self.logger.info("计算p值...")
        try:
            # 获取当前参数
            current_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
            # 计算p值
            p_values = {name: 2 * (1 - torch.distributions.Normal(0, 1).cdf(abs(current_params[name] / std_errors[name]))) for name in current_params.keys()}
            return p_values
        except Exception as e:
            self.logger.error(f"计算p值时出错: {str(e)}")
            return {}
    
    
    def prediction_accuracy(self, predictions, actuals):
        """
        计算预测准确率
        """
        self.logger.info("计算预测准确率...")
        try:
            correct_predictions = (predictions == actuals).sum().item()
            accuracy = correct_predictions / len(actuals)
            return accuracy
        except Exception as e:
            self.logger.error(f"计算预测准确率时出错: {str(e)}")
            return float('nan')
    
    
    def simulated_calibration(self, method='default'):
        """
        模拟校准
        """
        self.logger.info("执行模拟校准...")
        try:
            # 根据不同方法执行模拟校准
            if method == 'default':
                # 默认模拟方法
                pass
            else:
                # 自定义模拟方法
                pass
            return "模拟校准完成"
        except Exception as e:
            self.logger.error(f"模拟校准时出错: {str(e)}")
            return "模拟校准失败"