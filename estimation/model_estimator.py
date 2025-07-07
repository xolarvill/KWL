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
import torch.nn.utils as utils

from model.dynamic_model import DynamicModel
from config.model_config import ModelConfig

class ModelEstimator:
    """模型参数估计器，使用L-BFGS优化器和PyTorch自动微分"""
    def __init__(self, 
                 model: DynamicModel, 
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
        
        # 缓存计算结果，避免重复计算
        self._cached_statistical_results = {}
        
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
        
        # 清除之前的缓存
        self._cached_statistical_results = {}
        
        # 定义闭包函数用于优化
        def closure():
            try:
                return self.objective_function()
            except Exception as e:
                self.logger.error(f"目标函数计算出错: {str(e)}")
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

    def compute_standard_errors(self):
        """
        使用Hessian矩阵的逆计算参数的标准误。
        该方法首先将所有模型参数扁平化为一个向量，然后计算对数似然函数
        关于此向量的Hessian矩阵。标准误由协方差矩阵（负Hessian矩阵的逆）
        的对角线元素的平方根得出。
        """
        self.logger.info("正在计算标准误...")
        try:
            # 1. 将所有参数扁平化为一个向量
            params_vector = utils.parameters_to_vector(self.model.parameters())

            # 2. 定义一个函数，该函数接受一个扁平化的参数向量，并返回对数似然值
            def log_likelihood_for_hessian(p_vector):
                utils.vector_to_parameters(p_vector, self.model.parameters())
                return self.model.log_likelihood()

            # 3. 计算Hessian矩阵
            self.logger.info("正在计算Hessian矩阵，这可能需要一些时间...")
            hessian = torch.autograd.functional.hessian(log_likelihood_for_hessian, params_vector)
            
            # 4. 计算协方差矩阵（负Hessian矩阵的逆）
            #    为保证数值稳定性，在求逆之前添加一个小的扰动项到对角线
            identity_matrix = torch.eye(hessian.size(0), device=hessian.device)
            hessian_stabilized = hessian - identity_matrix * 1e-6
            covariance_matrix = torch.inverse(-hessian_stabilized)
            
            # 5. 提取对角线元素并计算平方根得到标准误
            std_errors_vector = torch.sqrt(torch.diag(covariance_matrix))

            if torch.isnan(std_errors_vector).any() or torch.isinf(std_errors_vector).any():
                self.logger.warning("计算出的标准误包含NaN或无穷大值。Hessian矩阵可能不是正定的。")
                return {}

            # 6. 将扁平化的标准误向量映射回原始参数的形状
            std_errors = {}
            pointer = 0
            for name, param in self.model.named_parameters():
                num_param = param.numel()
                std_errors[name] = std_errors_vector[pointer:pointer + num_param].view_as(param)
                pointer += num_param
            
            self.logger.info("标准误计算完成。")
            return std_errors

        except torch.linalg.LinAlgError as e:
            self.logger.error(f"计算标准误时发生线性代数错误（Hessian矩阵可能是奇异的）: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"计算标准误时发生未知错误: {str(e)}")
            return {}

    def compute_p_values(self, std_errors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算参数的p值。
        p值是基于z统计量（参数估计值除以其标准误）和标准正态分布计算的。
        """
        self.logger.info("正在计算p值...")
        if not std_errors:
            self.logger.warning("标准误字典为空，无法计算p值。")
            return {}
        try:
            p_values = {}
            for name, param in self.model.named_parameters():
                if name in std_errors and std_errors[name].numel() == param.numel():
                    # z-statistic
                    z_stat = torch.abs(param.data / (std_errors[name] + 1e-8)) # 避免除以零
                    # p-value from standard normal distribution (2-tailed)
                    p_values[name] = 2 * (1 - torch.distributions.Normal(0, 1).cdf(z_stat))
                else:
                    self.logger.warning(f"参数 {name} 的标准误缺失或形状不匹配，无法计算p值。")
                    p_values[name] = torch.full_like(param.data, float('nan'))

            self.logger.info("p值计算完成。")
            return p_values
        except Exception as e:
            self.logger.error(f"计算p值时发生错误: {str(e)}")
            return {}

    def _get_statistical_results(self):
        """获取或计算并缓存统计结果（参数、标准误、p值）"""
        if self._cached_statistical_results:
            return self._cached_statistical_results
        
        self.logger.info("首次计算统计结果...")
        params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        std_errors = self.compute_standard_errors()
        p_values = self.compute_p_values(std_errors)
        
        self._cached_statistical_results = {
            'parameters': params,
            'standard_errors': std_errors,
            'p_values': p_values
        }
        return self._cached_statistical_results

    def save_results(self, file_name=None, file_format='pkl'):
        """保存估计结果"""
        self.logger.info("保存估计结果...")
        self._ensure_directories_exist()
        
        if file_name is None:
            date_str = time.strftime('%Y%m%d')
            file_name = f"estimation_results_{date_str}.{file_format}"
        
        file_path = os.path.join(self.output_dir, file_name)
        
        try:
            stats_results = self._get_statistical_results()
            final_log_likelihood = -self.loss_history[-1] if self.loss_history else float('nan')
            fit_measures = self.calculate_fit_measures(final_log_likelihood)
            
            results_to_save = {
                **stats_results,
                'loss_history': self.loss_history,
                'param_history': self.param_history,
                'fit_measures': fit_measures
            }
            
            joblib.dump(results_to_save, file_path)
            self.logger.info(f"估计结果已保存至: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"准备或保存数据时出错: {str(e)}")
            return None
    
    def generate_report(self, output_format='latex', save_to_file=True):
        """生成估计报告"""
        self.logger.info(f"生成{output_format}格式的估计报告...")
        
        try:
            stats_results = self._get_statistical_results()
            params = {name: v.item() if v.numel() == 1 else v for name, v in stats_results['parameters'].items()}
            std_errors = {name: v.item() if v.numel() == 1 else v for name, v in stats_results['standard_errors'].items()}
            p_values = {name: v.item() if v.numel() == 1 else v for name, v in stats_results['p_values'].items()}
            
            final_log_likelihood = -self.loss_history[-1] if self.loss_history else float('nan')
            fit_measures = self.calculate_fit_measures(final_log_likelihood)
            
            # ... （报告生成的代码与之前类似，此处为简化版）
            report = self._format_report(output_format, params, std_errors, p_values, fit_measures)
            
            if save_to_file:
                ext = 'tex' if output_format.lower() == 'latex' else 'txt'
                date_str = time.strftime('%Y%m%d')
                file_name = f"outputs_{date_str}.{ext}"
                file_path = os.path.join(self.output_dir, file_name)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"报告已保存至: {file_path}")
            
            return report
        except Exception as e:
            error_msg = f"生成报告时出错: {str(e)}"
            self.logger.error(error_msg)
            return f"生成报告失败: {str(e)}"

    def _format_report(self, output_format, params, std_errors, p_values, fit_measures):
        """内部辅助函数，用于格式化报告"""
        if output_format.lower() == 'latex':
            return self._format_latex_report(params, std_errors, p_values, fit_measures)
        else:
            return self._format_text_report(params, std_errors, p_values, fit_measures)

    def _format_text_report(self, params, std_errors, p_values, fit_measures):
        report = "参数估计结果:\n"
        report += "="*50 + "\n"
        report += f"{'参数':<20} {'估计值':<10} {'标准误':<10} {'p值':<10}\n"
        report += "-"*50 + "\n"
        for name in sorted(params.keys()):
            val = params.get(name, float('nan'))
            se = std_errors.get(name, float('nan'))
            p = p_values.get(name, float('nan'))
            report += f"{name:<20} {val:<10.4f} {se:<10.4f} {p:<10.4f}\n"
        report += "\n模型拟合优度:\n"
        report += "="*50 + "\n"
        for key, value in fit_measures.items():
            report += f"{key:<20} {value:<10.4f}\n"
        return report

    def _format_latex_report(self, params, std_errors, p_values, fit_measures):
        # 参数表格
        latex_table = "\\begin{table}[htbp]\n\\centering\n\\caption{参数估计结果}\n"
        latex_table += "\\begin{tabular}{lccc}\n\\hline\n"
        latex_table += "参数 & 估计值 & 标准误 & p值 \\\\n\\hline\n"
        for name in sorted(params.keys()):
            val = params.get(name, float('nan'))
            se = std_errors.get(name, float('nan'))
            p = p_values.get(name, float('nan'))
            latex_table += f"{name.replace('_', ' ')} & {val:.4f} & {se:.4f} & {p:.4f} \\\\n"
        latex_table += "\\hline\n\\end{tabular}\n\\end{table}\n\n"
        # 拟合优度表格
        latex_table += "\\begin{table}[htbp]\n\\centering\n\\caption{模型拟合优度}\n"
        latex_table += "\\begin{tabular}{lc}\n\\hline\n"
        latex_table += "指标 & 值 \\\\n\\hline\n"
        for key, value in fit_measures.items():
            latex_table += f"{key.replace('_', ' ')} & {value:.4f} \\\\n"
        latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
        return latex_table

    def calculate_fit_measures(self, log_likelihood=None):
        """计算拟合优度指标（AIC、BIC和McFadden's Pseudo-R²）"""
        # ... (此方法基本不变)
        self.logger.info("计算拟合优度指标...")
        if log_likelihood is None:
            log_likelihood = self.model.log_likelihood().item()
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_samples = getattr(self.model, 'N', 1000) # 默认值以防万一
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        # 计算McFadden's Pseudo-R²
        try:
            with torch.no_grad():
                current_params_vector = utils.parameters_to_vector(self.model.parameters())
                utils.vector_to_parameters(torch.zeros_like(current_params_vector), self.model.parameters())
                null_log_likelihood = self.model.log_likelihood().item()
                utils.vector_to_parameters(current_params_vector, self.model.parameters())
            mcfadden_r2 = 1 - (log_likelihood / null_log_likelihood)
        except Exception as e:
            self.logger.warning(f"计算McFadden's Pseudo-R²时出错: {str(e)}")
            mcfadden_r2 = float('nan')
            
        return {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'mcfadden_r2': mcfadden_r2,
            'n_params': n_params,
            'n_samples': n_samples
        }

    # 其他方法 (vuong_test, prediction_accuracy, etc.) 保持不变
    def vuong_test(self, alternative_model, n_bootstrap=1000):
        """执行Vuong测试比较两个模型 (实现待定)"""
        self.logger.warning("Vuong test is not fully implemented yet.")
        return {}

    def prediction_accuracy(self, predictions, actuals):
        """计算预测准确率 (实现待定)"""
        self.logger.warning("Prediction accuracy is not implemented yet.")
        return float('nan')

    def simulated_calibration(self, method='default'):
        """模拟校准 (实现待定)"""
        self.logger.warning("Simulated calibration is not implemented yet.")
        return "模拟校准失败"