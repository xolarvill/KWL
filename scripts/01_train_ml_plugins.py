"""
ML插件训练脚本
训练机器学习模型以非参数化地估计滋扰函数，如工资方程
"""

import sys
import os
import time
import warnings
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.config.model_config import ModelConfig
from src.ml_plugins.data_interface import create_ml_interface, MLPredictionResult
from src.ml_plugins.wage_prediction import WagePredictionModel, WageModelEnsemble
from src.ml_plugins.kfold_cv import create_cross_fitter

# 忽略一些常见的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class MLPluginTrainer:
    """
    ML插件训练器
    
    负责训练所有ML插件模型
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化训练器
        
        Parameters:
        -----------
        config : ModelConfig
            模型配置
        """
        self.config = config
        self.ml_interface = create_ml_interface(config)
        self.results = {}
        
        # 创建结果目录
        self.results_dir = Path('results/ml_models')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path('results/logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def train_wage_model(self, 
                        model_type: str = 'single',
                        custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        训练工资预测模型
        
        Parameters:
        -----------
        model_type : str, default='single'
            模型类型，'single' 或 'ensemble'
        custom_params : Dict[str, Any], optional
            自定义参数
            
        Returns:
        --------
        Dict[str, Any]
            训练结果
        """
        print("=" * 60)
        print("开始训练工资预测模型...")
        print(f"模型类型: {model_type}")
        
        start_time = time.time()
        
        try:
            # 准备数据
            print("正在准备数据...")
            individual_data, regional_data = self.ml_interface.prepare_wage_data()
            
            print(f"数据准备完成:")
            print(f"  - 个体观测数: {len(individual_data)}")
            print(f"  - 地区观测数: {len(regional_data)}")
            print(f"  - 个体数: {individual_data['pid'].nunique()}")
            print(f"  - 时间跨度: {individual_data['year'].min()}-{individual_data['year'].max()}")
            
            # 检查工资数据
            wage_stats = individual_data['wage'].describe()
            print(f"工资数据统计:")
            print(f"  - 均值: {wage_stats['mean']:.2f}")
            print(f"  - 中位数: {wage_stats['50%']:.2f}")
            print(f"  - 标准差: {wage_stats['std']:.2f}")
            print(f"  - 缺失值: {individual_data['wage'].isna().sum()}")
            
            # 训练模型
            if model_type == 'single':
                results = self._train_single_wage_model(individual_data, regional_data, custom_params)
            elif model_type == 'ensemble':
                results = self._train_ensemble_wage_model(individual_data, regional_data, custom_params)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 保存结果
            model_name = f"wage_model_{model_type}"
            self.results[model_name] = results
            
            # 记录训练时间
            training_time = time.time() - start_time
            results['training_time'] = training_time
            
            print(f"工资模型训练完成! 用时: {training_time:.2f}秒")
            print(f"验证结果:")
            if 'validation_scores' in results:
                for metric, value in results['validation_scores'].items():
                    if isinstance(value, (int, float)):
                        print(f"  - {metric}: {value:.4f}")
            
            return results
            
        except Exception as e:
            print(f"工资模型训练失败: {str(e)}")
            raise
    
    def _train_single_wage_model(self, 
                                individual_data: pd.DataFrame,
                                regional_data: pd.DataFrame,
                                custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        训练单一工资预测模型
        """
        # 默认参数
        default_lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100
        }
        
        default_cv_params = {
            'n_splits': 5,
            'random_state': 42,
            'individual_col': 'pid',
            'time_col': 'year'
        }
        
        # 合并自定义参数
        if custom_params:
            lgb_params = {**default_lgb_params, **custom_params.get('lgb_params', {})}
            cv_params = {**default_cv_params, **custom_params.get('cv_params', {})}
        else:
            lgb_params = default_lgb_params
            cv_params = default_cv_params
        
        # 创建模型
        model = WagePredictionModel(
            lgb_params=lgb_params,
            cv_params=cv_params,
            feature_engineering=True,
            log_transform=True
        )
        
        # 训练模型
        print("正在进行交叉验证训练...")
        validation_scores = model.fit_cross_validate(individual_data, regional_data)
        
        # 获取特征重要性
        feature_importance = model.get_feature_importance()
        
        # 保存模型
        model_path = self.results_dir / "wage_model_single.joblib"
        model.save_model(str(model_path))
        
        # 创建预测结果对象
        ml_result = MLPredictionResult(
            predictions=model.cross_fit_predictions,
            prediction_type='wage',
            model_type='lightgbm_single',
            cross_validation_scores=validation_scores,
            feature_importance=feature_importance,
            model_metadata={
                'lgb_params': lgb_params,
                'cv_params': cv_params,
                'feature_names': model.feature_names,
                'model_path': str(model_path)
            }
        )
        
        # 存储结果
        self.ml_interface.store_ml_result('wage_model_single', ml_result)
        
        return {
            'model': model,
            'validation_scores': validation_scores,
            'feature_importance': feature_importance,
            'model_path': str(model_path),
            'ml_result': ml_result
        }
    
    def _train_ensemble_wage_model(self, 
                                  individual_data: pd.DataFrame,
                                  regional_data: pd.DataFrame,
                                  custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        训练集成工资预测模型
        """
        # 默认参数
        default_base_params = {
            'lgb_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 100
            },
            'cv_params': {
                'n_splits': 5,
                'random_state': 42,
                'individual_col': 'pid',
                'time_col': 'year'
            },
            'feature_engineering': True,
            'log_transform': True
        }
        
        n_models = custom_params.get('n_models', 3) if custom_params else 3
        base_params = {**default_base_params, **(custom_params.get('base_params', {}) if custom_params else {})}
        
        # 创建集成模型
        ensemble = WageModelEnsemble(
            n_models=n_models,
            base_params=base_params
        )
        
        # 训练集成模型
        print(f"正在训练{n_models}个基础模型的集成...")
        ensemble_scores = ensemble.fit(individual_data, regional_data)
        
        # 获取集成预测
        ensemble_predictions = ensemble.predict(individual_data, regional_data)
        
        # 计算集成特征重要性（平均）
        importance_dfs = []
        for i, model in enumerate(ensemble.models):
            importance = model.get_feature_importance()
            importance['model_idx'] = i
            importance_dfs.append(importance)
        
        combined_importance = pd.concat(importance_dfs)
        avg_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        # 保存集成模型
        ensemble_path = self.results_dir / "wage_model_ensemble.joblib"
        import joblib
        joblib.dump(ensemble, ensemble_path)
        
        # 创建预测结果对象
        ml_result = MLPredictionResult(
            predictions=ensemble_predictions,
            prediction_type='wage',
            model_type='lightgbm_ensemble',
            cross_validation_scores=ensemble_scores,
            feature_importance=avg_importance,
            model_metadata={
                'n_models': n_models,
                'base_params': base_params,
                'individual_scores': ensemble_scores.get('individual_scores', []),
                'model_path': str(ensemble_path)
            }
        )
        
        # 存储结果
        self.ml_interface.store_ml_result('wage_model_ensemble', ml_result)
        
        return {
            'ensemble': ensemble,
            'ensemble_scores': ensemble_scores,
            'feature_importance': avg_importance,
            'model_path': str(ensemble_path),
            'ml_result': ml_result
        }
    
    def train_all_models(self, 
                        wage_model_type: str = 'single',
                        custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        训练所有ML插件模型
        
        Parameters:
        -----------
        wage_model_type : str, default='single'
            工资模型类型
        custom_params : Dict[str, Any], optional
            自定义参数
            
        Returns:
        --------
        Dict[str, Any]
            所有训练结果
        """
        print("=" * 80)
        print("开始训练所有ML插件模型")
        print("=" * 80)
        
        total_start_time = time.time()
        all_results = {}
        
        try:
            # 1. 训练工资预测模型
            wage_results = self.train_wage_model(
                model_type=wage_model_type,
                custom_params=custom_params.get('wage_model', {}) if custom_params else None
            )
            all_results['wage_model'] = wage_results
            
            # 2. 可以在这里添加其他ML模型的训练
            # 例如：状态转移概率模型、选择概率模型等
            
            # 导出结果供结构模型使用
            print("\n正在导出ML结果...")
            exported_files = self.ml_interface.export_for_structural_model()
            all_results['exported_files'] = exported_files
            
            # 创建汇总报告
            summary_report = self.ml_interface.create_summary_report()
            summary_path = self.results_dir / "ml_summary_report.csv"
            summary_report.to_csv(summary_path, index=False)
            all_results['summary_report'] = summary_report
            
            print(f"\nML汇总报告已保存: {summary_path}")
            print("\nML模型汇总:")
            print(summary_report.to_string(index=False))
            
            # 记录总训练时间
            total_time = time.time() - total_start_time
            all_results['total_training_time'] = total_time
            
            print("=" * 80)
            print(f"所有ML插件模型训练完成! 总用时: {total_time:.2f}秒")
            print("=" * 80)
            
            return all_results
            
        except Exception as e:
            print(f"ML模型训练过程中出现错误: {str(e)}")
            raise
    
    def validate_models(self) -> Dict[str, Any]:
        """
        验证训练好的模型
        
        Returns:
        --------
        Dict[str, Any]
            验证结果
        """
        print("正在验证训练好的模型...")
        
        validation_results = {}
        
        # 验证工资模型
        wage_result = self.ml_interface.get_ml_result('wage_model_single')
        if wage_result:
            validation_results['wage_model_single'] = {
                'prediction_count': len(wage_result.predictions),
                'nan_count': np.isnan(wage_result.predictions).sum(),
                'prediction_range': (np.nanmin(wage_result.predictions), np.nanmax(wage_result.predictions)),
                'scores': wage_result.cross_validation_scores
            }
        
        ensemble_result = self.ml_interface.get_ml_result('wage_model_ensemble')
        if ensemble_result:
            validation_results['wage_model_ensemble'] = {
                'prediction_count': len(ensemble_result.predictions),
                'nan_count': np.isnan(ensemble_result.predictions).sum(),
                'prediction_range': (np.nanmin(ensemble_result.predictions), np.nanmax(ensemble_result.predictions)),
                'scores': ensemble_result.cross_validation_scores
            }
        
        return validation_results


def main():
    """主函数"""
    print("启动ML插件训练脚本...")
    
    try:
        # 加载配置
        config = ModelConfig()
        
        # 创建训练器
        trainer = MLPluginTrainer(config)
        
        # 自定义训练参数（可根据需要调整）
        custom_params = {
            'wage_model': {
                'lgb_params': {
                    'n_estimators': 200,  # 增加树的数量
                    'learning_rate': 0.03,  # 降低学习率
                    'num_leaves': 50,  # 增加叶子数
                    'min_data_in_leaf': 20,  # 设置叶子节点最小样本数
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'lambda_l1': 0.1,  # L1正则化
                    'lambda_l2': 0.1,  # L2正则化
                },
                'cv_params': {
                    'n_splits': 5,
                    'random_state': 42
                }
            }
        }
        
        # 训练所有模型
        # 可以选择 'single' 或 'ensemble'
        results = trainer.train_all_models(
            wage_model_type='single',  # 或 'ensemble'
            custom_params=custom_params
        )
        
        # 验证模型
        validation_results = trainer.validate_models()
        print("\n模型验证结果:")
        for model_name, val_result in validation_results.items():
            print(f"\n{model_name}:")
            for key, value in val_result.items():
                print(f"  {key}: {value}")
        
        print("\nML插件训练脚本执行完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
