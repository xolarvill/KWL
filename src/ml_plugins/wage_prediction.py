"""
工资预测模块
使用LightGBM实现非参数工资方程估计
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os

from .kfold_cv import PanelKFoldCrossFitting, create_cross_fitter


class WagePredictionModel:
    """
    工资预测模型类
    
    使用LightGBM实现非参数工资方程估计，捕捉年龄、教育、经验、地区等因素的复杂交互效应
    """
    
    def __init__(self, 
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cv_params: Optional[Dict[str, Any]] = None,
                 feature_engineering: bool = True,
                 log_transform: bool = True):
        """
        初始化工资预测模型
        
        Parameters:
        -----------
        lgb_params : Dict[str, Any], optional
            LightGBM参数
        cv_params : Dict[str, Any], optional
            交叉验证参数
        feature_engineering : bool, default=True
            是否进行特征工程
        log_transform : bool, default=True
            是否对工资进行对数变换
        """
        # 默认LightGBM参数
        self.lgb_params = lgb_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # 默认交叉验证参数
        self.cv_params = cv_params or {
            'n_splits': 5,
            'random_state': 42,
            'individual_col': 'pid',
            'time_col': 'year'
        }
        
        self.feature_engineering = feature_engineering
        self.log_transform = log_transform
        
        # 模型组件
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
        # 预测结果存储
        self.cross_fit_predictions = None
        self.validation_scores = {}
    
    def prepare_features(self, 
                        individual_data: pd.DataFrame,
                        regional_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备工资预测的特征
        
        Parameters:
        -----------
        individual_data : pd.DataFrame
            个体面板数据
        regional_data : pd.DataFrame
            地区特征数据
            
        Returns:
        --------
        pd.DataFrame
            合并后的特征数据
        """
        # 合并个体和地区数据
        merged_data = individual_data.merge(
            regional_data, 
            on=['provcd', 'year'], 
            how='left'
        )
        
        if self.feature_engineering:
            merged_data = self._engineer_features(merged_data)
        
        return merged_data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始数据
            
        Returns:
        --------
        pd.DataFrame
            工程化后的数据
        """
        data = data.copy()
        
        # 年龄相关特征
        if 'age' in data.columns:
            data['age_squared'] = data['age'] ** 2
            data['age_cubed'] = data['age'] ** 3
            
            # 年龄分组
            data['age_group'] = pd.cut(data['age'], 
                                     bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['young', 'early_career', 'mid_career', 'late_career', 'senior'])
        
        # 经验相关特征（如果有教育年限数据）
        if 'education_years' in data.columns and 'age' in data.columns:
            data['experience'] = np.maximum(0, data['age'] - data['education_years'] - 6)
            data['experience_squared'] = data['experience'] ** 2
        
        # 时间趋势
        if 'year' in data.columns:
            base_year = data['year'].min()
            data['time_trend'] = data['year'] - base_year
            data['time_trend_squared'] = data['time_trend'] ** 2
        
        # 地区经济发展水平交互项
        economic_vars = ['health', 'education', 'business', 'environment']
        available_economic_vars = [var for var in economic_vars if var in data.columns]
        
        if len(available_economic_vars) >= 2:
            # 创建经济发展综合指数
            data['economic_index'] = data[available_economic_vars].mean(axis=1)
            
            # 年龄与经济发展水平的交互
            if 'age' in data.columns:
                data['age_economic_interaction'] = data['age'] * data['economic_index']
        
        # 户籍状态特征
        if 'provcd' in data.columns and 'home_province' in data.columns:
            data['is_home_province'] = (data['provcd'] == data['home_province']).astype(int)
        
        # 迁移历史特征（如果有历史居住地信息）
        if 'previous_provcd' in data.columns:
            data['is_migrant'] = (data['provcd'] != data['previous_provcd']).astype(int)
        
        return data
    
    def _prepare_target_variable(self, data: pd.DataFrame, target_col: str = 'wage') -> pd.Series:
        """
        准备目标变量
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        target_col : str, default='wage'
            目标变量列名
            
        Returns:
        --------
        pd.Series
            处理后的目标变量
        """
        y = data[target_col].copy()
        
        # 处理异常值
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # 对数变换
        if self.log_transform:
            # 确保工资为正值
            y = y[y > 0]
            y = np.log(y)
        
        return y
    
    def _select_features(self, data: pd.DataFrame) -> List[str]:
        """
        选择用于建模的特征
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
            
        Returns:
        --------
        List[str]
            特征列名列表
        """
        # 基础特征
        base_features = ['age', 'provcd', 'year']
        
        # 个体特征
        individual_features = [
            'age_squared', 'age_cubed', 'age_group',
            'education_years', 'experience', 'experience_squared',
            'is_home_province', 'is_migrant'
        ]
        
        # 地区特征
        regional_features = [
            'health', 'education', 'business', 'environment',
            'economic_index', 'age_economic_interaction'
        ]
        
        # 时间特征
        time_features = ['time_trend', 'time_trend_squared']
        
        # 选择存在的特征
        all_potential_features = base_features + individual_features + regional_features + time_features
        selected_features = [feat for feat in all_potential_features if feat in data.columns]
        
        return selected_features
    
    def _encode_categorical_features(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        编码分类特征
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        features : List[str]
            特征列表
            
        Returns:
        --------
        pd.DataFrame
            编码后的数据
        """
        data_encoded = data[features].copy()
        
        categorical_features = ['provcd', 'age_group']
        
        for feat in categorical_features:
            if feat in data_encoded.columns:
                if feat not in self.label_encoders:
                    self.label_encoders[feat] = LabelEncoder()
                    data_encoded[feat] = self.label_encoders[feat].fit_transform(data_encoded[feat].astype(str))
                else:
                    # 处理新的类别
                    known_classes = set(self.label_encoders[feat].classes_)
                    new_classes = set(data_encoded[feat].astype(str).unique()) - known_classes
                    
                    if new_classes:
                        # 扩展编码器
                        all_classes = list(known_classes) + list(new_classes)
                        self.label_encoders[feat].classes_ = np.array(all_classes)
                    
                    data_encoded[feat] = self.label_encoders[feat].transform(data_encoded[feat].astype(str))
        
        return data_encoded
    
    def fit_cross_validate(self, 
                          individual_data: pd.DataFrame,
                          regional_data: pd.DataFrame,
                          target_col: str = 'wage') -> Dict[str, float]:
        """
        使用交叉验证拟合模型
        
        Parameters:
        -----------
        individual_data : pd.DataFrame
            个体面板数据
        regional_data : pd.DataFrame
            地区特征数据
        target_col : str, default='wage'
            目标变量列名
            
        Returns:
        --------
        Dict[str, float]
            验证指标
        """
        # 准备数据
        data = self.prepare_features(individual_data, regional_data)
        
        # 准备目标变量
        y = self._prepare_target_variable(data, target_col)
        
        # 确保数据对齐
        valid_idx = y.dropna().index
        data = data.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # 选择特征
        self.feature_names = self._select_features(data)
        
        # 编码特征
        X = self._encode_categorical_features(data, self.feature_names)
        
        # 创建交叉拟合器
        cross_fitter = create_cross_fitter(
            data_type='panel',
            **self.cv_params
        )
        
        # 创建LightGBM模型
        lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        
        # 执行交叉拟合预测
        self.cross_fit_predictions = cross_fitter.cross_fit_predict(X, y, lgb_model)
        
        # 计算验证指标
        valid_predictions = self.cross_fit_predictions[~np.isnan(self.cross_fit_predictions)]
        valid_targets = y[~np.isnan(self.cross_fit_predictions)]
        
        self.validation_scores = {
            'rmse': np.sqrt(mean_squared_error(valid_targets, valid_predictions)),
            'mae': mean_absolute_error(valid_targets, valid_predictions),
            'r2': r2_score(valid_targets, valid_predictions),
            'n_samples': len(valid_predictions)
        }
        
        # 在全数据上训练最终模型
        self.model = lgb.LGBMRegressor(**self.lgb_params)
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self.validation_scores
    
    def predict(self, 
                individual_data: pd.DataFrame,
                regional_data: pd.DataFrame) -> np.ndarray:
        """
        预测工资
        
        Parameters:
        -----------
        individual_data : pd.DataFrame
            个体面板数据
        regional_data : pd.DataFrame
            地区特征数据
            
        Returns:
        --------
        np.ndarray
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit_cross_validate方法")
        
        # 准备数据
        data = self.prepare_features(individual_data, regional_data)
        
        # 编码特征
        X = self._encode_categorical_features(data, self.feature_names)
        
        # 预测
        predictions = self.model.predict(X)
        
        # 如果进行了对数变换，需要反变换
        if self.log_transform:
            predictions = np.exp(predictions)
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        获取特征重要性
        
        Parameters:
        -----------
        importance_type : str, default='gain'
            重要性类型，'gain', 'split', 或 'weight'
            
        Returns:
        --------
        pd.DataFrame
            特征重要性
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Parameters:
        -----------
        filepath : str
            保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'lgb_params': self.lgb_params,
            'cv_params': self.cv_params,
            'feature_engineering': self.feature_engineering,
            'log_transform': self.log_transform,
            'validation_scores': self.validation_scores
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'WagePredictionModel':
        """
        加载模型
        
        Parameters:
        -----------
        filepath : str
            模型文件路径
            
        Returns:
        --------
        WagePredictionModel
            加载的模型实例
        """
        model_data = joblib.load(filepath)
        
        # 创建新实例
        instance = cls(
            lgb_params=model_data['lgb_params'],
            cv_params=model_data['cv_params'],
            feature_engineering=model_data['feature_engineering'],
            log_transform=model_data['log_transform']
        )
        
        # 恢复模型状态
        instance.model = model_data['model']
        instance.label_encoders = model_data['label_encoders']
        instance.feature_names = model_data['feature_names']
        instance.validation_scores = model_data['validation_scores']
        instance.is_fitted = True
        
        return instance


class WageModelEnsemble:
    """
    工资模型集成类
    
    使用多个模型的集成来提高预测精度和稳健性
    """
    
    def __init__(self, 
                 n_models: int = 3,
                 base_params: Optional[Dict[str, Any]] = None):
        """
        初始化模型集成
        
        Parameters:
        -----------
        n_models : int, default=3
            基础模型数量
        base_params : Dict[str, Any], optional
            基础模型参数
        """
        self.n_models = n_models
        self.base_params = base_params or {}
        self.models = []
        self.is_fitted = False
    
    def fit(self, 
            individual_data: pd.DataFrame,
            regional_data: pd.DataFrame,
            target_col: str = 'wage') -> Dict[str, Any]:
        """
        拟合集成模型
        
        Parameters:
        -----------
        individual_data : pd.DataFrame
            个体面板数据
        regional_data : pd.DataFrame
            地区特征数据
        target_col : str, default='wage'
            目标变量列名
            
        Returns:
        --------
        Dict[str, Any]
            集成验证结果
        """
        self.models = []
        ensemble_scores = []
        
        for i in range(self.n_models):
            # 为每个模型创建略微不同的参数
            model_params = self.base_params.copy()
            model_params['lgb_params'] = model_params.get('lgb_params', {}).copy()
            model_params['lgb_params']['random_state'] = 42 + i
            model_params['lgb_params']['bagging_fraction'] = 0.8 + 0.1 * (i / self.n_models)
            
            # 训练模型
            model = WagePredictionModel(**model_params)
            scores = model.fit_cross_validate(individual_data, regional_data, target_col)
            
            self.models.append(model)
            ensemble_scores.append(scores)
        
        self.is_fitted = True
        
        return {
            'individual_scores': ensemble_scores,
            'mean_rmse': np.mean([s['rmse'] for s in ensemble_scores]),
            'mean_r2': np.mean([s['r2'] for s in ensemble_scores]),
            'std_rmse': np.std([s['rmse'] for s in ensemble_scores]),
            'std_r2': np.std([s['r2'] for s in ensemble_scores])
        }
    
    def predict(self, 
                individual_data: pd.DataFrame,
                regional_data: pd.DataFrame) -> np.ndarray:
        """
        集成预测
        
        Parameters:
        -----------
        individual_data : pd.DataFrame
            个体面板数据
        regional_data : pd.DataFrame
            地区特征数据
            
        Returns:
        --------
        np.ndarray
            集成预测结果
        """
        if not self.is_fitted:
            raise ValueError("集成模型尚未拟合")
        
        predictions = []
        for model in self.models:
            pred = model.predict(individual_data, regional_data)
            predictions.append(pred)
        
        # 简单平均集成
        ensemble_prediction = np.mean(predictions, axis=0)
        
        return ensemble_prediction
