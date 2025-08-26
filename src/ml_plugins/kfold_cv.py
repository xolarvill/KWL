"""
K折交叉拟合模块
实现防止过拟合和信息泄露的交叉拟合策略
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Callable, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator
import warnings


class KFoldCrossFitting:
    """
    K折交叉拟合类，用于防止过拟合和信息泄露
    
    主要用于结构模型估计中的滋扰函数估计，确保样本外预测的有效性
    """
    
    def __init__(self, 
                 n_splits: int = 5, 
                 random_state: int = 42,
                 stratify: bool = False,
                 shuffle: bool = True):
        """
        初始化K折交叉拟合器
        
        Parameters:
        -----------
        n_splits : int, default=5
            折数
        random_state : int, default=42
            随机种子
        stratify : bool, default=False
            是否使用分层抽样
        shuffle : bool, default=True
            是否在分割前打乱数据
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify = stratify
        self.shuffle = shuffle
        
        if stratify:
            self.kfold = StratifiedKFold(
                n_splits=n_splits, 
                random_state=random_state, 
                shuffle=shuffle
            )
        else:
            self.kfold = KFold(
                n_splits=n_splits, 
                random_state=random_state, 
                shuffle=shuffle
            )
    
    def cross_fit_predict(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         estimator: BaseEstimator,
                         stratify_col: Optional[pd.Series] = None) -> np.ndarray:
        """
        执行K折交叉拟合预测
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        estimator : BaseEstimator
            估计器（需要实现fit和predict方法）
        stratify_col : pd.Series, optional
            用于分层的列（仅当stratify=True时使用）
            
        Returns:
        --------
        np.ndarray
            样本外预测结果
        """
        n_samples = len(X)
        predictions = np.full(n_samples, np.nan)
        
        # 确定分层变量
        if self.stratify and stratify_col is not None:
            split_generator = self.kfold.split(X, stratify_col)
        else:
            split_generator = self.kfold.split(X)
        
        for fold_idx, (train_idx, test_idx) in enumerate(split_generator):
            # 获取训练和测试数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            # 训练模型
            try:
                estimator_clone = self._clone_estimator(estimator)
                estimator_clone.fit(X_train, y_train)
                
                # 预测
                fold_predictions = estimator_clone.predict(X_test)
                predictions[test_idx] = fold_predictions
                
            except Exception as e:
                warnings.warn(f"第{fold_idx+1}折训练失败: {str(e)}")
                continue
        
        # 检查是否有未预测的样本
        nan_mask = np.isnan(predictions)
        if nan_mask.any():
            warnings.warn(f"有{nan_mask.sum()}个样本未能获得预测值")
        
        return predictions
    
    def cross_fit_predict_proba(self, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               estimator: BaseEstimator,
                               stratify_col: Optional[pd.Series] = None) -> np.ndarray:
        """
        执行K折交叉拟合概率预测（用于分类问题）
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        estimator : BaseEstimator
            估计器（需要实现fit和predict_proba方法）
        stratify_col : pd.Series, optional
            用于分层的列
            
        Returns:
        --------
        np.ndarray
            样本外概率预测结果
        """
        n_samples = len(X)
        n_classes = len(np.unique(y))
        predictions = np.full((n_samples, n_classes), np.nan)
        
        # 确定分层变量
        if self.stratify and stratify_col is not None:
            split_generator = self.kfold.split(X, stratify_col)
        else:
            split_generator = self.kfold.split(X)
        
        for fold_idx, (train_idx, test_idx) in enumerate(split_generator):
            # 获取训练和测试数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            # 训练模型
            try:
                estimator_clone = self._clone_estimator(estimator)
                estimator_clone.fit(X_train, y_train)
                
                # 预测概率
                fold_predictions = estimator_clone.predict_proba(X_test)
                predictions[test_idx] = fold_predictions
                
            except Exception as e:
                warnings.warn(f"第{fold_idx+1}折训练失败: {str(e)}")
                continue
        
        return predictions
    
    def cross_validate_score(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            estimator: BaseEstimator,
                            scoring: Callable[[np.ndarray, np.ndarray], float],
                            stratify_col: Optional[pd.Series] = None) -> Tuple[List[float], float, float]:
        """
        执行K折交叉验证评分
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        estimator : BaseEstimator
            估计器
        scoring : Callable
            评分函数
        stratify_col : pd.Series, optional
            用于分层的列
            
        Returns:
        --------
        Tuple[List[float], float, float]
            (各折得分, 平均得分, 标准差)
        """
        scores = []
        
        # 确定分层变量
        if self.stratify and stratify_col is not None:
            split_generator = self.kfold.split(X, stratify_col)
        else:
            split_generator = self.kfold.split(X)
        
        for fold_idx, (train_idx, test_idx) in enumerate(split_generator):
            # 获取训练和测试数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            try:
                estimator_clone = self._clone_estimator(estimator)
                estimator_clone.fit(X_train, y_train)
                
                # 预测和评分
                predictions = estimator_clone.predict(X_test)
                score = scoring(y_test, predictions)
                scores.append(score)
                
            except Exception as e:
                warnings.warn(f"第{fold_idx+1}折验证失败: {str(e)}")
                continue
        
        if not scores:
            raise RuntimeError("所有折的验证都失败了")
        
        return scores, np.mean(scores), np.std(scores)
    
    def _clone_estimator(self, estimator: BaseEstimator) -> BaseEstimator:
        """
        克隆估计器
        
        Parameters:
        -----------
        estimator : BaseEstimator
            要克隆的估计器
            
        Returns:
        --------
        BaseEstimator
            克隆的估计器
        """
        from sklearn.base import clone
        return clone(estimator)


class PanelKFoldCrossFitting(KFoldCrossFitting):
    """
    面板数据的K折交叉拟合类
    
    专门处理面板数据结构，确保同一个体的不同时期观测不会同时出现在训练集和测试集中
    """
    
    def __init__(self, 
                 n_splits: int = 5, 
                 random_state: int = 42,
                 individual_col: str = 'pid',
                 time_col: str = 'year'):
        """
        初始化面板数据K折交叉拟合器
        
        Parameters:
        -----------
        n_splits : int, default=5
            折数
        random_state : int, default=42
            随机种子
        individual_col : str, default='pid'
            个体标识符列名
        time_col : str, default='year'
            时间列名
        """
        super().__init__(n_splits=n_splits, random_state=random_state, 
                         stratify=False, shuffle=True)
        self.individual_col = individual_col
        self.time_col = time_col
    
    def cross_fit_predict(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         estimator: BaseEstimator,
                         **kwargs) -> np.ndarray:
        """
        执行面板数据的K折交叉拟合预测
        
        确保同一个体的所有时期观测都在同一折中
        """
        # 获取唯一个体
        unique_individuals = X[self.individual_col].unique()
        n_individuals = len(unique_individuals)
        
        # 对个体进行K折分割
        individual_kfold = KFold(
            n_splits=self.n_splits, 
            random_state=self.random_state, 
            shuffle=self.shuffle
        )
        
        n_samples = len(X)
        predictions = np.full(n_samples, np.nan)
        
        for fold_idx, (train_ind_idx, test_ind_idx) in enumerate(individual_kfold.split(unique_individuals)):
            # 获取训练和测试个体
            train_individuals = unique_individuals[train_ind_idx]
            test_individuals = unique_individuals[test_ind_idx]
            
            # 获取对应的样本索引
            train_mask = X[self.individual_col].isin(train_individuals)
            test_mask = X[self.individual_col].isin(test_individuals)
            
            train_idx = X.index[train_mask].tolist()
            test_idx = X.index[test_mask].tolist()
            
            # 获取训练和测试数据
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train = y.loc[train_idx]
            
            # 训练模型
            try:
                estimator_clone = self._clone_estimator(estimator)
                estimator_clone.fit(X_train, y_train)
                
                # 预测
                fold_predictions = estimator_clone.predict(X_test)
                predictions[test_idx] = fold_predictions
                
            except Exception as e:
                warnings.warn(f"第{fold_idx+1}折训练失败: {str(e)}")
                continue
        
        return predictions


def create_cross_fitter(data_type: str = 'cross_section', **kwargs) -> KFoldCrossFitting:
    """
    工厂函数，创建适当的交叉拟合器
    
    Parameters:
    -----------
    data_type : str, default='cross_section'
        数据类型，'cross_section' 或 'panel'
    **kwargs
        传递给交叉拟合器的参数
        
    Returns:
    --------
    KFoldCrossFitting
        交叉拟合器实例
    """
    if data_type == 'panel':
        return PanelKFoldCrossFitting(**kwargs)
    elif data_type == 'cross_section':
        return KFoldCrossFitting(**kwargs)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
