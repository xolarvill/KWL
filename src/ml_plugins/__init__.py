"""
ML插件模块
提供机器学习插件功能，用于非参数估计滋扰函数
"""

from .kfold_cv import (
    KFoldCrossFitting,
    PanelKFoldCrossFitting,
    create_cross_fitter
)

from .wage_prediction import (
    WagePredictionModel,
    WageModelEnsemble
)

from .data_interface import (
    MLDataBundle,
    MLPredictionResult,
    MLDataInterface,
    create_ml_interface
)

__all__ = [
    # K折交叉拟合
    'KFoldCrossFitting',
    'PanelKFoldCrossFitting',
    'create_cross_fitter',
    
    # 工资预测模型
    'WagePredictionModel',
    'WageModelEnsemble',
    
    # 数据接口
    'MLDataBundle',
    'MLPredictionResult',
    'MLDataInterface',
    'create_ml_interface'
]

# 版本信息
__version__ = '0.1.0'
__author__ = 'KWL Project Team'
__description__ = 'Machine Learning plugins for structural econometric models'
