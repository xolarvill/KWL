"""
ML插件数据接口模块
提供统一的数据输入输出接口，用于ML插件与主模型之间的数据交换
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pickle
import os
from pathlib import Path

from ..config.model_config import ModelConfig
from ..data_handler.data_loader import DataLoader


@dataclass
class MLDataBundle:
    """
    ML数据包类，封装ML插件所需的所有数据
    """
    individual_data: pd.DataFrame
    regional_data: pd.DataFrame
    adjacency_matrix: np.ndarray
    distance_matrix: np.ndarray
    linguistic_matrix: np.ndarray
    prov_code_ranked: List[str]
    
    # 元数据
    n_individuals: int
    n_regions: int
    n_periods: int
    time_range: Tuple[int, int]
    
    def __post_init__(self):
        """验证数据一致性"""
        self._validate_data()
    
    def _validate_data(self):
        """验证数据包的一致性"""
        # 检查个体数据
        required_individual_cols = ['pid', 'year', 'provcd', 'age']
        missing_cols = [col for col in required_individual_cols if col not in self.individual_data.columns]
        if missing_cols:
            raise ValueError(f"个体数据缺少必要列: {missing_cols}")
        
        # 检查地区数据
        required_regional_cols = ['provcd', 'year']
        missing_cols = [col for col in required_regional_cols if col not in self.regional_data.columns]
        if missing_cols:
            raise ValueError(f"地区数据缺少必要列: {missing_cols}")
        
        # 检查矩阵维度
        if self.adjacency_matrix.shape[0] != self.n_regions:
            raise ValueError(f"邻接矩阵维度不匹配: {self.adjacency_matrix.shape[0]} != {self.n_regions}")
        
        if self.distance_matrix.shape[0] != self.n_regions:
            raise ValueError(f"距离矩阵维度不匹配: {self.distance_matrix.shape[0]} != {self.n_regions}")
    
    def get_subset(self, 
                   individual_ids: Optional[List] = None,
                   provinces: Optional[List] = None,
                   years: Optional[List] = None) -> 'MLDataBundle':
        """
        获取数据子集
        
        Parameters:
        -----------
        individual_ids : List, optional
            个体ID列表
        provinces : List, optional
            省份代码列表
        years : List, optional
            年份列表
            
        Returns:
        --------
        MLDataBundle
            数据子集
        """
        # 筛选个体数据
        individual_subset = self.individual_data.copy()
        
        if individual_ids is not None:
            individual_subset = individual_subset[individual_subset['pid'].isin(individual_ids)]
        
        if provinces is not None:
            individual_subset = individual_subset[individual_subset['provcd'].isin(provinces)]
        
        if years is not None:
            individual_subset = individual_subset[individual_subset['year'].isin(years)]
        
        # 筛选地区数据
        regional_subset = self.regional_data.copy()
        
        if provinces is not None:
            regional_subset = regional_subset[regional_subset['provcd'].isin(provinces)]
        
        if years is not None:
            regional_subset = regional_subset[regional_subset['year'].isin(years)]
        
        # 更新元数据
        new_n_individuals = individual_subset['pid'].nunique()
        new_n_periods = individual_subset['year'].nunique()
        new_time_range = (individual_subset['year'].min(), individual_subset['year'].max())
        
        return MLDataBundle(
            individual_data=individual_subset,
            regional_data=regional_subset,
            adjacency_matrix=self.adjacency_matrix,
            distance_matrix=self.distance_matrix,
            linguistic_matrix=self.linguistic_matrix,
            prov_code_ranked=self.prov_code_ranked,
            n_individuals=new_n_individuals,
            n_regions=self.n_regions,
            n_periods=new_n_periods,
            time_range=new_time_range
        )


@dataclass
class MLPredictionResult:
    """
    ML预测结果类
    """
    predictions: np.ndarray
    prediction_type: str  # 'wage', 'transition_prob', etc.
    model_type: str  # 'lightgbm', 'ensemble', etc.
    cross_validation_scores: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    model_metadata: Optional[Dict[str, Any]] = None
    
    def save(self, filepath: str):
        """保存预测结果"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MLPredictionResult':
        """加载预测结果"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class MLDataInterface:
    """
    ML数据接口类
    
    负责管理ML插件与主模型之间的数据交换
    """
    
    def __init__(self, config: ModelConfig):
        """
        初始化数据接口
        
        Parameters:
        -----------
        config : ModelConfig
            模型配置
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self._data_bundle = None
        self._ml_results = {}
    
    def load_data(self) -> MLDataBundle:
        """
        加载所有ML插件所需的数据
        
        Returns:
        --------
        MLDataBundle
            数据包
        """
        if self._data_bundle is not None:
            return self._data_bundle
        
        print("正在加载ML插件数据...")
        
        # 加载各类数据
        individual_data = self.data_loader.load_individual_data()
        regional_data = self.data_loader.load_regional_data()
        adjacency_matrix = self.data_loader.load_adjacency_matrix()
        distance_matrix = self.data_loader.load_distance_matrix()
        linguistic_matrix = self.data_loader.load_linguistic_matrix()
        prov_code_ranked = self.data_loader.load_prov_code_ranked()
        
        # 计算元数据
        n_individuals = individual_data['pid'].nunique()
        n_regions = len(prov_code_ranked)
        n_periods = individual_data['year'].nunique()
        time_range = (individual_data['year'].min(), individual_data['year'].max())
        
        # 创建数据包
        self._data_bundle = MLDataBundle(
            individual_data=individual_data,
            regional_data=regional_data,
            adjacency_matrix=adjacency_matrix,
            distance_matrix=distance_matrix,
            linguistic_matrix=linguistic_matrix,
            prov_code_ranked=prov_code_ranked,
            n_individuals=n_individuals,
            n_regions=n_regions,
            n_periods=n_periods,
            time_range=time_range
        )
        
        print(f"数据加载完成: {n_individuals}个个体, {n_regions}个地区, {n_periods}个时期")
        
        return self._data_bundle
    
    def prepare_wage_data(self, 
                         target_col: str = 'wage',
                         include_lagged_vars: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备工资预测所需的数据
        
        Parameters:
        -----------
        target_col : str, default='wage'
            目标变量列名
        include_lagged_vars : bool, default=True
            是否包含滞后变量
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (个体数据, 地区数据)
        """
        data_bundle = self.load_data()
        
        individual_data = data_bundle.individual_data.copy()
        regional_data = data_bundle.regional_data.copy()
        
        # 检查目标变量是否存在
        if target_col not in individual_data.columns:
            raise ValueError(f"目标变量 '{target_col}' 不存在于个体数据中")
        
        # 添加滞后变量
        if include_lagged_vars:
            individual_data = self._add_lagged_variables(individual_data)
        
        # 添加地区排名信息
        prov_rank_map = {prov: rank for rank, prov in enumerate(data_bundle.prov_code_ranked)}
        individual_data['prov_rank'] = individual_data['provcd'].map(prov_rank_map)
        regional_data['prov_rank'] = regional_data['provcd'].map(prov_rank_map)
        
        return individual_data, regional_data
    
    def _add_lagged_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加滞后变量
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始数据
            
        Returns:
        --------
        pd.DataFrame
            包含滞后变量的数据
        """
        data = data.copy()
        data = data.sort_values(['pid', 'year'])
        
        # 添加个体层面的滞后变量
        lag_vars = ['wage', 'provcd']
        
        for var in lag_vars:
            if var in data.columns:
                data[f'{var}_lag1'] = data.groupby('pid')[var].shift(1)
                
                # 对于省份代码，创建是否迁移的指示变量
                if var == 'provcd':
                    data['moved'] = (data['provcd'] != data['provcd_lag1']).astype(int)
                    data['moved'] = data['moved'].fillna(0)  # 第一期设为0
        
        return data
    
    def store_ml_result(self, 
                       result_name: str, 
                       result: MLPredictionResult,
                       save_to_disk: bool = True):
        """
        存储ML预测结果
        
        Parameters:
        -----------
        result_name : str
            结果名称
        result : MLPredictionResult
            预测结果
        save_to_disk : bool, default=True
            是否保存到磁盘
        """
        self._ml_results[result_name] = result
        
        if save_to_disk:
            results_dir = Path('results/ml_models')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = results_dir / f"{result_name}.pkl"
            result.save(str(filepath))
            
            print(f"ML结果已保存: {filepath}")
    
    def get_ml_result(self, result_name: str) -> Optional[MLPredictionResult]:
        """
        获取ML预测结果
        
        Parameters:
        -----------
        result_name : str
            结果名称
            
        Returns:
        --------
        MLPredictionResult or None
            预测结果
        """
        if result_name in self._ml_results:
            return self._ml_results[result_name]
        
        # 尝试从磁盘加载
        filepath = Path('results/ml_models') / f"{result_name}.pkl"
        if filepath.exists():
            result = MLPredictionResult.load(str(filepath))
            self._ml_results[result_name] = result
            return result
        
        return None
    
    def get_wage_predictions(self, 
                           model_name: str = 'wage_model') -> Optional[np.ndarray]:
        """
        获取工资预测结果
        
        Parameters:
        -----------
        model_name : str, default='wage_model'
            模型名称
            
        Returns:
        --------
        np.ndarray or None
            工资预测结果
        """
        result = self.get_ml_result(model_name)
        if result and result.prediction_type == 'wage':
            return result.predictions
        return None
    
    def export_for_structural_model(self, 
                                   output_dir: str = 'data/processed/ml_outputs') -> Dict[str, str]:
        """
        导出ML结果供结构模型使用
        
        Parameters:
        -----------
        output_dir : str, default='data/processed/ml_outputs'
            输出目录
            
        Returns:
        --------
        Dict[str, str]
            导出文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for result_name, result in self._ml_results.items():
            # 导出预测结果
            pred_file = output_path / f"{result_name}_predictions.npy"
            np.save(pred_file, result.predictions)
            exported_files[f"{result_name}_predictions"] = str(pred_file)
            
            # 导出特征重要性
            if result.feature_importance is not None:
                importance_file = output_path / f"{result_name}_feature_importance.csv"
                result.feature_importance.to_csv(importance_file, index=False)
                exported_files[f"{result_name}_importance"] = str(importance_file)
            
            # 导出验证分数
            scores_file = output_path / f"{result_name}_scores.json"
            import json
            with open(scores_file, 'w') as f:
                json.dump(result.cross_validation_scores, f, indent=2)
            exported_files[f"{result_name}_scores"] = str(scores_file)
        
        print(f"ML结果已导出到: {output_dir}")
        return exported_files
    
    def create_summary_report(self) -> pd.DataFrame:
        """
        创建ML结果汇总报告
        
        Returns:
        --------
        pd.DataFrame
            汇总报告
        """
        if not self._ml_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for result_name, result in self._ml_results.items():
            summary_row = {
                'model_name': result_name,
                'prediction_type': result.prediction_type,
                'model_type': result.model_type,
                'n_predictions': len(result.predictions),
                **result.cross_validation_scores
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)


def create_ml_interface(config: ModelConfig) -> MLDataInterface:
    """
    工厂函数，创建ML数据接口
    
    Parameters:
    -----------
    config : ModelConfig
        模型配置
        
    Returns:
    --------
    MLDataInterface
        数据接口实例
    """
    return MLDataInterface(config)
