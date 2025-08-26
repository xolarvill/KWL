import pandas as pd
import numpy as np
import os
import json
from typing import List

# 假设这些模块在当前路径或Python路径中
from src.data_handler import data_person, data_region, adjacent, distance, linguistic
from src.config import ModelConfig

class DataLoader:
    '''
    数据加载器，用于加载个体面板数据、地区特征数据和地区临近矩阵。
    '''
    def __init__(self, config: ModelConfig):
        self.config = config

    def _validate_path(self, path: str, file_description: str) -> None:
        """辅助函数，用于验证文件路径是否存在。"""
        if not path or not isinstance(path, str):
            raise ValueError(f"{file_description} 的路径配置不正确，必须是一个非空的字符串。")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{file_description} 未在指定路径找到: {path}")

    def load_individual_data(self) -> pd.DataFrame:
        """加载并预处理个体面板数据(CFPS)。"""
        path = self.config.individual_data_path
        self._validate_path(path, "个体数据(dta)")
        
        try:
            df_individual = data_person.data_read(path)
            df_individual = data_person.data_fix(df_individual)
        except Exception as e:
            raise RuntimeError(f"读取或处理个体数据时出错 (路径: {path}): {str(e)}")
            
        return df_individual
        
    def load_regional_data(self) -> pd.DataFrame:
        """加载并预处理地区特征数据。"""
        path = self.config.regional_data_path
        self._validate_path(path, "地区特征数据(xlsx)")
        
        try:
            df_region = data_region.main_read(path)
        except Exception as e:
            raise RuntimeError(f"读取或处理地区数据时出错 (路径: {path}): {str(e)}")
            
        return df_region
        
    def load_adjacency_matrix(self) -> np.ndarray:
        """加载地区邻接矩阵。"""
        path = self.config.adjacency_matrix_path
        self._validate_path(path, "地区邻接矩阵(xlsx)")
        
        try:
            adjacency_matrix = pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"读取邻接矩阵时出错 (路径: {path}): {str(e)}")
        
        return adjacency_matrix.to_numpy()
    
    def load_prov_code_ranked(self) -> List[str]:
        """从JSON文件加载有序的省份代码列表。"""
        path = self.config.prov_code_ranked_path
        self._validate_path(path, "省份排名JSON")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                provcd_rank = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            raise RuntimeError(f"读取或解析省份排名JSON时出错 (路径: {path}): {str(e)}")
         
        return provcd_rank

    def load_distance_matrix(self) -> np.ndarray:
        """加载地区距离矩阵。"""
        path = self.config.distance_matrix_path
        self._validate_path(path, "地区距离矩阵(csv)")
        
        try:
            distance_matrix = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"读取距离矩阵时出错 (路径: {path}): {str(e)}")
            
        return distance_matrix.to_numpy()
        
    def load_linguistic_matrix(self) -> np.ndarray:
        """加载语言亲近度矩阵。"""
        path = self.config.linguistic_matrix_path
        self._validate_path(path, "语言亲近度矩阵(csv)")
        
        try:
            linguistic_matrix = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"读取语言亲近度矩阵时出错 (路径: {path}): {str(e)}")

        return linguistic_matrix.to_numpy()


if __name__ == '__main__':
    from src.config import ModelConfig
    config = ModelConfig
    data = DataLoader(config)
    print(data.load_distance_matrix())