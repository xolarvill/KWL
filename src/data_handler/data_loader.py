import pandas as pd
import numpy as np
import os
import json
from typing import List, Tuple, Dict
from src.data_handler import data_person, data_region, adjacent, distance, linguistic
from src.data_handler.data_person import preprocess_individual_data
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

    def _valide_all_files(self):
        """
        检查所有文件
        """
        self._validate_path(self.config.individual_data_path, "个体数据(csv)")
        self._validate_path(self.config.regional_data_path, "地区特征数据(csv)")
        self._validate_path(self.config.adjacency_matrix_path, "地区邻接矩阵(xlsx)")
        self._validate_path(self.config.prov_code_ranked_path, "省份排名JSON")
        self._validate_path(self.config.distance_matrix_path, "地区距离矩阵(csv)")
        self._validate_path(self.config.linguistic_data_path, "语言特征数据(xlsx)")
        

    def load_individual_data(self) -> pd.DataFrame:
        """加载并预处理个体面板数据。"""
        path = self.config.individual_data_path
        self._validate_path(path, "个体数据(csv)")
        
        try:
            df_individual = preprocess_individual_data(path)
        except Exception as e:
            raise RuntimeError(f"读取或处理个体数据时出错 (路径: {path}): {str(e)}")
            
        return df_individual
        
    def load_regional_data(self) -> pd.DataFrame:
        """加载并预处理地区特征数据。"""
        path = self.config.regional_data_path
        self._validate_path(path, "地区特征数据(csv)")
        
        try:
            df_region = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"读取或处理地区数据时出错 (路径: {path}): {str(e)}")
            
        return df_region
        
    def load_adjacency_matrix(self) -> np.ndarray:
        """加载地区邻接矩阵。"""
        path = self.config.adjacency_matrix_path
        self._validate_path(path, "地区邻接矩阵(xlsx)")

        try:
            adjacency_df = pd.read_excel(path)
            adjacency_matrix = adjacency_df.iloc[:, 1:].to_numpy(dtype=float)
        except Exception as e:
            raise RuntimeError(f"读取邻接矩阵时出错 (路径: {path}): {str(e)}")

        return adjacency_matrix
    
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
            if path.endswith('.csv'):
                distance_matrix = pd.read_csv(path)
            elif path.endswith('.xlsx'):
                distance_matrix = pd.read_excel(path, index_col=0)
            else:
                raise ValueError(f"不支持的距离矩阵文件格式: {path}")
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

    def create_estimation_dataset_and_state_space(self, simplified_state: bool = False):
        df_individual = self.load_individual_data()
        df_region = self.load_regional_data()

        # 将省级代码映射到0-30的索引
        prov_codes = sorted(df_region['provcd'].unique())
        prov_to_idx = {code: i for i, code in enumerate(prov_codes)}

        df_individual['provcd_idx'] = df_individual['provcd'].map(prov_to_idx)
        df_individual['prev_provcd_idx'] = df_individual['prev_provcd'].map(prov_to_idx)
        # **新增**: 加载户籍和家乡的索引
        df_individual['hukou_prov_idx'] = df_individual['hukou_prov_code'].map(prov_to_idx)
        df_individual['hometown_prov_idx'] = df_individual['hometown_prov_code'].map(prov_to_idx)

        # 创建状态空间
        if simplified_state:
            # 状态空间现在需要包含户籍和家乡信息
            state_space_cols = ['age', 'prev_provcd_idx', 'hukou_prov_idx', 'hometown_prov_idx']
            state_space = df_individual[state_space_cols].drop_duplicates().reset_index(drop=True)
        else:
            # 完整的状态空间（如果需要）
            # ... (可以扩展)
            pass
        
        # 创建转移矩阵 (这里简化为与年龄无关)
        transition_matrices = self._create_simplified_transition_matrices(state_space)

        # 准备输出字典
        state_data = {
            'age': state_space['age'].values,
            'prev_provcd_idx': state_space['prev_provcd_idx'].values,
            'hukou_prov_idx': state_space['hukou_prov_idx'].values,
            'hometown_prov_idx': state_space['hometown_prov_idx'].values
        }

        # 创建简化的转移矩阵（单位矩阵，假设状态转移由选择决定）
        n_states = len(state_space)
        transition_matrices = {i: np.eye(n_states) for i in range(len(prov_codes))}

        return df_individual, state_space, transition_matrices, df_region

    def _create_simplified_transition_matrices(self, state_space: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        创建简化的转移矩阵（占位实现）
        """
        n_states = len(state_space)
        n_choices = 31  # 31个省份
        transition_matrices = {i: np.eye(n_states) for i in range(n_choices)}
        return transition_matrices


if __name__ == '__main__':
    from src.config import ModelConfig
    config = ModelConfig
    data = DataLoader(config)
    print(data.load_individual_data())