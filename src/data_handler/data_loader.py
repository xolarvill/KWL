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

    def create_estimation_dataset_and_state_space(self, simplified_state: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        加载并准备所有数据，创建状态空间，并为观测值添加状态索引。
        """
        print("开始创建估计数据集和状态空间...")

        self.config.individual_data_path = os.path.join(self.config.processed_data_dir, 'clds_preprocessed_with_wages.csv')
        df_individual = self.load_individual_data()
        
        self.config.regional_data_path = os.path.join(self.config.processed_data_dir, 'geo_amenities.csv')
        df_region = self.load_regional_data()

        if not simplified_state:
            raise NotImplementedError("完整状态空间尚未实现。")

        ages = np.arange(self.config.age_min, self.config.age_max + 1)
        locations = sorted(df_region['provcd'].unique())
        location_map = {loc: i for i, loc in enumerate(locations)}
        
        state_space = pd.DataFrame(
            [(age, loc) for age in ages for loc in locations],
            columns=['age', 'prev_provcd']
        )
        state_space['state_index'] = state_space.index
        
        # --- FIX: Add the required index column for vectorization ---
        state_space['prev_provcd_idx'] = state_space['prev_provcd'].map(location_map)
        print(f"创建了简化的状态空间，包含 {len(state_space)} 个状态，并添加了 'prev_provcd_idx' 列。")

        df_individual = pd.merge(
            df_individual,
            state_space,
            left_on=['age_t', 'prev_provcd'],
            right_on=['age', 'prev_provcd'],
            how='left'
        )
        
        df_individual['choice_index'] = df_individual['provcd_t'].map(location_map)

        n_states = len(state_space)
        transition_matrices = {}
        state_map = state_space.set_index(['age', 'prev_provcd'])['state_index']

        for j_idx, j_loc in enumerate(locations):
            P_j = np.zeros((n_states, n_states))
            for s_idx, row in state_space.iterrows():
                current_age, current_loc = row['age'], row['prev_provcd']
                next_age, next_loc = current_age + 1, j_loc
                if (next_age, next_loc) in state_map.index:
                    s_next_idx = state_map.loc[(next_age, next_loc)]
                    P_j[s_idx, s_next_idx] = 1.0
            transition_matrices[j_idx] = P_j
        print(f"构建了 {len(transition_matrices)} 个状态转移矩阵。")
        
        if df_individual['state_index'].isnull().any():
            unmatched_states = df_individual.loc[df_individual['state_index'].isnull(), ['age_t', 'prev_provcd']].drop_duplicates()
            print(f"警告: {df_individual['state_index'].isnull().sum()} 条观测未能匹配到状态。未匹配的状态组合:\n{unmatched_states}")
            print("正在丢弃这些观测。")
            df_individual.dropna(subset=['state_index', 'choice_index'], inplace=True)
            df_individual['state_index'] = df_individual['state_index'].astype(int)
            df_individual['choice_index'] = df_individual['choice_index'].astype(int)

        print("数据集、状态空间和转移矩阵创建完成。")
        return df_individual, df_region, state_space, transition_matrices


if __name__ == '__main__':
    from src.config import ModelConfig
    config = ModelConfig
    data = DataLoader(config)
    print(data.load_individual_data())