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
            # 调用新的预处理函数
            df_individual = preprocess_individual_data(path)
        except Exception as e:
            raise RuntimeError(f"读取或处理个体数据时出错 (路径: {path}): {str(e)}")
            
        return df_individual
        
    def load_regional_data(self) -> pd.DataFrame:
        """加载并预处理地区特征数据。"""
        # 注意：此方法现在假定地区数据是CSV格式
        path = self.config.regional_data_path
        self._validate_path(path, "地区特征数据(csv)")
        
        try:
            # 修改为读取CSV
            df_region = pd.read_csv(path)
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

    def create_estimation_dataset_and_state_space(self, simplified_state: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        创建用于模型估计的最终数据集、状态空间和转移矩阵。

        Args:
            simplified_state (bool): 是否使用简化的状态空间（age, prev_loc）。

        返回:
            Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
                - df_estimation: 用于估计的面板数据集，包含 state_index 和 choice_index。
                - state_space: 描述模型所有可能状态的DataFrame。
                - transition_matrices: 状态转移矩阵。
        """
        print("开始创建估计数据集和状态空间...")

        # 1. 加载和预处理数据
        df_individual = self.load_individual_data()
        self.config.regional_data_path = self.config.regional_data_path.replace('geo.xlsx', 'geo_amenities.csv')
        df_region = self.load_regional_data()

        # 2. 定义状态变量并创建状态空间
        if not simplified_state:
            raise NotImplementedError("包含历史访问集合的完整状态空间尚未实现。")

        ages = np.arange(df_individual['age_t'].min(), df_individual['age_t'].max() + 1)
        locations = sorted(df_region['provcd'].unique())
        
        state_space = pd.DataFrame(
            [(age, loc) for age in ages for loc in locations],
            columns=['age', 'prev_provcd']
        )
        state_space['state_index'] = state_space.index
        print(f"创建了简化的状态空间，包含 {len(state_space)} 个状态。")

        # 3. 将观测数据映射到状态空间
        df_estimation = pd.merge(
            df_individual,
            state_space,
            left_on=['age_t', 'prev_provcd'],
            right_on=['age', 'prev_provcd'],
            how='left'
        )
        
        # 添加选择索引
        location_map = {loc: i for i, loc in enumerate(locations)}
        df_estimation['choice_index'] = df_estimation['provcd_t'].map(location_map)

        # 4. 构建转移矩阵 (简化版)
        # 在这个简化模型中，年龄是确定性转移的，位置是内生选择的
        # 转移矩阵 P_j (shape: n_states x n_states) 表示：
        # 如果我今天在状态 s (age, prev_loc)，选择了 j，那么明天我会到哪个状态 s' (age+1, j)
        n_states = len(state_space)
        n_choices = len(locations)
        transition_matrices = {}

        # 创建一个从 (age, loc) 到 state_index 的映射，用于快速查找
        state_map = state_space.set_index(['age', 'prev_provcd'])['state_index']

        for j_idx, j_loc in enumerate(locations):
            P_j = np.zeros((n_states, n_states))
            
            for s_idx, row in state_space.iterrows():
                current_age = row['age']
                
                next_age = current_age + 1
                next_loc = j_loc # 下一期的 'prev_loc' 就是本期的选择 'j'
                
                # 查找下一个状态的索引
                if (next_age, next_loc) in state_map.index:
                    s_next_idx = state_map.loc[(next_age, next_loc)]
                    P_j[s_idx, s_next_idx] = 1.0
                # 如果年龄超出范围，则没有下一个状态（吸收态），保持为0
            
            transition_matrices[j_idx] = P_j

        print(f"构建了 {len(transition_matrices)} 个状态转移矩阵。")
        
        # 清理和检查
        if df_estimation['state_index'].isnull().any():
            print("警告: 部分观测未能匹配到状态。正在丢弃这些观测。")
            df_estimation.dropna(subset=['state_index'], inplace=True)
            df_estimation['state_index'] = df_estimation['state_index'].astype(int)

        print("数据集、状态空间和转移矩阵创建完成。")
        return df_estimation, state_space, transition_matrices


if __name__ == '__main__':
    from src.config import ModelConfig
    config = ModelConfig
    data = DataLoader(config)
    print(data.load_individual_data())