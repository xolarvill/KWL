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
        

    def load_individual_data(self, prov_to_idx: dict = None) -> pd.DataFrame:
        """
        加载并预处理个体面板数据。

        Args:
            prov_to_idx (dict, optional): 省份代码到索引的映射字典。

        Returns:
            pd.DataFrame: 预处理后的个体数据
        """
        path = self.config.individual_data_path
        self._validate_path(path, "个体数据(csv)")

        try:
            df_individual = preprocess_individual_data(path, prov_to_idx=prov_to_idx)
        except Exception as e:
            raise RuntimeError(f"读取或处理个体数据时出错 (路径: {path}): {str(e)}")

        return df_individual
        
    def load_regional_data(self) -> pd.DataFrame:
        """
        加载并合并地区特征数据。

        合并两个数据源：
        1. geo.xlsx - 原始数据（人口、收入、房价等）
        2. geo_amenities.csv - PCA综合amenity指标

        Returns:
            合并后的DataFrame，包含原始数据和amenity指标
        """
        # 1. 加载原始地区数据
        path_raw = self.config.regional_data_path
        self._validate_path(path_raw, "原始地区数据")

        try:
            if path_raw.endswith('.csv'):
                df_raw = pd.read_csv(path_raw)
            elif path_raw.endswith('.xlsx') or path_raw.endswith('.xls'):
                df_raw = pd.read_excel(path_raw)
            else:
                raise ValueError(f"不支持的文件格式: {path_raw}。仅支持.csv或.xlsx格式")
        except Exception as e:
            raise RuntimeError(f"读取原始地区数据时出错 (路径: {path_raw}): {str(e)}")

        # 2. 加载amenity综合指标数据
        path_amenity = self.config.regional_amenity_path
        self._validate_path(path_amenity, "amenity综合指标数据")

        try:
            df_amenity = pd.read_csv(path_amenity)
        except Exception as e:
            raise RuntimeError(f"读取amenity数据时出错 (路径: {path_amenity}): {str(e)}")

        # 3. 合并两个数据源
        # 使用provcd和year作为合并键
        amenity_cols = ['provcd', 'year'] + [col for col in df_amenity.columns if 'amenity' in col]
        df_amenity_subset = df_amenity[amenity_cols]

        try:
            df_region = pd.merge(
                df_raw,
                df_amenity_subset,
                on=['provcd', 'year'],
                how='left',
                validate='1:1'
            )
        except Exception as e:
            raise RuntimeError(f"合并原始数据和amenity数据时出错: {str(e)}")

        # 4. 检查合并结果
        amenity_columns = [col for col in df_region.columns if 'amenity' in col]
        if not amenity_columns:
            raise RuntimeError("警告：合并后的数据中没有找到amenity列！")

        # 5. 创建房价收入比指标（如果不存在）
        if '房价收入比' not in df_region.columns and '房价（元每平方）' in df_region.columns and '人均可支配收入（元） ' in df_region.columns:
            df_region['房价收入比'] = df_region['房价（元每平方）'] / df_region['人均可支配收入（元） ']

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
        """
        创建估计数据集和状态空间

        Returns:
            df_individual: 包含state_index和choice_index列的个体数据
            state_space: 状态空间DataFrame，包含state_index列
            transition_matrices: 转移矩阵字典
            df_region: 地区数据
        """
        # 先加载地区数据以创建prov_to_idx映射
        df_region = self.load_regional_data()

        # 将省级代码映射到0-30的索引
        prov_codes = sorted(df_region['provcd'].unique())
        prov_to_idx = {code: i for i, code in enumerate(prov_codes)}

        # 传递prov_to_idx给个体数据加载函数
        df_individual = self.load_individual_data(prov_to_idx=prov_to_idx)

        # 注意：preprocess_individual_data()将provcd重命名为provcd_t，IID重命名为individual_id
        df_individual['provcd_idx'] = df_individual['provcd_t'].map(prov_to_idx)
        df_individual['prev_provcd_idx'] = df_individual['prev_provcd'].map(prov_to_idx)
        # **新增**: 加载户籍和家乡的索引
        df_individual['hukou_prov_idx'] = df_individual['hukou_prov'].map(prov_to_idx)
        df_individual['hometown_prov_idx'] = df_individual['hometown'].map(prov_to_idx)

        # 创建状态空间
        if simplified_state:
            # 状态空间现在需要包含户籍和家乡信息
            state_space_cols = ['age_t', 'prev_provcd_idx', 'hukou_prov_idx', 'hometown_prov_idx']
            state_space = df_individual[state_space_cols].drop_duplicates().reset_index(drop=True)
            # 重命名age_t为age以保持与Bellman求解器的兼容性
            state_space = state_space.rename(columns={
                'age_t': 'age',
                'prev_provcd_idx': 'prev_provcd_idx',
                'hukou_prov_idx': 'hukou_prov_idx',
                'hometown_prov_idx': 'hometown_prov_idx'
            })
            # 添加state_index列
            state_space['state_index'] = state_space.index
        else:
            # 完整的状态空间（如果需要）
            # ... (可以扩展)
            pass

        # 将df_individual与state_space合并以添加state_index
        # 合并键: age_t, prev_provcd_idx, hukou_prov_idx, hometown_prov_idx
        df_individual = pd.merge(
            df_individual,
            state_space[['age', 'prev_provcd_idx', 'hukou_prov_idx', 'hometown_prov_idx', 'state_index']],
            left_on=['age_t', 'prev_provcd_idx', 'hukou_prov_idx', 'hometown_prov_idx'],
            right_on=['age', 'prev_provcd_idx', 'hukou_prov_idx', 'hometown_prov_idx'],
            how='left'
        )

        # 删除合并产生的重复的'age'列
        if 'age' in df_individual.columns:
            df_individual = df_individual.drop(columns=['age'])

        # 检查是否有未匹配的观测
        unmatched_count = df_individual['state_index'].isnull().sum()
        if unmatched_count > 0:
            print(f"警告: {unmatched_count} 条观测无法匹配到状态空间，将被过滤")
            df_individual = df_individual.dropna(subset=['state_index'])

        # 添加choice_index列（当期选择的省份索引）
        df_individual['choice_index'] = df_individual['provcd_idx']

        # 确保state_index和choice_index是整数类型
        df_individual['state_index'] = df_individual['state_index'].astype(int)
        df_individual['choice_index'] = df_individual['choice_index'].astype(int)

        # 创建转移矩阵 (这里简化为与年龄无关)
        transition_matrices = self._create_simplified_transition_matrices(state_space)

        # 准备输出字典（已废弃，state_space现在直接作为DataFrame使用）

        # 创建简化的转移矩阵（单位矩阵，假设状态转移由选择决定）
        n_states = len(state_space)
        transition_matrices = {i: np.eye(n_states) for i in range(len(prov_codes))}

        print(f"数据准备完成: {len(df_individual)} 条观测, {len(state_space)} 个状态")

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