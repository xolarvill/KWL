import pandas as pd
import numpy as np
import data_person
import data_region
import adjacent
import subsample
import distance
import json  
from config import ModelConfig
from typing import List

class DataLoader:
    '''
    数据加载器，用于加载个体面板数据、地区特征数据和地区临近矩阵。
    ---
    ModelConfig: 模型配置参数，DataLoader读取，之后的函数都需要使用
    ---
    load_individual_data() -> pd.DataFrame: 加载个体面板数据
    load_regional_data(): 加载地区特征数据
    load_adjacency_matrix(): 加载地区临近矩阵
    '''
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def load_individual_data(self) -> pd.DataFrame:
        """加载个体面板数据(CFPS)"""
        # 读取config中指定的路径
        path = self.config.individual_data_path
        subsample_group = self.config.subsample_group
        
        # 优化数据处理，直接读取
        if path is None:
            # 使用默认路径
            path = 'file/cfps10_22mc.dta'
            df_individual = pd.read_stata(path)
        else:
            # 确保路径是字符串类型
            if not isinstance(path, str):
                raise ValueError("路径参数必须是字符串类型")
                
            # 使用pandas读取dta文件
            try:
                df_individual = data_person.data_read(path)
                # 处理数据
                df_individual = data_person.data_fix(df_individual)  
            except Exception as e:
                raise RuntimeError(f"读取或处理数据时出错: {str(e)}")
            
        # 人群子样本处理
        if subsample_group == 1:
            df_individual = subsample.subsample(df_individual, demand = '1')
        elif subsample_group == 2:
            df_individual = subsample.subsample(df_individual, demand = '2')
        elif subsample_group == 3:
            df_individual = subsample.subsample(df_individual, demand = '3')
                
        return df_individual
        
    def load_regional_data(self) -> pd.DataFrame:
        """加载地区特征数据"""
        # 读取config中指定的路径
        path = self.config.region_data_path
        if not isinstance(path, str):
            raise ValueError("路径参数必须是字符串类型")
        
        # 优化数据处理，直接读取
        if path is None:
            df_region = pd.read_excel('file/geo.xlsx')
        else:
            df_region = data_region.main_read(path)
            
        # 返回处理后的数据框
        return df_region
        
    def load_adjacency_matrix(self) -> np.array:
        """加载地区临近矩阵"""
        # 加载并处理临近矩阵
        path = self.config.adjacency_matrix_path
        if not isinstance(path, str):
            raise ValueError("路径参数必须是字符串类型")
        
        adjacent = adjacent.adjmatrix(path)
        # 返回处理后的矩阵
        return adjacent
    
    def load_prov_code_ranked(self) -> List :
        """加载地区排名"""
        path = self.config.prov_code_ranked_path
        if not isinstance(path, str):
            raise ValueError("路径参数必须是字符串类型")

        # 打开json文件并读取为List
        with open(path, 'r') as f:
            provcd_rank = json.load(f)
         
        return provcd_rank

    def load_distance_matrix(self) -> np.ndarray:
        """加载地区距离矩阵"""
        # to save time, use the result of distance.py that stored in file instead of calculating again
        path = self.config.distance_matrix_path # 读取config中指定的路径
        path2 = self.config.prov_name_ranked_path
        
        # 如果路径非空，说明已经计算过距离矩阵，直接读取
        if path is not None:
            if not isinstance(path, str):
                raise ValueError("路径参数必须是字符串类型")
            
            distance_matrix = pd.read_csv(path)
            
            return distance_matrix
        
        # 如果路径为空，说明还没有计算过距离矩阵，需要计算
        else:
            distance_matrix = distance.distance_matrix(path2)
            return distance_matrix