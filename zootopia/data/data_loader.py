import pandas as pd
import numpy as np
import data_person, data_region, distance, adjacent, subsample
from config import ModelConfig

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
            df_individual = df_individual[df_individual['age'] < 18]
        elif subsample_group == 2:
            df_individual = df_individual[df_individual['age'] < 50]
        elif subsample_group == 3:
            df_individual = df_individual[df_individual['age'] < 80]
                
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
        
