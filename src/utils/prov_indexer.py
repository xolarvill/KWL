# -*- coding: utf-8 -*-
import pandas

class ProvIndexer():
    def __init__(self,
                 config: None):
        
        if config == None:
            from src.config.model_config import ModelConfig
            config = ModelConfig
        
        prov_standard = config.prov_standard_path 
        
        # read this csv file as pandas dataframe
        prov_standard_map = pandas.read_csv(prov_standard)

    def _find_nth_element(self, df , value, n):
        df = self.prov_standard_map
        
        matching_rows = df.isin([value]).any(axis=1)

        if not matching_rows.any():
            return None  # 未找到
        
        # 获取第一个匹配行的索引
        first_match_index = matching_rows.idxmax()
        
        # 检查 n 是否在列范围内
        if n < 0 or n >= df.shape[1]:
            raise IndexError(f"列索引 {n} 超出范围，DataFrame 共有 {df.shape[1]} 列。")
        
        return df.iloc[first_match_index, n]
            
    def index(self, value, type: None) -> None:
        """
        type 可以是以下几种:
        - None: 默认，返回省份代码，例如11
        - 'code': 返回省份代码，例如11
        - 'full_code': 返回完整省份代码，例如110000
        - 'name': 返回省份名称，例如北京市
        - 'rank': 返回省在矩阵中的排名，例如1
        """
        if type == None:
            n = 0
        elif type == 'code':
            n = 0
        elif type == 'full_code':
            n = 1
        elif type == 'name':
            n = 2
        elif type == 'rank':
            n = 3
        
        return self._find_nth_element(self.prov_standard_map, value, n)
    
