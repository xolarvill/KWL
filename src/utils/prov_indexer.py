# -*- coding: utf-8 -*-
import pandas
import os
from typing import Dict

class ProvIndexer():
    def __init__(self, config=None):
        """
        初始化ProvIndexer
        
        Args:
            config: ModelConfig对象，如果为None则自动导入
        """
        if config is None:
            from src.config.model_config import ModelConfig
            config = ModelConfig()
        
        # 构建完整的文件路径
        prov_standard_path = config.prov_standard_path
        if not os.path.isabs(prov_standard_path):
            # 如果不是绝对路径，假设它相对于项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            prov_standard_path = os.path.join(project_root, prov_standard_path)
        
        # 读取CSV文件作为pandas dataframe
        self.prov_standard_map = pandas.read_csv(
            prov_standard_path, 
            header=None, 
            names=['code', 'full_code', 'name', 'rank'],
            dtype={'code': str, 'full_code': str, 'name': str, 'rank': int}
        )

    def get_prov_to_idx_map(self) -> Dict[int, int]:
        """
        返回从2位省份代码 (e.g., 11) 到其矩阵索引 (rank-1) 的映射字典。

        Returns:
            Dict[int, int]: 省份代码到索引的映射字典。
        """
        # 确保 'code' 和 'rank' 列是正确的数字类型
        codes = pandas.to_numeric(self.prov_standard_map['code'])
        ranks = pandas.to_numeric(self.prov_standard_map['rank'])
        
        # 创建字典，将rank减1以匹配0基索引
        prov_to_idx = dict(zip(codes, ranks - 1))
        return prov_to_idx

    def _find_nth_element(self, value, n):
        """
        在标准映射表中查找指定值并返回第n列的对应值
        
        Args:
            value: 要查找的值
            n: 要返回的列索引
            
        Returns:
            对应列的值，如果未找到则返回None
        """
        # 在所有列中查找匹配的值
        matching_rows = self.prov_standard_map.isin([str(value)]).any(axis=1)

        if not matching_rows.any():
            return None  # 未找到匹配项
        
        # 获取第一个匹配行的索引
        first_match_index = matching_rows.idxmax()
        
        # 检查 n 是否在列范围内
        if n < 0 or n >= self.prov_standard_map.shape[1]:
            raise IndexError(f"列索引 {n} 超出范围，DataFrame 共有 {self.prov_standard_map.shape[1]} 列。")
        
        # 返回第n列的值
        result = self.prov_standard_map.iloc[first_match_index, n]
        
        # 根据列索引进行类型转换
        if n in [0, 1]:  # code 和 full_code 列应该是整数
            try:
                return int(result)
            except (ValueError, TypeError):
                return result
        elif n == 3:  # rank 列应该是整数
            return int(result)
        else:  # name 列保持为字符串
            return result
            
    def index(self, value, type=None):
        """
        将任意省份标识符转换为标准格式
        
        Args:
            value: 省份标识符（可以是省份名称、2位代码、6位代码等）
            type: 返回值类型
                - None: 默认，返回省份代码，例如11
                - 'code': 返回省份代码，例如11
                - 'full_code': 返回完整省份代码，例如110000
                - 'name': 返回省份名称，例如北京市
                - 'rank': 返回省在矩阵中的排名，例如1
                
        Returns:
            转换后的标准值
        """
        if type is None or type == 'code':
            n = 0
        elif type == 'full_code':
            n = 1
        elif type == 'name':
            n = 2
        elif type == 'rank':
            n = 3
        else:
            raise ValueError(f"未知的类型: {type}")
        
        return self._find_nth_element(value, n)
    
