import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json
from typing import List
import pandas as pd
from tqdm import tqdm
from src.utils.prov_indexer import ProvIndexer

def distance_matrix(prov_name_list: List[str]) -> np.ndarray:
    """
    根据省份名称列表计算距离矩阵。
    基于WGS84椭球模型，使用Vincenty算法迭代计算两点间的最短测地线距离。考虑地球的扁率精度更高。
    一个二维数组，其中 [i, j] 处的元素表示输入列表中第 i 个和第 j 个省份之间的距离。对角线元素为零。
    """
    # 获取数量
    n: int = len(prov_name_list)
    
    # 将省份名转化为经纬度坐标
    geolocator = Nominatim(user_agent="geopyformythesis")
    prov_geocode_list: List[tuple] = []
    
    for i in range(n):
        prov: str = prov_name_list[i]
        prov_geocode = geolocator.geocode(prov)
        if prov_geocode:
            prov_geocode_list.append((prov_geocode.latitude, prov_geocode.longitude))
        else:
            raise ValueError(f"Geocode not found for province: {prov}")

    # 创建空矩阵
    matrix: np.ndarray = np.zeros((n, n))
    
    # 用geodesic获取物理距离
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = geodesic(prov_geocode_list[i], prov_geocode_list[j]).kilometers
            else:
                matrix[i, j] = 0  # Distance to itself is zero
    
    return matrix

# 存储为csv文件
def save_distance_matrix_to_csv(matrix: np.ndarray, path: str):
    df = pd.DataFrame(matrix)
    df.to_csv(path, index=False, header=False)

if __name__ == '__main__':
    # Wrap the geocoding process with tqdm for progress visualization
    def distance_matrix_with_progress(prov_name_list: List[str]) -> np.ndarray:
        n: int = len(prov_name_list)
        geolocator = Nominatim(user_agent="geopyformythesis")
        prov_geocode_list: List[tuple] = []
        
        for prov in tqdm(prov_name_list, desc="Geocoding provinces"):
            prov_geocode = geolocator.geocode(prov)
            if prov_geocode:
                prov_geocode_list.append((prov_geocode.latitude, prov_geocode.longitude))
            else:
                raise ValueError(f"Geocode not found for province: {prov}")

        matrix: np.ndarray = np.zeros((n, n))
        
        for i in tqdm(range(n), desc="Calculating distances"):
            for j in range(n):
                if i != j:
                    matrix[i, j] = geodesic(prov_geocode_list[i], prov_geocode_list[j]).kilometers
                else:
                    matrix[i, j] = 0
        
        return matrix

    # 使用 ProvIndexer 获取排序好的省份列表
    indexer = ProvIndexer()
    ordered_provinces_df = indexer.prov_standard_map.sort_values(by='rank')
    ordered_provinces = ordered_provinces_df['name'].tolist()

    output_path = 'data/processed/distance_matrix.csv'
    
    matrix = distance_matrix_with_progress(ordered_provinces)
    print(matrix)
    save_distance_matrix_to_csv(matrix, output_path)
    print(f"Distance matrix saved to {output_path}")