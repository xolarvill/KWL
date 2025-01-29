import numpy as np
from pandas import read_excel
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# example
# location_matrix = locmatrix(geodata)
# sorted_provcd = np.sort(geodata['provcd'].unique())
# distance = distance(loc1, loc2)

geodata = read_excel('geodata.xls')

def locmatrix(geodata):
    '''
    根据输入的geodata获取一个按顺序的位置距离矩阵
    基于 WGS84椭球模型，使用Vincenty算法迭代计算两点间的最短测地线距离。考虑地球的扁率（赤道半径6378.137 km，极半径6356.752 km），精度更高。
    '''
    geodata.unique()
    N = geodata['省会'].nunique()
    matrix = np.zeros((N, N))
    for i, city1 in enumerate(geodata['省会'].unique()):
        for j, city2 in enumerate(geodata['省会'].unique()):
            if i != j:
                geolocator = Nominatim(user_agent="geopythesis")
                location_i = geolocator.geocode(city1)
                location_j = geolocator.geocode(city2)
                distance = geodesic(
                    (location_i.longitude, location_i.latitude), (location_j.longitude, location_j.latitude)
                    ).km
                matrix[i, j] = distance # 使用.2f将浮点数格式化为小数点后保留两位数字的形式输出
            if i == j:
                matrix[i, j] = 0
    return matrix


sorted_provcd = np.sort(geodata['provcd'].unique()) # 提前列出排序后的provcd避免每次都需要重新计算
def distance(loc1, loc2 , matrix):
    '''
    loc1 (int):
    loc2 (int):
    matrix (matrix): distance matrix
    '''
    loc1_index = np.where(sorted_provcd == loc1)[0][0]
    loc2_index = np.where(sorted_provcd == loc2)[0][0]
    distance = matrix[loc1_index, loc2_index]
    return distance
    