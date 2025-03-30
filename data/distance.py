import numpy as np
from pandas import read_excel
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# example
# location_matrix = dismatrix(geodata)
# sorted_provcd = np.sort(geodata['provcd'].unique())
# distance = distance(loc1, loc2)

geodata = read_excel('geodata.xls')

def dismatrix(geodata):
    """
    Generate a location distance matrix based on the input geodata.
    根据输入的geodata获取一个按顺序的位置距离矩阵。基于WGS84椭球模型，使用Vincenty算法迭代计算两点间的最短测地线距离。考虑地球的扁率精度更高。
    -----------------
    Parameters:
    geodata (pandas.DataFrame): A DataFrame containing geographical data with a column named '省份' (province).
    -----------------
    Returns:
    numpy.ndarray: A 2D array (matrix) where the element at [i, j] represents the distance between the i-th
                   and j-th unique provinces in the input geodata. The diagonal elements are zero, indicating
                   zero distance between the same locations.
    """
    geodata.unique()
    N = geodata['省份'].nunique()
    matrix = np.zeros((N, N))
    for i, city1 in enumerate(geodata['省份'].unique()):
        for j, city2 in enumerate(geodata['省份'].unique()):
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
    根据已经排序号的sorted_provcd和已经得到的dismatrix进一步直接获取两个地点的距离值
    ------------------------
    input:
    loc1 (int): 省份1provcd
    loc2 (int): 省份2provcd
    matrix (matrix): distance matrix
    ------------------------
    return:
    distance :距离值
    '''
    loc1_index = np.where(sorted_provcd == loc1)[0][0]
    loc2_index = np.where(sorted_provcd == loc2)[0][0]
    distance = matrix[loc1_index, loc2_index]
    return distance

if __name__ == '__main__':
    geolocator = Nominatim(user_agent="geopythesis")
    location_i = geolocator.geocode('浙江省')
    print(location_i.longitude, location_i.latitude)