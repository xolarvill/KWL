from function import adjacent, distance, linguistic, descriptive, subsample , llh_individual, llh_log_sample, nelder_mead, nelder_mead1, newton_line_search, compare_vec, llh_individual_ds, llh_log_sample_ds, optimal, std
import numpy as np
import sympy as sp
import pandas as pd
from time import time

def main():
    '''
    基于动态最优居住地选择模型对影响10-22年中国劳动力流动的决定因素进行分析
    数据选自CFPS、国家统计局等部门
    '''
    # 原始数据读取、清洗、添加新变量
    ## 基础读取和清洗
    # Cfpsdata = data_person.main_read('D:\\STUDY\\CFPS\\merged')
    # Geodata = data_geo.main_read('D:\\STUDY\\CFPS\\geo')

    # 直接使用已清洗的数据节省时间
    Cfpsdata = pd.read_stata('D:\\STUDY\\CFPS\\merged\\KWL\\data\\cfps10_22mc.dta')
    Geodata = pd.read_excel('D:\\STUDY\\CFPS\\merged\\KWL\\data\\geo\\geo.xls')

    # 给出样本的描述性统计，并写入txt文件中
    description = descriptive.des(dataframe = Cfpsdata, Geodata = Geodata)
    with open('description.txt', 'w') as f:
            f.write(f"{description}")
    
    # 是否要使用子样本进行分类回归
    cut_par = 1
    if cut_par == 1:
        Cfpsdata = subsample.cutout1(Cfpsdata)
    elif cut_par == 2:
        Cfpsdata = subsample.cutout2(Cfpsdata)
    
    # 获取基础参数
    year_list = Cfpsdata['year'].unique() # 年份列表
    pid_list = Cfpsdata['pid'].unique() # 个体ID列表
    provcd_list = Cfpsdata['provcd'].unique() # 省份代码列表
    T = len(year_list) # 总期数
    J = len(provcd_list) #地点数
    I = len(pid_list) # 样本数
    adjacent_matrix = adjacent.adjmatrix(
        adj_path='D:\\STUDY\\CFPS\\merged\\KWL\\data\\geo\\adjacent.xlsx'
        ) # 邻近矩阵
    distance_matrix = distance.dismatrix(
        Geodata=Geodata
        ) # 物理距离矩阵
    linguistic_matrix = linguistic.linmatrix(
        excel_path='',
        json_path='D:\\STUDY\\CFPS\\merged\\KWL\\data\\linguistic.json'
        ) # 文化距离矩阵

    # 从个人轨迹的似然贡献得到个人所有年份的似然函数（包括工资似然和迁移似然），从而估计参数
    # 执行参数估计
    print("Starting parameter estimation...")
    results = optimal.estimate_parameters(Cfpsdata, Geodata, adjacent_matrix, distance_matrix)
    
    # 输出结果
    print("\nParameter Estimates with Standard Errors:")
    print(results.to_dataframe())
    results.save_to_file('parameter_results.tex', format='latex')
    print("\nResults saved to parameter_results.tex")
    
    # 返回程序运行结果        
    print('\nThe algorithm is done')
    print('\nRead readme.md for more information')
    

if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
