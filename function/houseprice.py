import numpy as np
import pandas as pd


data = pd.read_excel('D:\\STUDY\\2000-2022年296个地级以上城市房价数据.xlsx')
output = []

# 自定义熵值计算函数
def entropy(series):
    probabilities = series.value_counts(normalize=True)
    return -np.sum(probabilities * np.log2(probabilities))

# 使用groupby和apply方法
grouped = data.groupby('省份')
selected = grouped['2010'].apply(entropy).reset_index()


for index, row in selected.iterrows():
    output.append([row['省份'], 2010, row['2010']])

print[output]