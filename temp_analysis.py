
import pandas as pd
import numpy as np

file_path = 'data/processed/clds_preprocessed_with_wages.csv'
df = pd.read_csv(file_path)

print('--- 真实工资 (income) ---')
# 替换0值为NaN以便进行更准确的统计，因为log(0)是未定义的
df['income_no_zero'] = df['income'].replace(0, np.nan)
print(df['income_no_zero'].describe())
print(f"收入为0或缺失的记录数: {df['income_no_zero'].isnull().sum()}")


print('\n--- 预测工资 (wage_predicted) ---')
print(df['wage_predicted'].describe())
