import pandas as pd
import os

def data_read(data_folder):
    # 读取dta格式数据文件
    data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.dta') and file != 'cfps10_22mc.dta']

    # 合并文件
    df_list = [pd.read_stata(file, convert_categoricals=False) for file in data_files]
    df = pd.append(df_list, ignore_index=True)
    
    return df

def data_fix(df):
    # 删除变量
    df.drop(columns=[
        
        ], inplace=True)

    # 重命名年份列
    df.rename(columns={'cyear': 'year'}, inplace=True)

    # 删除重复行
    df = df.drop_duplicates(subset=['pid', 'year'])

    # 计算每个pid的年份数
    df['years_per_pid'] = df.groupby('pid')['year'].transform('count')

    # 只保留年份数为7的pid
    df = df[df['years_per_pid'] == 7].drop(columns=['years_per_pid'])

    # 排序
    df = df.sort_values(by=['pid', 'year'])

    # 补全省标
    df['provcd12'] = df['provcd'].where(df['year'] == 2012)
    df['provcd10'] = df['provcd'].where(df['year'] == 2010)
    df['provcd22'] = df['provcd'].astype(str).where(df['year'] == 2022)
    df['provcd20'] = df['provcd'].astype(str).where(df['year'] == 2020)
    df['provcd18'] = df['provcd'].astype(str).where(df['year'] == 2018)
    df['provcd16'] = df['provcd'].astype(str).where(df['year'] == 2016)
    df['provcd14'] = df['provcd'].astype(str).where(df['year'] == 2014)

    # 删除临时列
    df.drop(columns=['provcd22', 'provcd20', 'provcd18', 'provcd16', 'provcd14', 'provcd'], inplace=True)

    # 重命名列
    df.rename(columns={'provincecode22': 'provcd22', 'provincecode20': 'provcd20', 'provincecode18': 'provcd18', 'provincecode16': 'provcd16', 'provincecode14': 'provcd14'}, inplace=True)

    # 补全省标
    df['provcd'] = df['provcd22'].where(df['year'] == 2022)
    df['provcd'] = df['provcd'].where(df['year'] == 2020, df['provcd20'])
    df['provcd'] = df['provcd'].where(df['year'] == 2018, df['provcd18'])
    df['provcd'] = df['provcd'].where(df['year'] == 2016, df['provcd16'])
    df['provcd'] = df['provcd'].where(df['year'] == 2014, df['provcd14'])
    df['provcd'] = df['provcd'].where(df['year'] == 2012, df['provcd12'])
    df['provcd'] = df['provcd'].where(df['year'] == 2010, df['provcd10'])

    # 补全性别
    df['rsex'] = df['gender'].astype(str)
    df['Rsex'] = df['rsex'].astype(int)
    df['gender'] = df.groupby('pid')['Rsex'].transform('mean')

    # 补全年龄
    df['age'] = df['age'].where(df['year'] != 2016, df['age'].shift(-1) - 2)
    df['age'] = df['age'].where(df['year'] != 2014, df['age'].shift(-1) - 2)
    df['age'] = df['age'].where(df['year'] != 2012, df['age'].shift(-1) - 2)
    df['age'] = df['age'].where(df['year'] != 2010, df['age'].shift(-1) - 2)

    # 添加移动标识
    df['provcd_count'] = df.groupby('pid')['provcd'].transform(lambda x: (x != x.shift()).cumsum())
    df['moved'] = (df['provcd_count'] != 1).astype(int)
    df['moving'] = (df['provcd'] != df.groupby('pid')['provcd'].shift()).astype(int)
    df['moving'] = df['moving'].where(df['year'] != 2010, 0)
    df['move_add'] = df.groupby('pid')['moving'].cumsum()

    # 修正错误数据
    df.loc[45358, 'age'] = 31
    df.loc[45357, 'age'] = 29
    df.loc[45356, 'age'] = 27
    df.loc[45355, 'age'] = 25
    df.loc[45354, 'age'] = 23

    # 删除在10年开始未成年的个体
    df = df[~((df['year'] == 2010) & (df['age'] < 18))]
    df = df[~((df['year'] == 2012) & (df['age'] < 20))]
    df = df[~((df['year'] == 2014) & (df['age'] < 22))]
    df = df[~((df['year'] == 2016) & (df['age'] < 24))]
    df = df[~((df['year'] == 2018) & (df['age'] < 26))]
    df = df[~((df['year'] == 2020) & (df['age'] < 28))]
    df = df[~((df['year'] == 2022) & (df['age'] < 30))]

    # 退休个体
    df['retirestring'] = df['retire'].astype(str)
    df['Retire'] = df['retirestring'].astype(int)
    df['Retire'] = df['Retire'].fillna(0)
    df['Retire'] = df['Retire'].replace(-8, 0)
    df['retirestat'] = df.groupby('pid')['Retire'].transform('sum')
    df['Retire'] = (df['retirestat'] != 0).astype(int)

    # 教育程度
    df['Education'] = df['cfps2022edu']
    df['Education'] = df.groupby('pid')['Education'].transform(lambda x: x.ffill().bfill())

    # 保存数据
    df.to_stata('cfps10_22mc.dta', write_index=False)

    # 打印一些统计信息
    print(df[df['year'] == 2022]['moved'].value_counts())
    print(df[df['year'] == 2022]['move_add'].value_counts())
    print(df[df['move_add'] > 0]['move_add'].value_counts())


    return df

def main_read(fileloc):
    if not isinstance(fileloc, str):
        raise ValueError("argument must be a string")
    df = data_read(fileloc)
    df = data_fix(df)
    return df

