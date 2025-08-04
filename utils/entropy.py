import pandas as pd
import numpy as np

def entropy(df: pd.DataFrame, columns: list, new_var_name: str):
    """
    based on a given dataframe, calculate a new entropy variable. then add it to the original file.
    """
    X = df.copy()
    # 确保没有缺失值，否则发出提醒
    if X.isnull().values.any():
        print("Warning: The dataframe contains missing values.")
    
    # 计算entropy值
    entropy_values = {}
    for col in columns:
        # 计算每个类别的频率
        value_counts = X[col].value_counts(normalize=True)
        # 计算entropy
        entropy_val = -sum(p * np.log2(p) for p in value_counts)
        entropy_values[col] = entropy_val
    
    # 确认权重
    total_entropy = sum(entropy_values.values())
    weights = {col: entropy_val/total_entropy for col, entropy_val in entropy_values.items()}
    
    # 综合得分，获得新指标
    X[new_var_name] = 0
    for col in columns:
        # 标准化处理
        X[f"{col}_normalized"] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())
        # 加权求和
        X[new_var_name] += X[f"{col}_normalized"] * weights[col]
        # 删除临时标准化列
        X.drop(f"{col}_normalized", axis=1, inplace=True)
    
    # 添加到原表格
    df[new_var_name] = X[new_var_name]

    
def entropy_to_excel(read_path, columns, new_var_name, save_path):
    """
    选取excel表格，通过指定的列，生成新的entropy指标
    """
    df = pd.read_excel(read_path)
    df = df[columns]
    entropy(df, columns, new_var_name)
    df.to_excel(save_path)  
    
if __name__ == '__main__':
    entropy_to_excel('data/geo.xlsx', ['城市用水普及率','城市燃气普及率','每万人拥有公共交通车数量','每万人拥有公共厕所数量','人均公园绿地面积'], "公共设施", 'data/entropy.xlsx')