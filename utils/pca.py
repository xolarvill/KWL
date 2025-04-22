import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def pca(df: pd.DataFrame, new_var_name: str, n_components=None):
    """
    for a given dataframe, use sklearn.decomposition.LCA to create new PCA variable. then add it to the original file.
    """
    
    X = df.copy()
    
    # 确保没有缺失值，否则发出提醒
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 自动确定主成分数量（解释80%方差）
    if n_components is None:
        pca = PCA().fit(X_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cum_var >= 0.8) + 1
        print(f"自动选择 {n_components} 个主成分（累计解释方差：{cum_var[n_components-1]:.1%}）")
        
    # 执行PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    # 线性平移保证不出现非负值
    ## 获取最小值min_score，保证非负值
    min_score = components.min()
    C = abs(min_score) + 0.01
    ## 为所有PCA添加线形位移
    components = components + C
    
    # 生成新列名
    suffix = "" if n_components == 1 else "_PC"
    new_columns = [f"{new_var_name}{suffix}{i+1}" for i in range(n_components)]
    
    # 将结果合并到原始数据
    df[new_columns] = components
    


def pca_to_excel(read_path, columns, new_var_name, save_path):
    """
    选取excel表格，通过指定的列，生成新的PCA指标
    """
    df = pd.read_excel(read_path)
    df = df[columns]
    pca(df, new_var_name)
    df.to_excel(save_path)
    
if __name__ == "__main__":
    pca_to_excel("file/geo.xlsx", ['普高师生比','初中师生比','小学师生比'], "基础教育师生比", "file/pca.xlsx")
    
    
    


