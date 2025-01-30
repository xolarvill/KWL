import pandas as pd
import os
import method_entropy, method_pca

def read_geo(directory_path):
    # 使用PCA和entropy对geodata中添加相应指标变量
    method_pca.pca_to_excel(directory_path,[],'health_service')
    method_entropy.add_entropy_variable_per_row(directory_path,[],'education_service')
    
    # 读取
    if not isinstance(directory_path, str):
        raise ValueError("argument must be a string")
    xlsx_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]
    data_frames = [pd.read_excel(os.path.join(directory_path, file)) for file in xlsx_files]
    return pd.concat(data_frames, ignore_index=True)

    
df = read_geo('D:\\STUDY\\CFPS\\geo')
print(df.head())