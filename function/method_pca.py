import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
import os
from factor_analyzer import calculate_kmo

def pca_to_excel(
    input_path,       # 输入文件路径
    output_path,      # 输出文件路径
    variables,        # 需要处理的变量列表
    new_var_name,     # 新变量名称（可追加数字后缀如_PC1）
    sheet_name=0,     # 处理的Sheet页（默认第一个）
    n_components=None,# 主成分数量（默认自动选择解释80%方差的成分）
    missing_strategy='mean',  # 缺失值处理策略：'drop'或'mean'
    overwrite=False   # 是否允许覆盖原文件（需显式开启）
):
    """
    goal:
    使用PCA处理Excel数据并生成新变量
    ---
    input:
    input_path, 输入文件路径
    output_path, 输出文件路径
    variables, 需要处理的变量列表
    new_var_name, 新变量名称（可追加数字后缀如_PC1）
    sheet_name=0, 处理的Sheet页（默认第一个）
    n_components=None,主成分数量（默认自动选择解释80%方差的成分）
    missing_strategy='mean', 缺失值处理策略：'drop'或'mean'
    overwrite=False, 是否允许覆盖原文件（需显式开启）
    ---
    note:
    主成分分析是一种降维技术，通过线性变换将原始数据转换到新的坐标系中，使得新变量（主成分）是原始变量的线性组合，且彼此正交。第一个主成分解释了最大的方差，第二个次之，依此类推。`n_components`参数决定了保留多少个主成分。
    """
    
    # 参数校验
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")
    if input_path == output_path and not overwrite:
        raise ValueError("覆盖原文件需显式设置 overwrite=True")
    
    # 读取数据
    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"读取Excel文件失败：{str(e)}")
    
    # 校验变量是否存在
    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        raise KeyError(f"变量 {missing_vars} 不存在于数据中")
    
    # 预检验数据
    kmo_all, kmo_model = calculate_kmo(df[variables])
    print(f"KMO检验值：{kmo_model:.3f}")  # >0.6适合PCA
    
    corr_matrix = df[variables].corr()
    print('预检验：相关系数矩阵',corr_matrix)
    
    # 提取目标数据
    X = df[variables].copy()
    
    # 缺失值处理
    if X.isnull().sum().sum() > 0:
        if missing_strategy == 'drop':
            X = X.dropna()
            print(f"警告：删除包含缺失值的 {len(df) - len(X)} 行")
        elif missing_strategy == 'mean':
            X = X.fillna(X.mean())
            print("警告：使用均值填充缺失值")
        else:
            raise ValueError("不支持的缺失值处理策略")
    
    # 数据标准化
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
    
    # 生成新列名
    suffix = "" if n_components == 1 else "_PC"
    new_columns = [f"{new_var_name}{suffix}{i+1}" for i in range(n_components)]
    
    # 将结果合并到原始数据
    # 注意处理可能的行数不一致（当使用'drop'策略时）
    if len(df) == len(components):
        df[new_columns] = components
    else:
        print("警告：因删除缺失值导致行数变化，新列将包含NaN")
        for col in new_columns:
            df[col] = np.nan
        df.loc[X.index, new_columns] = components
    
    # 保存结果
    try:
        if output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"处理完成，结果已保存至：{output_path}")
    except Exception as e:
        raise RuntimeError(f"保存文件失败：{str(e)}")
    
    # 检查KMO值（需安装factor_analyzer）
    


if __name__ == "__main__":
    # 示例用法
    pca_to_excel(
        input_path="D:\STUDY\CFPS\geo\geo.xlsx",
        output_path="D:\STUDY\CFPS\geo\geo_updatedd.xlsx",
        variables=['医疗卫生机构数','每万人医疗卫生机构床位数','每万人卫生技术人员','医院平均住院日','地方财政医疗支出 亿元'],  # 需替换为实际变量
        new_var_name="医疗综合指标",
        sheet_name=0,
        n_components=1,
        missing_strategy='mean',
        overwrite=True
    )
