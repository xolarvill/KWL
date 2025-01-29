import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize
import warnings

def entropy_weight_to_excel(file_path, variables, new_var_name, 
                          sheet_name=0, output_path=None, 
                          positive=True, winsor_limits=(0.01, 0.01)):
    """
    在Excel文件中为指定变量执行熵值法分析，生成综合指标
    
    参数：
    file_path: str - 原始Excel文件路径
    variables: list - 需要分析的变量列表
    new_var_name: str - 生成的综合指标列名
    sheet_name: str/int - 工作表名称/索引
    output_path: str - 输出文件路径(None时覆盖原文件)
    positive: bool/str/list - 指标方向性处理方式
    winsor_limits: tuple - 异常值缩尾比例
    """
    
    # ==================== 数据读取与校验 ====================
    try:
        with pd.ExcelFile(file_path) as excel:
            sheets = {sheet: excel.parse(sheet) for sheet in excel.sheet_names}
            original_df = sheets[sheet_name].copy()
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")

    missing_vars = [var for var in variables if var not in original_df.columns]
    if missing_vars:
        raise ValueError(f"以下变量不存在: {missing_vars}")

    # ==================== 数据预处理 ====================
    try:
        # 提取目标变量并转换为数值型
        X = original_df[variables].apply(pd.to_numeric, errors='coerce')
        
        # 处理全空列
        invalid_cols = X.columns[X.isna().all()]
        if invalid_cols.any():
            raise ValueError(f"变量包含无效数据: {invalid_cols.tolist()}")

        # 缺失值处理（分位数填充）
        for col in X.columns:
            q25, q75 = X[col].quantile([0.25, 0.75])
            fill_value = (q25 + q75) / 2
            X[col].fillna(fill_value, inplace=True)

        # 异常值处理（缩尾法）
        if winsor_limits:
            X = X.apply(lambda x: winsorize(x, limits=winsor_limits), axis=0)
            X = pd.DataFrame(X, columns=variables)

        # 指标方向处理
        if isinstance(positive, list):
            if len(positive) != len(variables):
                raise ValueError("方向性参数长度与变量数不一致")
            reverse_cols = [variables[i] for i, val in enumerate(positive) if not val]
        elif positive is True:
            reverse_cols = []
        elif positive == 'auto':
            reverse_cols = auto_detect_direction(X)
        else:
            reverse_cols = variables

        # 逆向指标处理
        if reverse_cols:
            X[reverse_cols] = 1 / (X[reverse_cols] + 1e-6)  # 避免除零

        # 归一化处理
        scaler = MinMaxScaler()
        X_normalized = pd.DataFrame(scaler.fit_transform(X), 
                                   columns=variables,
                                   index=original_df.index)
    except Exception as e:
        raise RuntimeError(f"数据预处理失败: {str(e)}")

    # ==================== 熵值法计算 ====================
    try:
        # 计算指标比重
        epsilon = 1e-6  # 避免log(0)
        p = X_normalized.div(X_normalized.sum(axis=0) + epsilon, axis=1)
        
        # 计算熵值
        k = 1 / np.log(len(X_normalized))
        entropy = (-k * (p * np.log(p + epsilon)).sum(axis=0)).values
        
        # 计算权重
        diff_coefficient = 1 - entropy
        weights = diff_coefficient / diff_coefficient.sum()
        
        # 计算综合得分
        score = (X_normalized * weights).sum(axis=1)
    except Exception as e:
        raise RuntimeError(f"熵值计算失败: {str(e)}")

    # ==================== 结果整合 ====================
    original_df[new_var_name] = score
    
    # 保留原始文件格式
    sheets[sheet_name] = original_df
    final_path = output_path if output_path else file_path

    try:
        with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
            for sheet in sheets:
                sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)
    except Exception as e:
        raise RuntimeError(f"文件保存失败: {str(e)}")

    # ==================== 分析报告 ====================
    print("熵值法分析结果：")
    print(f"综合指标列名: {new_var_name}")
    print("变量权重分布：")
    for var, weight in zip(variables, weights):
        print(f"{var}: {weight:.4f}")
    
    return original_df

def auto_detect_direction(df, threshold=0.7):
    """自动检测指标方向性"""
    reverse_cols = []
    corr_matrix = df.corr().abs()
    
    for col in df.columns:
        if (corr_matrix[col] > threshold).sum() > len(df.columns)/2:
            reverse_cols.append(col)
    return reverse_cols

# 使用示例
if __name__ == "__main__":
    df = entropy_weight_to_excel(
        file_path="data.xlsx",
        variables=["医院数", "病床数", "死亡率"],  # 死亡率需要逆向处理
        new_var_name="医疗综合指数",
        sheet_name="医疗数据",
        positive=['auto', 'auto', False]  # 显式指定第三个指标为逆向
    )