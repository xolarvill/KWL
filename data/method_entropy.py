import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.preprocessing import MinMaxScaler
import warnings

def entropy_weight_to_excel(file_path, variables, new_var_name, 
                          positive_orient=True, sheet_name=0, output_path=None):
    """
    在Excel文件中为指定变量执行熵值法分析，生成综合得分并添加到原数据
    
    参数：
    file_path: str - 原始Excel文件路径
    variables: list - 需要分析的变量列表
    new_var_name: str - 新列名称
    positive_orient: bool/str/list - 指标方向性处理方式：
        True: 自动检测逆向指标(需提供参考)
        False: 全部正向处理
        list: 手动指定逆向指标列表
    sheet_name: str/int - 工作表名称/索引
    output_path: str - 输出文件路径(None时覆盖原文件)
    
    返回：
    tuple - (包含综合得分的DataFrame, 权重字典)
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
        # 提取目标数据并转换类型
        df = original_df[variables].apply(pd.to_numeric, errors='coerce')
        
        # 处理逆向指标
        if isinstance(positive_orient, list):
            reverse_vars = positive_orient
        elif positive_orient is True:
            # 自动检测逆向指标
            corr_matrix = df.corr().mean()
            reverse_vars = df.columns[corr_matrix < 0].tolist()
        else:
            reverse_vars = []
        
        # 逆向指标正向化处理
        df_processed = df.copy()
        for var in reverse_vars:
            if var in df_processed.columns:
                df_processed[var] = 1 / (df_processed[var] + 1e-8)  # 避免除零

        # 缺失值处理
        df_filled = df_processed.fillna(df_processed.median())

        # 标准化处理 (确保非负)
        scaler = MinMaxScaler(feature_range=(0.001, 1))  # 避免0值
        X_normalized = scaler.fit_transform(df_filled)
        df_normalized = pd.DataFrame(X_normalized, 
                                   columns=df.columns,
                                   index=df.index)
    except Exception as e:
        raise RuntimeError(f"数据预处理失败: {str(e)}")

    # ==================== 熵值法计算 ====================
    # 计算指标比重
    epsilon = 1e-8  # 防止log(0)
    p = df_normalized.div(df_normalized.sum(axis=0) + epsilon, axis=1)

    # 计算信息熵
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entropy = (-1 / np.log(len(df_normalized))) * (p * np.log(p + epsilon)).sum(axis=0)

    # 计算权重
    weights = (1 - entropy) / (1 - entropy).sum()

    # 计算综合得分
    composite_score = (df_normalized * weights).sum(axis=1)
    
    # ==================== 结果整合 ====================
    merged_df = original_df.copy()
    merged_df[new_var_name] = composite_score
    
    # 更新工作表数据
    sheets[sheet_name] = merged_df

    # ==================== 文件保存 ====================
    final_path = output_path if output_path else file_path
    try:
        with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
            for sheet in sheets:
                sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)
    except Exception as e:
        raise RuntimeError(f"文件保存失败: {str(e)}")

    # ==================== 分析报告 ====================
    print("熵值法分析报告：")
    print(f"逆向指标处理: {reverse_vars if reverse_vars else '无'}")
    print("\n指标权重分布：")
    for var, weight in weights.items():
        print(f"{var}: {weight:.4f}")
    
    return merged_df, weights.to_dict()


# 场景1：自动检测逆向指标
# df_auto, _ = entropy_weight_to_excel(
#     "data.xlsx", 
#     ["医院床位数", "死亡率", "就诊等待时间"],
#     "医疗质量指数",
#     positive_orient=True
# )

# 场景2：手动指定逆向指标
# df_manual, weights = entropy_weight_to_excel(
#     "env_data.xlsx",
#     ["PM2.5", "绿化覆盖率", "工业废水排放量"],
#     "环境质量指数",
#     positive_orient=["PM2.5", "工业废水排放量"],
#     output_path="env_result.xlsx"
# )

# 场景3：完全正向处理
# df_positive, _ = entropy_weight_to_excel(
#     "edu_data.xlsx",
#     ["师生比", "升学率", "教育经费"],
#     "教育质量指数",
#     positive_orient=False
# )

if __name__ == "__main__":
    df, weights = entropy_weight_to_excel(
        file_path="D:\STUDY\CFPS\geo\geo.xlsx",
        variables=['医疗卫生机构数','每万人医疗卫生机构床位数','每万人卫生技术人员','医院平均住院日','地方财政医疗支出 亿元'],  # 死亡率需要逆向处理
        new_var_name="医疗综合指数",
        positive_orient=['医院平均住院日'],  # 指定逆向指标
        sheet_name="Sheet1",
        output_path="D:\STUDY\CFPS\geo\geodata_updated.xlsx"
    )