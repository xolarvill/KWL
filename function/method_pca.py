import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def pca_to_excel(file_path, variables, new_var_name, 
                n_components=None, sheet_name=0, output_path=None):
    """
    在Excel文件中为指定变量执行PCA分析，并将主成分添加到原数据
    
    参数：
    file_path: str - 原始Excel文件路径
    variables: list - 需要分析的变量列表
    new_var_name: str - 主成分列名前缀
    n_components: int/None - 主成分数量，None时自动确定
    sheet_name: str/int - 工作表名称/索引
    output_path: str - 输出文件路径(None时覆盖原文件)
    
    返回：
    pd.DataFrame - 包含主成分的DataFrame
    """
    
    # ==================== 数据读取与校验 ====================
    try:
        # 读取整个Excel文件结构
        with pd.ExcelFile(file_path) as excel:
            # 保留所有工作表数据
            sheets = {sheet: excel.parse(sheet) for sheet in excel.sheet_names}
            original_df = sheets[sheet_name].copy()
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")

    # 校验变量是否存在
    missing_vars = [var for var in variables if var not in original_df.columns]
    if missing_vars:
        raise ValueError(f"以下变量不存在: {missing_vars}")

    # ==================== 数据预处理 ====================
    try:
        # 提取目标变量并转换为数值型
        X = original_df[variables].apply(pd.to_numeric, errors='coerce')
        
        # 检查无效列
        invalid_cols = X.columns[X.isna().all()]
        if not invalid_cols.empty:
            raise ValueError(f"变量包含非数值数据: {invalid_cols.tolist()}")

        # 缺失值处理 (保留行索引)
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X),
                                columns=X.columns,
                                index=X.index)

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    except Exception as e:
        raise RuntimeError(f"数据预处理失败: {str(e)}")

    # ==================== PCA分析 ====================
    # 自动确定主成分数量
    if n_components is None:
        pca = PCA()
        pca.fit(X_scaled)
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        
        # 自动选择策略
        n_components = np.argmax(cumulative_var >= 0.85) + 1  # 85%方差解释率
        n_components = max(1, min(n_components, X_scaled.shape[1]))  # 确保至少1个
        
        # 重新拟合
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

    # 执行PCA转换
    pca_components = pca.fit_transform(X_scaled)
    
    # ==================== 结果整合 ====================
    # 生成列名
    pc_columns = [f"{new_var_name}_PC{i+1}" for i in range(n_components)]
    
    # 创建结果DataFrame
    df_pca = pd.DataFrame(pca_components, 
                         columns=pc_columns,
                         index=original_df.index)
    
    # 合并到原始数据
    merged_df = pd.concat([original_df, df_pca], axis=1)
    sheets[sheet_name] = merged_df

    # ==================== 结果保存 ====================
    final_path = output_path if output_path else file_path
    
    try:
        # 使用openpyxl保留原文件格式
        with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
            # 写入所有工作表
            for sheet in sheets:
                # 处理原文件格式
                if sheet == sheet_name:
                    sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)
                else:
                    # 直接写入原数据保持格式
                    sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)
    except Exception as e:
        raise RuntimeError(f"文件保存失败: {str(e)}")

    # ==================== 分析报告 ====================
    print(f"成功生成{n_components}个主成分")
    print("各主成分方差解释率:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.2%}")
    
    return merged_df

# 使用示例
if __name__ == "__main__":
    # 示例调用
    df = pca_to_excel(
        file_path="data.xlsx",
        variables=["教育经费", "师生比", "升学率"],
        new_var_name="Edu_PC",
        sheet_name="教育数据"
    )