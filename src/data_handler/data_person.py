import pandas as pd
import os

def preprocess_individual_data(file_path: str) -> pd.DataFrame:
    """
    加载 CLDS 数据并进行预处理，为结构化估计做准备。

    主要任务:
    1. 读取 clds.csv 数据。
    2. 按个体和年份排序。
    3. 创建上一期位置 'prev_provcd'。
    4. 重命名列以符合后续合并的需要。

    Args:
        file_path (str): clds.csv 文件的路径。

    Returns:
        pd.DataFrame: 预处理后的个体面板数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"个体数据文件未找到: {file_path}")

    df = pd.read_csv(file_path)

    # 为清晰起见，重命名核心列
    # 'provcd' 代表个体在 'year' 这一期所在的省份
    df.rename(columns={'provcd': 'provcd_t', 'year': 'year_t', 'age': 'age_t', 'IID': 'individual_id'}, inplace=True)

    # 确保数据按个体和时间正确排序
    df.sort_values(['individual_id', 'year_t'], inplace=True)

    # 创建上一期的位置 (t-1)
    # 对于每个个体的第一条记录，prev_provcd 将是 NaN
    df['prev_provcd'] = df.groupby('individual_id')['provcd_t'].shift(1)

    # 在模型中，我们通常处理那些有明确起始位置的个体
    # 因此，删除每个个体的第一条观测记录，因为他们的迁移成本无法计算
    df.dropna(subset=['prev_provcd'], inplace=True)
    
    # 将 prev_provcd 转换为整数类型，因为它可能是浮点数
    df['prev_provcd'] = df['prev_provcd'].astype(int)

    print(f"个体数据预处理完成。保留了 {len(df)} 条可用于分析的观测。")
    
    return df


# --- 原有代码保留在此处，以备将来参考 ---

def data_read(data_folder: str) -> pd.DataFrame:
    # ... (original code)
    pass

def data_fix(df: pd.DataFrame) -> pd.DataFrame:
    # ... (original code)
    pass

def main_read(fileloc):
    # ... (original code)
    pass

if __name__ == '__main__':
    # 用于测试新函数的示例代码
    # 假设 clds.csv 在 'data/processed' 目录下
    clds_path = '/Users/victor/F_Repository/KWL/data/processed/clds.csv'
    processed_df = preprocess_individual_data(clds_path)
    print("预处理后的数据样本:")
    print(processed_df.head())
    print("\n检查 'prev_provcd' 列:")
    print(processed_df[['individual_id', 'year_t', 'provcd_t', 'prev_provcd']].head(10))