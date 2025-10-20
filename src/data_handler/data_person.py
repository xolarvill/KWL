import pandas as pd
import os
import json # Added for json.load

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

    # 检查省份列是否已经是数字编码
    if pd.api.types.is_numeric_dtype(df['provcd_t']):
        print("省份列已是数字编码，跳过名称映射。")
    else:
        # 加载省份名称列表和省份代码列表
        prov_name_ranked_path = os.path.join(os.path.dirname(file_path), 'prov_name_ranked.json')
        prov_code_ranked_path = os.path.join(os.path.dirname(file_path), 'prov_code_ranked.json')

        with open(prov_name_ranked_path, 'r', encoding='utf-8') as f:
            prov_names = json.load(f)
        with open(prov_code_ranked_path, 'r', encoding='utf-8') as f:
            prov_codes = json.load(f)

        # 创建省份名称到行政区划代码的映射
        prov_name_to_admin_code = {name: code for name, code in zip(prov_names, prov_codes)}

        # 将无效的缺失值编码替换为NaN（在映射之前）
        invalid_values = ['0', '90000', '澳门', '香港']
        for col in ['provcd_t', 'hukou_prov', 'hometown']:
            df[col] = df[col].replace(invalid_values, pd.NA)

        # 将省份名称转换为数字编码
        df['provcd_t'] = df['provcd_t'].map(prov_name_to_admin_code)
        df['hukou_prov'] = df['hukou_prov'].map(prov_name_to_admin_code)
        df['hometown'] = df['hometown'].map(prov_name_to_admin_code)

    # 转换 'gender' 列为数字 (男=1, 女=0)
    df['gender'] = df['gender'].map({'男': 1, '女': 0})

    # 转换 'is_at_hukou' 列为数字 (是=1, 否=0)
    df['is_at_hukou'] = df['is_at_hukou'].map({'是': 1, '否': 0})

    # 确保映射成功，处理可能存在的NaN（未匹配的省份名称）
    initial_len_before_filter = len(df)
    if df['provcd_t'].isnull().any():
        nan_count = df['provcd_t'].isnull().sum()
        print(f"过滤 'provcd_t' 列中的 {nan_count} 条无效/缺失记录")
        df.dropna(subset=['provcd_t'], inplace=True)
    if df['hukou_prov'].isnull().any():
        nan_count = df['hukou_prov'].isnull().sum()
        print(f"过滤 'hukou_prov' 列中的 {nan_count} 条无效/缺失记录")
        df.dropna(subset=['hukou_prov'], inplace=True)
    if df['hometown'].isnull().any():
        nan_count = df['hometown'].isnull().sum()
        print(f"过滤 'hometown' 列中的 {nan_count} 条无效/缺失记录")
        df.dropna(subset=['hometown'], inplace=True)

    filtered_count = initial_len_before_filter - len(df)
    if filtered_count > 0:
        print(f"共过滤掉 {filtered_count} 条含无效省份值的观测。当前剩余 {len(df)} 条观测。")

    # 过滤掉年龄超出配置范围的个体
    # 需要先加载 ModelConfig
    from src.config.model_config import ModelConfig
    config = ModelConfig()
    initial_len = len(df)
    df = df[(df['age_t'] >= config.age_min) & (df['age_t'] <= config.age_max)]
    if len(df) < initial_len:
        print(f"过滤掉 {initial_len - len(df)} 条年龄超出范围的观测。当前剩余 {len(df)} 条观测。")

    # 确保数据按个体和时间正确排序
    df.sort_values(['individual_id', 'year_t'], inplace=True)

    # 创建上一期的位置 (t-1) - 必须在调用 get_compact_state_info 之前
    # 对于每个个体的第一条记录，prev_provcd 将是 NaN
    df['prev_provcd'] = df.groupby('individual_id')['provcd_t'].shift(1)

    # --- 新增：为每个个体创建紧凑的状态空间信息 ---
    def get_compact_state_info(group):
        # 识别所有相关的位置：当前位置、上一期位置、户籍地、家乡
        all_locations = pd.concat([
            group['provcd_t'],
            group['prev_provcd'].dropna(),
            group['hukou_prov'],
            group['hometown']
        ])
        # 获取唯一的、排序的地点列表
        visited_locations = sorted(list(all_locations.unique()))
        # 创建从地点ID到紧凑索引的映射
        location_map = {loc: i for i, loc in enumerate(visited_locations)}
        
        group['visited_locations'] = [visited_locations] * len(group)
        group['location_map'] = [location_map] * len(group)
        return group

    print("为每个个体生成紧凑状态空间信息...")
    # 在应用此函数之前，'prev_provcd' 列必须存在
    df = df.groupby('individual_id').apply(get_compact_state_info, include_groups=False)
    # 重置索引，因为groupby.apply可能会改变索引结构
    df.reset_index(drop=True, inplace=True)
    # --- 新增结束 ---

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