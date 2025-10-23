import pandas as pd
import os
import json # Added for json.load

def preprocess_individual_data(file_path: str, prov_to_idx: dict = None) -> pd.DataFrame:
    """
    加载 CLDS 数据并进行预处理，为结构化估计做准备。

    主要任务:
    1. 读取 clds.csv 数据。
    2. 按个体和年份排序。
    3. 创建上一期位置 'prev_provcd'。
    4. 重命名列以符合后续合并的需要。

    Args:
        file_path (str): clds.csv 文件的路径。
        prov_to_idx (dict, optional): 省份代码到索引的映射字典。

    Returns:
        pd.DataFrame: 预处理后的个体面板数据。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"个体数据文件未找到: {file_path}")

    df = pd.read_csv(file_path)

    # 为清晰起见，重命名核心列
    # 'provcd' 代表个体在 'year' 这一期所在的省份
    df.rename(columns={'provcd': 'provcd_t', 'year': 'year_t', 'age': 'age_t', 'IID': 'individual_id'}, inplace=True)

    # 使用ProvIndexer统一处理所有省份编码
    from src.utils.prov_indexer import ProvIndexer
    prov_indexer = ProvIndexer()

    # 将无效的缺失值编码替换为NaN（在标准化之前）
    invalid_values = ['0', '90000', '澳门', '香港']
    for col in ['provcd_t', 'hukou_prov', 'hometown']:
        df[col] = df[col].replace(invalid_values, pd.NA)
    
    # 如果prev_provcd列存在，也需要处理
    if 'prev_provcd' in df.columns:
        df['prev_provcd'] = df['prev_provcd'].replace(invalid_values, pd.NA)

    # 使用ProvIndexer标准化所有省份编码列为统一的2位数字编码
    # 处理当前所在省份
    df['provcd_t'] = df['provcd_t'].apply(lambda x: prov_indexer.index(x) if pd.notna(x) else x)
    
    # 处理户籍省份
    df['hukou_prov'] = df['hukou_prov'].apply(lambda x: prov_indexer.index(x) if pd.notna(x) else x)
    
    # 处理家乡
    df['hometown'] = df['hometown'].apply(lambda x: prov_indexer.index(x) if pd.notna(x) else x)
    
    # 如果prev_provcd列存在，也需要标准化
    if 'prev_provcd' in df.columns:
        df['prev_provcd'] = df['prev_provcd'].apply(lambda x: prov_indexer.index(x) if pd.notna(x) else x)

    # 转换 'gender' 列为数字 (男=1, 女=0)
    df['gender'] = df['gender'].map({'男': 1, '女': 0})

    # 转换 'is_at_hukou' 列为数字 (是=1, 否=0)
    df['is_at_hukou'] = df['is_at_hukou'].map({'是': 1, '否': 0})

    # 确保标准化成功，处理可能存在的NaN（未匹配的省份名称）
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
    def get_compact_state_info(group, name, prov_to_idx):
        """Helper to create compact state space info for one individual."""
        # Step 1: Gather all relevant locations from different columns.
        sources = [
            pd.to_numeric(group['provcd_t'], errors='coerce').dropna(),
            pd.to_numeric(group['prev_provcd'], errors='coerce').dropna(),
            pd.to_numeric(group['hukou_prov'], errors='coerce').dropna(),
            pd.to_numeric(group['hometown'], errors='coerce').dropna()
        ]
        # **FIX for FutureWarning**: Filter out empty Series before concatenation.
        non_empty_sources = [s for s in sources if not s.empty]
        
        if not non_empty_sources:
            # Handle cases where an individual has no valid location data
            group['visited_locations'] = [[]] * len(group)
            group['location_map'] = [{}] * len(group)
            group['individual_id'] = name
            return group

        all_locations = pd.concat(non_empty_sources)
        
        # Step 2: Get unique raw province codes (e.g., 11, 31, 44) and sort them.
        visited_provcds = sorted(list(all_locations.unique().astype(int)))

        # Step 3: Create the mappings.
        if prov_to_idx is not None:
            # `visited_locations` stores the global province indices (0-30) for the utility function.
            visited_locations_indices = sorted([
                prov_to_idx[provcd] for provcd in visited_provcds if provcd in prov_to_idx
            ])
            # **FIX for invalid observations**: `location_map` must map from the raw province code
            # (used for lookup in likelihood.py) to the compact index (0, 1, 2...).
            location_map = {provcd: i for i, provcd in enumerate(visited_provcds)}
            
            group['visited_locations'] = [visited_locations_indices] * len(group)
            group['location_map'] = [location_map] * len(group)
        else:
            # Fallback for backward compatibility.
            location_map = {loc: i for i, loc in enumerate(visited_provcds)}
            group['visited_locations'] = [visited_provcds] * len(group)
            group['location_map'] = [location_map] * len(group)

        group['individual_id'] = name
        return group

    print("为每个个体生成紧凑状态空间信息...")
    df_list = [get_compact_state_info(group, name, prov_to_idx) for name, group in df.groupby('individual_id')]
    df = pd.concat(df_list)
    
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