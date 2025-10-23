"""
临时调试脚本：检查数据中NaN值的来源和分布
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.config.model_config import ModelConfig
from src.utils.prov_indexer import ProvIndexer

def main():
    """
    检查原始数据和处理后数据中的NaN值
    """
    print("=" * 80)
    print("NaN值调试分析")
    print("=" * 80)

    config = ModelConfig()
    indexer = ProvIndexer(config)

    # 1. 读取原始CSV数据（在preprocess_individual_data之前）
    print("\n[1] 检查原始CSV数据...")
    raw_csv_path = config.individual_data_path
    print(f"读取文件: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)
    print(f"原始数据形状: {df_raw.shape}")
    print(f"原始数据列名: {df_raw.columns.tolist()}")

    # 检查provcd列（注意：在preprocess之前还没有重命名）
    provcd_col = 'provcd' if 'provcd' in df_raw.columns else 'provcd_t'
    hukou_col = 'hukou_prov'
    hometown_col = 'hometown'

    print(f"\n[2] 检查各列的NaN情况...")
    print(f"\n列: {provcd_col}")
    print(f"  - 总记录数: {len(df_raw)}")
    print(f"  - NaN数量: {df_raw[provcd_col].isna().sum()}")
    print(f"  - NaN比例: {df_raw[provcd_col].isna().sum() / len(df_raw) * 100:.2f}%")
    if df_raw[provcd_col].isna().any():
        print(f"  - NaN值所在行的样本:")
        nan_rows = df_raw[df_raw[provcd_col].isna()].head(10)
        print(nan_rows[['IID' if 'IID' in df_raw.columns else 'individual_id',
                        'year' if 'year' in df_raw.columns else 'year_t',
                        provcd_col, hukou_col, hometown_col]])

    print(f"\n列: {hukou_col}")
    print(f"  - 总记录数: {len(df_raw)}")
    print(f"  - NaN数量: {df_raw[hukou_col].isna().sum()}")
    print(f"  - NaN比例: {df_raw[hukou_col].isna().sum() / len(df_raw) * 100:.2f}%")
    if df_raw[hukou_col].isna().any():
        print(f"  - NaN值所在行的样本:")
        nan_rows = df_raw[df_raw[hukou_col].isna()].head(10)
        print(nan_rows[['IID' if 'IID' in df_raw.columns else 'individual_id',
                        'year' if 'year' in df_raw.columns else 'year_t',
                        provcd_col, hukou_col, hometown_col]])

    print(f"\n列: {hometown_col}")
    print(f"  - 总记录数: {len(df_raw)}")
    print(f"  - NaN数量: {df_raw[hometown_col].isna().sum()}")
    print(f"  - NaN比例: {df_raw[hometown_col].isna().sum() / len(df_raw) * 100:.2f}%")
    if df_raw[hometown_col].isna().any():
        print(f"  - NaN值所在行的样本:")
        nan_rows = df_raw[df_raw[hometown_col].isna()].head(10)
        print(nan_rows[['IID' if 'IID' in df_raw.columns else 'individual_id',
                        'year' if 'year' in df_raw.columns else 'year_t',
                        provcd_col, hukou_col, hometown_col]])

    # 3. 检查数据类型
    print(f"\n[3] 检查数据类型...")
    print(f"{provcd_col} 数据类型: {df_raw[provcd_col].dtype}")
    print(f"{hukou_col} 数据类型: {df_raw[hukou_col].dtype}")
    print(f"{hometown_col} 数据类型: {df_raw[hometown_col].dtype}")

    # 4. 检查唯一值
    print(f"\n[4] 检查唯一值（前20个）...")
    print(f"\n{provcd_col} 的唯一值:")
    unique_provcd = df_raw[provcd_col].dropna().unique()[:20]
    print(f"  数量: {len(df_raw[provcd_col].unique())} (包含NaN)")
    print(f"  样本: {unique_provcd}")

    print(f"\n{hukou_col} 的唯一值:")
    unique_hukou = df_raw[hukou_col].dropna().unique()[:20]
    print(f"  数量: {len(df_raw[hukou_col].unique())} (包含NaN)")
    print(f"  样本: {unique_hukou}")

    print(f"\n{hometown_col} 的唯一值:")
    unique_hometown = df_raw[hometown_col].dropna().unique()[:20]
    print(f"  数量: {len(df_raw[hometown_col].unique())} (包含NaN)")
    print(f"  样本: {unique_hometown}")

    # 5. 检查是否是数字编码还是省份名称
    print(f"\n[5] 检查列是否已经是数字编码...")
    print(f"{provcd_col} 是数字类型: {pd.api.types.is_numeric_dtype(df_raw[provcd_col])}")
    print(f"{hukou_col} 是数字类型: {pd.api.types.is_numeric_dtype(df_raw[hukou_col])}")
    print(f"{hometown_col} 是数字类型: {pd.api.types.is_numeric_dtype(df_raw[hometown_col])}")

    # 6. 如果不是数字，检查映射文件是否存在
    if not pd.api.types.is_numeric_dtype(df_raw[provcd_col]):
        print(f"\n[6] 检查省份映射...")
        
        print(f"\n省份映射表 (前10个):")
        for index, row in indexer.prov_standard_map.head(10).iterrows():
            print(f"  {row['name']} -> {row['code']}")

        # 检查数据中的省份名是否都在映射表中
        print(f"\n[7] 检查是否有未映射的省份名称...")
        for col_name in [provcd_col, hukou_col, hometown_col]:
            if col_name in df_raw.columns:
                unmapped = []
                for val in df_raw[col_name].dropna().unique():
                    if indexer.index(val) is None:
                        unmapped.append(val)

                if unmapped:
                    print(f"\n{col_name} 中未能映射的值:")
                    print(f"  数量: {len(unmapped)}")
                    print(f"  样本: {unmapped[:10]}")
                else:
                    print(f"\n{col_name}: 所有非NaN值都能成功映射")

    # 7. 分析NaN值的个体特征
    print(f"\n[8] 分析含NaN值的个体特征...")
    iid_col = 'IID' if 'IID' in df_raw.columns else 'individual_id'

    # 找出含有NaN的个体ID
    individuals_with_nan = set()
    for col in [provcd_col, hukou_col, hometown_col]:
        nan_individuals = df_raw[df_raw[col].isna()][iid_col].unique()
        individuals_with_nan.update(nan_individuals)

    print(f"含有NaN值的个体数量: {len(individuals_with_nan)}")
    print(f"总个体数量: {df_raw[iid_col].nunique()}")
    print(f"比例: {len(individuals_with_nan) / df_raw[iid_col].nunique() * 100:.2f}%")

    if len(individuals_with_nan) > 0:
        print(f"\n含NaN个体的样本 (最多5个):")
        sample_individuals = list(individuals_with_nan)[:5]
        for ind_id in sample_individuals:
            ind_data = df_raw[df_raw[iid_col] == ind_id]
            print(f"\n  个体ID: {ind_id}")
            print(f"    观测数: {len(ind_data)}")
            year_col = 'year' if 'year' in df_raw.columns else 'year_t'
            if year_col in ind_data.columns:
                print(f"    年份范围: {ind_data[year_col].min()} - {ind_data[year_col].max()}")
            print(f"    {provcd_col} NaN数: {ind_data[provcd_col].isna().sum()}")
            print(f"    {hukou_col} NaN数: {ind_data[hukou_col].isna().sum()}")
            print(f"    {hometown_col} NaN数: {ind_data[hometown_col].isna().sum()}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    main()