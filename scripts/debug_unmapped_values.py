"""
临时调试脚本：详细检查无法映射的省份值
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
from src.config.model_config import ModelConfig

def main():
    """
    详细检查无法映射的省份值
    """
    print("=" * 80)
    print("未映射省份值详细分析")
    print("=" * 80)

    config = ModelConfig()
    raw_csv_path = config.individual_data_path
    df_raw = pd.read_csv(raw_csv_path)

    # 加载映射表
    processed_dir = os.path.dirname(raw_csv_path)
    prov_name_path = os.path.join(processed_dir, 'prov_name_ranked.json')
    prov_code_path = os.path.join(processed_dir, 'prov_code_ranked.json')

    with open(prov_name_path, 'r', encoding='utf-8') as f:
        prov_names = json.load(f)
    with open(prov_code_path, 'r', encoding='utf-8') as f:
        prov_codes = json.load(f)

    prov_name_to_code = {name: code for name, code in zip(prov_names, prov_codes)}

    print(f"\n映射表中的省份 (共{len(prov_name_to_code)}个):")
    for name, code in prov_name_to_code.items():
        print(f"  {name} -> {code}")

    # 检查每一列的未映射值
    for col_name in ['provcd', 'hukou_prov', 'hometown']:
        print(f"\n{'=' * 80}")
        print(f"列: {col_name}")
        print(f"{'=' * 80}")

        unmapped_values = []
        for val in df_raw[col_name].dropna().unique():
            if val not in prov_name_to_code:
                unmapped_values.append(val)

        if unmapped_values:
            print(f"\n发现 {len(unmapped_values)} 个无法映射的值:")
            for val in unmapped_values:
                count = (df_raw[col_name] == val).sum()
                print(f"\n  值: '{val}'")
                print(f"    - 数据类型: {type(val)}")
                print(f"    - 出现次数: {count}")
                print(f"    - 影响的观测数: {count}")

                # 显示这些观测的样本
                sample_rows = df_raw[df_raw[col_name] == val].head(5)
                print(f"    - 样本记录:")
                for idx, row in sample_rows.iterrows():
                    print(f"      IID={row['IID']}, year={row['year']}, "
                          f"provcd={row['provcd']}, hukou={row['hukou_prov']}, "
                          f"hometown={row['hometown']}")

                # 检查这些个体的其他信息
                affected_individuals = df_raw[df_raw[col_name] == val]['IID'].unique()
                print(f"    - 影响的个体数: {len(affected_individuals)}")

        else:
            print(f"\n✓ 所有非NaN值都能成功映射")

    # 统计总影响
    print(f"\n{'=' * 80}")
    print(f"总结")
    print(f"{'=' * 80}")

    total_affected = 0
    for col_name in ['provcd', 'hukou_prov', 'hometown']:
        col_unmapped = []
        for val in df_raw[col_name].dropna().unique():
            if val not in prov_name_to_code:
                col_unmapped.append(val)

        if col_unmapped:
            affected = df_raw[col_name].isin(col_unmapped).sum()
            total_affected += affected
            print(f"\n{col_name}:")
            print(f"  - 无法映射的值: {col_unmapped}")
            print(f"  - 影响的观测数: {affected}")

    total_nan = df_raw[['provcd', 'hukou_prov', 'hometown']].isna().sum().sum()
    total_observations = len(df_raw)

    print(f"\n总计:")
    print(f"  - 总观测数: {total_observations}")
    print(f"  - NaN值总数: {total_nan}")
    print(f"  - 无法映射的观测数: {total_affected}")
    print(f"  - 需要处理的观测数: {total_nan + total_affected}")
    print(f"  - 需要处理的比例: {(total_nan + total_affected) / total_observations * 100:.2f}%")

    # 检查这些问题值是否可以修正
    print(f"\n{'=' * 80}")
    print(f"建议的修正方案")
    print(f"{'=' * 80}")

    print(f"\n1. '澳门' - 这是一个特别行政区，不在31个省份中")
    print(f"   建议: 删除这些观测（因为模型只考虑31个省份）")

    print(f"\n2. '香港' - 这是一个特别行政区，不在31个省份中")
    print(f"   建议: 删除这些观测（因为模型只考虑31个省份）")

    print(f"\n3. '0' - 这可能是缺失值的编码")
    print(f"   建议: 将'0'视为NaN处理")

    print(f"\n4. '90000' - 这可能是缺失值的编码")
    print(f"   建议: 将'90000'视为NaN处理")

    print(f"\n5. hometown列的NaN值 (87个)")
    print(f"   建议: 这些是真实的缺失值，需要决定是否保留这些个体")


if __name__ == '__main__':
    main()
