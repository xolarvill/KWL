import pandas as pd

# 1. 读取原始数据
def houseprice_data():
    df = pd.read_excel('data/raw/geo_backup/house_prix.xlsx')

    # 2. 检查列名，确保年份列是数值型（可选：重命名或转换）
    # 假设原始列名是 '省份', '城市', '2000', '2001', ... '2004'
    # 如果列名不是数字，可先转换

    # 获取所有年份列（排除前两列：省份、城市）
    year_cols = df.columns[2:].tolist()

    # 3. 按省份分组，对每个年份列求平均值
    province_avg = df.groupby('省份')[year_cols].mean().reset_index()

    # 4. 重命名列（可选，使列名更清晰）
    # 例如：'2000' -> '2000_平均房价'，但保持简洁也可不改

    # 5. 保存为新的xlsx文件
    output_file = 'province_avg_house_prices.xlsx'
    province_avg.to_excel(output_file, index=False, sheet_name='各省年均房价')

    print(f"✅ 已成功计算并保存各省平均房价至 {output_file}")

if __name__ == "__main__":
    pass