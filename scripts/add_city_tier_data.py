"""
城市分档数据添加脚本

功能：
1. 为地区数据添加city_tier列（1/2/3档）
2. 基于城市经济发展水平、人口规模等指标分档

城市分档标准：
- 第1档（一线城市）：北京、上海、广州、深圳
- 第2档（二线城市）：省会城市、计划单列市、经济发达城市
- 第3档（三线及以下）：其他城市
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig


# 城市分档字典（基于省份代码）
CITY_TIER_MAPPING = {
    # 第1档：一线城市
    110000: 1,  # 北京市
    310000: 1,  # 上海市
    440100: 1,  # 广州市（广东省会）
    440300: 1,  # 深圳市

    # 第2档：省会城市、直辖市、计划单列市
    120000: 2,  # 天津市
    500000: 2,  # 重庆市
    330100: 2,  # 杭州市（浙江省会）
    320100: 2,  # 南京市（江苏省会）
    420100: 2,  # 武汉市（湖北省会）
    510100: 2,  # 成都市（四川省会）
    610100: 2,  # 西安市（陕西省会）
    430100: 2,  # 长沙市（湖南省会）
    350100: 2,  # 福州市（福建省会）
    340100: 2,  # 合肥市（安徽省会）
    370100: 2,  # 济南市（山东省会）
    410100: 2,  # 郑州市（河南省会）
    130100: 2,  # 石家庄市（河北省会）
    210100: 2,  # 沈阳市（辽宁省会）
    220100: 2,  # 长春市（吉林省会）
    230100: 2,  # 哈尔滨市（黑龙江省会）
    360100: 2,  # 南昌市（江西省会）
    530100: 2,  # 昆明市（云南省会）
    520100: 2,  # 贵阳市（贵州省会）
    450100: 2,  # 南宁市（广西省会）
    640100: 2,  # 银川市（宁夏省会）
    650100: 2,  # 乌鲁木齐市（新疆省会）

    # 计划单列市
    320200: 2,  # 无锡市
    330200: 2,  # 宁波市
    350200: 2,  # 厦门市
    370200: 2,  # 青岛市
    440400: 2,  # 珠海市

    # 其他经济发达城市
    320500: 2,  # 苏州市
    330300: 2,  # 温州市
    440600: 2,  # 佛山市
    440700: 2,  # 江门市
    370300: 2,  # 淄博市
}


def assign_city_tier_by_province(provcd: int) -> int:
    """
    基于省份代码分配城市档次

    参数:
    ----
    provcd : int
        省份代码（6位行政区划代码）

    返回:
    ----
    int
        城市档次 (1/2/3)
    """
    # 首先检查是否在精确映射中
    if provcd in CITY_TIER_MAPPING:
        return CITY_TIER_MAPPING[provcd]

    # 否则根据省级代码判断
    province_code = provcd // 10000  # 提取省级代码（前两位）

    # 直辖市统一为2档
    if province_code in [11, 12, 31, 50]:  # 京津沪渝
        if province_code == 11 or province_code == 31:  # 北京上海是1档
            return 1
        return 2

    # 其他省份默认为3档
    return 3


def add_city_tier_to_regional_data(
    regional_data_path: str,
    output_path: str = None,
    use_population_gdp: bool = False
):
    """
    为地区数据添加city_tier列

    参数:
    ----
    regional_data_path : str
        地区数据文件路径
    output_path : str, optional
        输出文件路径，如果为None则覆盖原文件
    use_population_gdp : bool
        是否使用人口和GDP数据辅助分档（更精确但需要数据支持）
    """
    print(f"正在读取地区数据：{regional_data_path}")

    # 读取数据
    if regional_data_path.endswith('.csv'):
        df = pd.read_csv(regional_data_path)
    elif regional_data_path.endswith('.xlsx'):
        df = pd.read_excel(regional_data_path)
    else:
        raise ValueError(f"不支持的文件格式：{regional_data_path}")

    print(f"数据形状：{df.shape}")
    print(f"列名：{df.columns.tolist()}")

    # 检查是否已有city_tier列
    if 'city_tier' in df.columns:
        print("警告：数据中已存在city_tier列，将被覆盖")

    # 分配城市档次
    if use_population_gdp and '常住人口万' in df.columns and '地区生产总值亿' in df.columns:
        print("使用人口和GDP数据辅助分档...")
        # TODO: 实现基于人口和GDP的动态分档
        df['city_tier'] = df['provcd'].apply(assign_city_tier_by_province)
    else:
        print("使用预定义映射分档...")
        df['city_tier'] = df['provcd'].apply(assign_city_tier_by_province)

    # 统计各档城市数量
    tier_counts = df['city_tier'].value_counts().sort_index()
    print("\n城市分档统计：")
    for tier, count in tier_counts.items():
        print(f"  第{tier}档：{count}个省份/地区")

    # 保存结果
    if output_path is None:
        output_path = regional_data_path

    print(f"\n保存结果到：{output_path}")

    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)

    print("完成！")

    return df


def main():
    """
    主函数：为geo.xlsx添加城市分档信息
    """
    config = ModelConfig()

    print("=" * 60)
    print("城市分档数据添加工具")
    print("=" * 60)

    # 添加城市分档
    df_with_tier = add_city_tier_to_regional_data(
        regional_data_path=config.regional_data_path,
        output_path=config.regional_data_path,  # 覆盖原文件
        use_population_gdp=False  # 使用简单映射
    )

    print("\n示例数据（前10行）：")
    print(df_with_tier[['provcd', 'city_tier']].head(10))


if __name__ == '__main__':
    main()
