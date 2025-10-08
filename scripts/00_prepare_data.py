
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def create_composite_indicator(df: pd.DataFrame, variables: list, new_indicator_name: str) -> pd.DataFrame:
    """
    使用PCA为一组变量创建一个综合指标。

    参数:
    - df: 包含数据的DataFrame。
    - variables: 用于创建指标的列名列表。
    - new_indicator_name: 新的综合指标列的名称。

    返回:
    - 带有新综合指标列的DataFrame。
    """
    # 确保所有变量都存在
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"以下变量在DataFrame中缺失: {missing_vars}")

    # 提取数据并处理缺失值（使用均值填充）
    data = df[variables].copy()
    for col in data.columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].mean(), inplace=True)

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 应用PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(scaled_data)

    # 为了确保指标的解释性，我们检查第一主成分与原始变量的相关性。
    # 如果大部分相关性为负，说明主成分与指标的“方向”相反（例如，值越小代表越好），
    # 此时我们将其反转，以确保值越大代表越“好”或越“多”。
    loadings = pca.components_[0]
    if np.sum(np.sign(loadings)) < 0:
        principal_component = -principal_component

    # 将主成分添加到DataFrame
    df[new_indicator_name] = principal_component
    
    print(f"创建了综合指标: '{new_indicator_name}'")
    print(f"第一主成分解释的方差比例: {pca.explained_variance_ratio_[0]:.4f}")
    print("-" * 30)
    
    return df

def main():
    """
    主函数，用于读取数据、定义变量组、创建综合指标并保存结果。
    """
    # 定义文件路径
    input_path = '/Users/victor/F_Repository/KWL/data/processed/geo.xlsx'
    output_path = '/Users/victor/F_Repository/KWL/data/processed/geo_amenities.csv'

    # 读取数据
    try:
        df = pd.read_excel(input_path)
        print("--- Original geo.xlsx columns before PCA ---")
        print(df.columns.tolist())
    except FileNotFoundError:
        print(f"错误: 文件未找到 at {input_path}")
        return

    # 定义变量组
    climate_vars = ['温度', '露点温度', '气压', '风向', '风速', '云量', '六小时降水']
    health_vars = ['医疗卫生机构数（反映资源）', '每万人医疗卫生机构床位数', '每万人卫生技术人员（反映医疗服务质量）', '医院平均住院日', '地方财政医疗支出（亿元）（反映投入）']
    education_vars = ['普高师生比', '初中师生比', '小学师生比', '教育经费（万元）', '地方财政教育支出（亿元）（投入）']
    public_services_vars = ['公共汽电车运营总长度 公里', '人均日生活用水', '城市燃气普及率', '每万人拥有公共交通车数量', '每万人拥有公共厕所数量', '人均公园绿地面积']

    # 创建综合指标
    df = create_composite_indicator(df, climate_vars, 'amenity_climate')
    df = create_composite_indicator(df, health_vars, 'amenity_health')
    df = create_composite_indicator(df, education_vars, 'amenity_education')
    df = create_composite_indicator(df, public_services_vars, 'amenity_public_services')

    # 选择要保留的列（原始标识符 + 新的综合指标 + 其他重要变量）
    columns_to_keep = [
        'provcd', 'prov_name', 'year', 'identifier', 'area',
        '常住人口万', '人均可支配收入（元） ', '自然灾害受灾人口万',
        '房价（元每平方）', '房价收入比', '移动电话普及率', '地区基本经济面', '代表性方言',
        '户籍获取难度',  # 新增：三档城市分类（3=一线，2=二线，1=三线）
        'amenity_climate', 'amenity_health', 'amenity_education', 'amenity_public_services'
    ]
    
    # 确保所有要保留的列都存在
    final_columns = [col for col in columns_to_keep if col in df.columns]
    final_df = df[final_columns]

    # 保存处理后的数据
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"处理完成，数据已保存到: {output_path}")

if __name__ == '__main__':
    main()
