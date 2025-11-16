"""
ABM省份过滤器
因为ABM模拟只使用29个省份（不含西藏、海南、港澳台）
"""
import numpy as np
import pandas as pd

# ABM排除的省份列表（西藏、海南、港澳台）
EXCLUDED_PROVINCES = {
    '西藏自治区', '西藏', '拉萨',  # 西藏
    '海南省', '海南', '海口',      # 海南
    '香港特别行政区', '香港',       # 香港
    '澳门特别行政区', '澳门',       # 澳门
    '台湾省', '台湾', '台北'        # 台湾
}

# 对应的province codes（从prov_standard_map）
EXCLUDED_CODES = {'54', '46', '81', '82', '71'}  # 西藏、海南、香港、澳门、台湾


def filter_abm_provinces(prov_standard_df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤ABM使用的省份（29个）
    
    Args:
        prov_standard_df: 原始省份映射DataFrame (31个省份)
        
    Returns:
        filtered_df: 过滤后的DataFrame (29个省份)
    """
    # 过滤掉EXCLUDED_CODES中的省份
    filtered_df = prov_standard_df[~prov_standard_df['code'].astype(str).isin(EXCLUDED_CODES)].copy()
    
    # 重新分配rank（1-29）
    filtered_df['rank'] = range(1, len(filtered_df) + 1)
    
    print(f"ABM省份过滤完成: {len(prov_standard_df)} → {len(filtered_df)}个省份")
    print(f"排除省份: {EXCLUDED_CODES}")
    print(f"新rank范围: {filtered_df['rank'].min()}-{filtered_df['rank'].max()}")
    
    return filtered_df


def create_abm_prov_indexer(config=None):
    """
    创建ABM专用的ProvIndexer（29个省份）
    
    Returns:
        abm_indexer: 过滤后的索引器
        n_regions: ABM地区数量 (29)
    """
    from src.config.model_config import ModelConfig
    from src.utils.prov_indexer import ProvIndexer
    
    if config is None:
        config = ModelConfig()
    
    # 加载标准映射并过滤
    standard_path = config.prov_standard_path
    prov_df = pd.read_csv(
        standard_path,
        header=None,
        names=['code', 'full_code', 'name', 'rank'],
        dtype={'code': str, 'full_code': str, 'name': str, 'rank': int}
    )
    
    # 过滤ABM省份
    abm_prov_df = filter_abm_provinces(prov_df)
    
    # 创建临时文件供ProvIndexer使用
    temp_path = "data/processed/abm_prov_standard_map.csv"
    abm_prov_df.to_csv(temp_path, index=False, header=False)
    
    # 创建索引器（使用临时文件）
    class ABMProvIndexer:
        def __init__(self, df):
            self.prov_standard_map = df
    
    abm_indexer = ABMProvIndexer(abm_prov_df)
    
    # 提供get_prov_to_idx_map方法
    def get_abm_prov_to_idx_map():
        codes = pd.to_numeric(abm_indexer.prov_standard_map['code'])
        ranks = pd.to_numeric(abm_indexer.prov_standard_map['rank'])
        return dict(zip(codes, ranks - 1))  # 转为0基索引
    
    abm_indexer.get_prov_to_idx_map = get_abm_prov_to_idx_map
    abm_indexer.get_prov_to_idx_map = get_abm_prov_to_idx_map
    
    return abm_indexer, len(abm_prov_df)


def test_abm_filter():
    """测试ABM省份过滤"""
    print("="*60)
    print("ABM省份过滤测试")
    print("="*60)
    
    # 读取原始数据
    df = pd.read_csv(
        "data/processed/prov_standard_map.csv",
        header=None,
        names=['code', 'full_code', 'name', 'rank']
    )
    
    print(f"原始省份数: {len(df)}")
    print("前10个省份:")
    print(df[['code', 'name', 'rank']].head(10))
    
    # 过滤
    abm_df = filter_abm_provinces(df)
    
    print(f"\nABM省份数: {len(abm_df)}")
    print("ABM前10个省份:")
    print(abm_df[['code', 'name', 'rank']].head(10))
    
    # 测试映射
    indexer, n_regions = create_abm_prov_indexer()
    prov_to_idx = indexer.get_prov_to_idx_map()
    
    print(f"\nABM地区数量: {n_regions}")
    print("映射示例:")
    for code, idx in list(prov_to_idx.items())[:5]:
        print(f"  省份代码 {code} → 索引 {idx}")
    
    return indexer, n_regions


if __name__ == '__main__':
    test_abm_filter()