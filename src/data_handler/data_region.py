import pandas as pd
from src.utils import method_entropy, method_pca
import os
from . import houseprice
from src.utils import pca, entropy, topsis


def main_read(directory_path: str) -> pd.DataFrame:
    """
    Reads and processes geographical and socio-economic data from Excel files.
    """
    # ========================= 添加底层数据 ========================= 
    # 添加房价
    houseprice(
        raw_excel_loc = 'data/2000-2022年296个地级以上城市房价数据.xlsx', 
        geodata_loc = 'data/geo.xlsx')
    
    # ========================= 添加指标数据  ========================= 
    # 使用PCA或者熵值法添加教育、医疗、公共交通等指数
    # NOTE: Hardcoded absolute paths below were replaced with relative paths 
    # to ensure the project is portable and runs on different machines.
    method_pca.pca_to_excel(
        input_path="data/geo.xlsx",
        output_path="data/geod.xlsx",
        variables=['医疗卫生机构数','每万人医疗卫生机构床位数','每万人卫生技术人员','医院平均住院日','地方财政医疗支出 亿元'],
        new_var_name="医疗综合指标",
        sheet_name=0,
        n_components=1,
        missing_strategy='mean',
        overwrite=True
    )
    method_pca.pca_to_excel(
        input_path="data/geo.xlsx",
        output_path="data/geod.xlsx",
        variables=['医疗卫生机构数','每万人医疗卫生机构床位数','每万人卫生技术人员','医院平均住院日','地方财政医疗支出 亿元'],  # 需替换为实际变量
        new_var_name="教育综合指标",
        sheet_name=0,
        n_components=1,
        missing_strategy='mean'
    )
    method_pca.pca_to_excel(
        input_path="data/geo.xlsx",
        output_path="data/geod.xlsx",
        variables=['医疗卫生机构数','每万人医疗卫生机构床位数','每万人卫生技术人员','医院平均住院日','地方财政医疗支出 亿元'],  # 需替换为实际变量
        new_var_name="交通综合指标",
        sheet_name=0,
        n_components=1,
        missing_strategy='mean',
        overwrite=True
    )
    method_pca.pca_to_excel(
        input_path="data/geo.xlsx",
        output_path="data/geod.xlsx",
        variables=['医疗卫生机构数','每万人医疗卫生机构床位数','每万人卫生技术人员','医院平均住院日','地方财政医疗支出 亿元'],  # 需替换为实际变量
        new_var_name="天气综合指标",
        sheet_name=0,
        n_components=1,
        missing_strategy='mean',
        overwrite=True
    )
    
    # ========================= 读取 ========================= 
    if not isinstance(directory_path, str):
        raise ValueError("argument must be a string")
    xlsx_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]
    data_frames = [pd.read_excel(os.path.join(directory_path, file)) for file in xlsx_files]
    return pd.concat(data_frames, ignore_index=True)

    
