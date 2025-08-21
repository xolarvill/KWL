import pandas as pd


def topsis(df: pd.DataFrame):
    """
    对给定的DataFrame进行TOPSIS分析，并返回得分最高的行
    """
    