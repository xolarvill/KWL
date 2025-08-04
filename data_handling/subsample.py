import pandas as pd
def subsample(dataframe, demand: str) -> pd.DataFrame:
    if not isinstance(demand, str):
        raise ValueError('要求必须为字符串')
    elif demand == '1':
        subsampled_df = dataframe[dataframe['age'] < 18]
    elif demand == '2':
        subsampled_df = dataframe[dataframe['age'] < 45]
    elif demand == '3':
        subsampled_df = dataframe[dataframe['age'] < 65]
    return subsampled_df