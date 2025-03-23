def subsample(dataframe, demand):
    subsampled_df = dataframe.groupby('demand').apply(lambda x: x.sample(frac=demand)).reset_index(drop=True)
    return subsampled_df