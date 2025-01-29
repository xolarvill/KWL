
def des(dataframe, geodata):
    '''
    descriptive statistics
    '''
    # 针对dataframe给出描述性统计并保存
    dataframe[dataframe['year'] == 2022]['moved'].value_counts()
    dataframe[dataframe['year'] == 2022]['move_add'].value_counts()
    dataframe[dataframe['move_add'] > 0]['move_add'].value_counts()
    
    # 针对geodata给出描述性统计并保存
    
    # 输出
    return