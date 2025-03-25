import numpy as np
import pandas as pd

# 个人的位置矩阵
def calK_all_time(dataframe: pd.DataFrame, individual_index: int, option: str) -> np.array:
    
    '''
    goal:
    - This is the personal location matrix. It is used to record the location of each individual in each year. The location is denoted by the 'provcd' column in the dataframe. The location matrix is used to calculate the likelihood of an individual history (including wages and migration decisions).
    - After indexing the location in order the individual appeared, we use the notation calK_it^0 and calK_it^1 to denote the current location and the previous location.
    - Option allows the user to select the current location, the previous location, or both by not inputting.
    
    parameters:
    - dataframe (pandas.DataFrame): The input dataframe containing 'pid', 'year', and 'provcd' columns.
    - individual_index (float): The index of the individual to be selected from the dataframe.
    - option (str): The option to select the current location, the previous location, or both. The default is None.

    return:
    calK_all_time (np.array (2*n)): in each time period, a calK is denoted. The array is transposed (T) to have the shape (2, n), where n is the number of time periods.
    '''
    selected_data = dataframe[dataframe['pid'] == individual_index][['year', 'provcd']]
    selected_data = selected_data.sort_values(by='year')
    calK_all_time = []
    
    for i in range(1, len(selected_data)):
        current_year_provcd = selected_data.iloc[i]['provcd']
        previous_year_provcd = selected_data.iloc[i - 1]['provcd']
        calK_all_time.append([current_year_provcd, previous_year_provcd]) # (2, n)

    calK_all_time = np.array(calK_all_time)

    if option == 'previous': # calK_it^1
        return calK_all_time[:, 0] # (n, )
    elif option == 'current': # calK_it^0
        return calK_all_time[:, 1] # (n, )
    else:
        return calK_all_time


# Example
if __name__ == '__main__':
    import pandas as pd
    a = pd.DataFrame({'pid': [1, 1, 1, 1, 1], 'year': [2000, 2001, 2002, 2003, 2004], 'provcd': [1, 1, 3, 4, 5]})
    print(calK_all_time(dataframe = a, individual_index = 1))
