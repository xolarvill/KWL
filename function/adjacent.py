import openpyxl
import numpy as np

def adjacent(excel_location_path):
    '''
    NOTICE: pd.read_excel() cannot work properly here for some unknown reason.
    '''
    workbook = openpyxl.load_workbook(excel_location_path)
    sheet = workbook['Sheet1']
    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(list(row))
    matrix = np.array(data)
    
    if np.array_equal(matrix, matrix.T):
        print("The matrix is already symmetric.")
        return matrix
    else: 
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix is not square")
        
        # upper to lower
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                matrix.iloc[j, i] = matrix.iloc[i, j]
        # insurance, lower to upper
        for i in range(matrix.shape[0]):
            for j in range(i):
                matrix.iloc[i, j] = matrix.iloc[j, i]
    
    return matrix

if __name__ == '__main__':
    workbook = openpyxl.load_workbook('D:\\STUDY\\CFPS\\geo\\adjacent\\adjacent.xlsx')
    sheet = workbook['Sheet1']
    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(list(row))
    matrix = np.array(data)
    print(matrix.shape)
    
    print(adjacent('D:\\STUDY\\CFPS\\geo\\adjacent\\adjacent.xlsx').shape)
        