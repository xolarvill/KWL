from pandas import read_excel

def adjacent():
    excel = read_excel('D:\\STUDY\\CFPS\\geo\\adjacent.xlsx')
    matrix = excel.values
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix is not square")
    
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            matrix.iloc[j, i] = matrix.iloc[i, j]
    
    # insurance
    for i in range(matrix.shape[0]):
        for j in range(i):
            matrix.iloc[i, j] = matrix.iloc[j, i]
    
    return matrix

if __name__ == '__main__':
    excel = read_excel('D:\\STUDY\\CFPS\\geo\\adjacent\\adjacent.xlsx')
    print(excel)
    matrix = excel.values
    print(matrix.shape)
    
