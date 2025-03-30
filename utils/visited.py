import pandas as pd

def visited_sequence(pid: int, data: pd.DataFrame):
    # 获取pid的访问地点序列，并按year升序排列
    result = data[data['pid'] == pid][['provcd', 'year']].sort_values(by='year')
    return result
