# 将判断出后的布尔值转化为数字

def isnot(j, k): # 判断地点j与地点k是否相同
    if j == k:
        return 1
    else:
        return 0

def inverse_isnot(j, k): # 判断地点j与地点k是否不同
    if j != k:
        return 1
    else:
        return 0

def isin(j, K): # 判断地点j是否在地点集k中
    if j in K:
        return 1
    else:
        return 0
    
def inverse_isin(j, K): # 判断地点j是否在地点集k中
    if j in K:
        return 0
    else:
        return 1
    
