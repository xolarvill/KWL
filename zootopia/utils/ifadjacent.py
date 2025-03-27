def get_province_rank(provcd, provcdlist):
    """获取省份在列表中的排序位置"""
    try:
        return provcdlist.index(provcd)
    except ValueError:
        return -1  # 如果省份不在列表中返回-1


# 首先先判断个体是否迁移，如果是计算迁移成本，此时需要判断是否为邻接省份
idx1 = provcdlist.index(provcd1) #此时省份的索引
idx2 = provcdlist.index(provcd2) #之前省份的索引
ifadjacent = adjacency_matrix[idx1][idx2] #判断是否为邻接省份