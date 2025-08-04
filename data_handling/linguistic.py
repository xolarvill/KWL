import json
import pandas as pd
import numpy as np


class TreeNode:
    def __init__(self, name):
        self.name = name      # 节点名称（如方言或语系名）
        self.parent = None    # 父节点
        self.children = []    # 子节点列表
        self.depth = 0        # 节点深度（根节点深度为0）

# 递归构建树的代码
def build_tree(data, parent=None):
    node = TreeNode(data["name"])
    node.parent = parent
    node.depth = parent.depth + 1 if parent else 0
    if "children" in data:
        for child_data in data["children"]:
            child_node = build_tree(child_data, parent=node)
            node.children.append(child_node)
    return node

# 调整节点深度至同一层
def _move_to_depth(node, target_depth):
    while node.depth > target_depth:
        node = node.parent
    return node

# 查找最近公共祖先
def find_lca(node1, node2):
    # 调整到同一深度
    if node1.depth > node2.depth:
        node1 = _move_to_depth(node1, node2.depth)
    else:
        node2 = _move_to_depth(node2, node1.depth)
    
    # 同步上移直到找到共同祖先
    while node1 != node2:
        node1 = node1.parent
        node2 = node2.parent
    return node1

# 亲疏关系计算
def calculate_distance(node1, node2):
    lca = find_lca(node1, node2)
    distance = 0
    # 计算 node1 到 LCA 的距离
    n = node1
    while n != lca:
        distance += 1
        n = n.parent
    # 计算 node2 到 LCA 的距离
    n = node2
    while n != lca:
        distance += 1
        n = n.parent
    return distance


def dialect_distance(l1, l2, jsondata):
    """
    计算基于语言树的两种方言之间的距离。
    
    Args:
        l1 (str): 第一种方言的名称。
        l2 (str): 第二种方言的名称。
        jsondata (Dict): 包含语言树结构的JSON数据。
        
    Returns:
        int: 两种方言之间的距离。
        
    Raises:
        KeyError: 如果指定的方言名称在语言树中不存在。
    """ 
    # 加载JSON数据
    data = jsondata
    root = build_tree(data)

    # 创建名称到节点的映射
    nodes_dict = {}
    def map_nodes(node):
        nodes_dict[node.name] = node
        for child in node.children:
            map_nodes(child)
    map_nodes(root)
    
    # 检查方言是否存在
    if l1 not in nodes_dict:
        raise KeyError(f"方言 '{l1}' 在语言树中不存在")
    if l2 not in nodes_dict:
        raise KeyError(f"方言 '{l2}' 在语言树中不存在")
    
    # 计算两个方言的距离
    dialect_a = nodes_dict[l1]
    dialect_b = nodes_dict[l2]
    distance = calculate_distance(dialect_a, dialect_b)
    return distance

def linguistic_matrix(excel_path, json_path) -> np.ndarray:
    '''
    读取省份的代表性语言，计算各自的语言远近距离，输出矩阵
    '''
    data = pd.read_csv(excel_path) # excel数据分布
    with open(json_path, encoding='utf-8') as f:
        linguistic_tree = json.load(f) # 语言谱系树

    matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                matrix[i, j] = dialect_distance(data.iloc[i, 0], data.iloc[j, 0], linguistic_tree)
            else:
                matrix[i, j] = 0
    
    # 输出矩阵
    return matrix

def save_to_csv(matrix, path):
    df = pd.DataFrame(matrix)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    with open("data/linguistic.json", encoding='utf-8') as f:
        linguistic_tree = json.load(f)
    print(dialect_distance('吴语','湘语',linguistic_tree))
    