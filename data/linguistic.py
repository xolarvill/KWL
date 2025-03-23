import json
import pandas as pd

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
    Calculate the distance between two dialects based on a linguistic tree.
    Args:
        l1 (str): The name of the first dialect.
        l2 (str): The name of the second dialect.
    Returns:
        None: This function prints the distance between the two dialects.
    The function performs the following steps:
    1. Loads JSON data from a specified file path.
    2. Builds a linguistic tree from the JSON data.
    3. Creates a mapping from dialect names to their corresponding nodes in the tree.
    4. Calculates the distance between the two specified dialects.
    5. Prints the calculated distance.
    Note:
        The JSON file should be encoded in UTF-8 and should contain the necessary data to build the linguistic tree.
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
    
    # 计算两个方言的距离
    dialect_a = nodes_dict[l1]
    dialect_b = nodes_dict[l2]
    distance = calculate_distance(dialect_a, dialect_b)
    print(f"亲疏距离: {distance}")  # 输出示例：亲疏距离: 3


def linmatrix(excel_path, json_path,):
    '''
    读取省份的语言人口分布数据，计算各自的语言远近距离，输出矩阵
    '''
    #  ========================= 读取数据 ========================= 
    data = pd.read_excel(excel_path) # excel数据分布
    with open(json_path, encoding='utf-8') as f:
        linguistic_tree = json.load(f) # 语言谱系树
    # ========================= 计算数据 ========================= 
    
    matrix = []
    # ========================= 输出矩阵 ========================= 

    return matrix



if __name__ == '__main__':
    with open("D:\\STUDY\\CFPS\\merged\\KWL\\data\\linguistic.json", encoding='utf-8') as f:
        linguistic_tree = json.load(f)
    print(dialect_distance('吴语','中原官话',linguistic_tree))