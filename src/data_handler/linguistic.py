import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class TreeNode:
    """树节点类，用于表示语言树中的一个节点。"""
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []
        self.depth = 0

def build_tree(data: Dict[str, Any], parent: TreeNode = None) -> TreeNode:
    """递归地从字典数据构建语言树。"""
    node = TreeNode(data["name"])
    node.parent = parent
    node.depth = parent.depth + 1 if parent else 0
    if "children" in data:
        for child_data in data["children"]:
            child_node = build_tree(child_data, parent=node)
            node.children.append(child_node)
    return node

def _move_to_depth(node: TreeNode, target_depth: int) -> TreeNode:
    """将节点移动到指定的深度。"""
    while node.depth > target_depth:
        node = node.parent
    return node

def find_lca(node1: TreeNode, node2: TreeNode) -> TreeNode:
    """查找两个节点的最近公共祖先 (LCA)。"""
    if node1.depth > node2.depth:
        node1 = _move_to_depth(node1, node2.depth)
    else:
        node2 = _move_to_depth(node2, node1.depth)
    
    while node1 != node2:
        node1 = node1.parent
        node2 = node2.parent
    return node1

def calculate_distance(node1: TreeNode, node2: TreeNode) -> int:
    """计算两个节点之间的距离。"""
    lca = find_lca(node1, node2)
    distance = 0
    n = node1
    while n != lca:
        distance += 1
        n = n.parent
    n = node2
    while n != lca:
        distance += 1
        n = n.parent
    return distance

def dialect_distance(l1: str, l2: str, jsondata: Dict[str, Any]) -> int:
    """计算基于语言树的两种方言之间的距离。"""
    root = build_tree(jsondata)
    nodes_dict = {}
    def map_nodes(node: TreeNode):
        nodes_dict[node.name] = node
        for child in node.children:
            map_nodes(child)
    map_nodes(root)
    
    if l1 not in nodes_dict:
        raise KeyError(f"方言 '{l1}' 在语言树中不存在")
    if l2 not in nodes_dict:
        raise KeyError(f"方言 '{l2}' 在语言树中不存在")
    
    dialect_a = nodes_dict[l1]
    dialect_b = nodes_dict[l2]
    return calculate_distance(dialect_a, dialect_b)

def linguistic_matrix(
    prov_lang_path: str, 
    json_tree_path: str, 
    ordered_provinces: List[str]
) -> pd.DataFrame:
    """
    根据给定的省份排序，计算语言距离矩阵。

    Args:
        prov_lang_path (str): 省份-语言对应关系的CSV文件路径。
        json_tree_path (str): 语言谱系树的JSON文件路径。
        ordered_provinces (List[str]): 需要输出的、排序好的省份列表。

    Returns:
        pd.DataFrame: 排序并标记好的省份间语言距离矩阵。
    """
    lang_data = pd.read_csv(prov_lang_path, header=None, index_col=0, names=['dialect'])
    lang_dict = lang_data['dialect'].to_dict()

    with open(json_tree_path, encoding='utf-8') as f:
        linguistic_tree = json.load(f)

    n = len(ordered_provinces)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0
                continue

            prov1 = ordered_provinces[i]
            prov2 = ordered_provinces[j]
            
            dialect1 = lang_dict.get(prov1)
            dialect2 = lang_dict.get(prov2)

            if dialect1 is None or dialect2 is None:
                raise ValueError(f"在语言数据中找不到省份 {prov1} 或 {prov2}")

            if dialect1 == '其他' or dialect2 == '其他':
                matrix[i, j] = 10  # 为'其他'设置最大距离
            else:
                matrix[i, j] = dialect_distance(dialect1, dialect2, linguistic_tree)
    
    df = pd.DataFrame(matrix, index=ordered_provinces, columns=ordered_provinces)
    return df

def save_to_csv(df: pd.DataFrame, path: str):
    """将DataFrame保存为CSV文件，包含索引。"""
    df.to_csv(path, index=True)

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    prov_lang_path = os.path.join(project_root, "data/processed/prov_language.csv")
    json_tree_path = os.path.join(project_root, "data/processed/linguistic_tree.json")
    ranked_prov_path = os.path.join(project_root, "data/processed/prov_name_ranked.json")
    output_path = os.path.join(project_root, "data/processed/linguistic_distance_matrix.csv")
    
    with open(ranked_prov_path, 'r', encoding='utf-8') as f:
        ordered_provinces = json.load(f)
        
    distance_df = linguistic_matrix(
        prov_lang_path=prov_lang_path, 
        json_tree_path=json_tree_path,
        ordered_provinces=ordered_provinces
    )
    
    save_to_csv(distance_df, output_path)
    print(f"Linguistic distance matrix saved to {output_path}")