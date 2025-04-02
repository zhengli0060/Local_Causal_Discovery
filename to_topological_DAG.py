"""
FILE: to_topological_DAG.py
CREATED: 2025-4-2
DESCRIPTION:
    This script processes a Bayesian Network represented by a .bif file.
    It performs the following tasks:
    1. Reads the Bayesian Network structure from a .bif file.
    2. Constructs a directed graph (DAG) using NetworkX.
    3. Performs topological sorting on the DAG.
    4. Reorders the adjacency matrix based on the topological order.
    5. Saves the reordered adjacency matrix to a CSV file.
    6. Reads the saved CSV file and prints its content.
USAGE:
    Update the `DAG_name` variable to specify the Bayesian Network name.
    Ensure the input .bif file exists in the `data_bif` directory.
    Run the script to generate the topologically sorted adjacency matrix and save it as a CSV file.
DEPENDENCIES:
    - NetworkX
    - Matplotlib
    - NumPy
    - Pandas
    - bnlearn
    - OS (Standard Library)
NOTES:
    - Ensure the .bif file is formatted correctly.
    - The output CSV file will be saved in the `data_bif` directory.
"""


import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import bnlearn as bn
# 读取存储贝叶斯网络的 .bif文件
DAG_name = 'asia'
bifile = f'data_bif/{DAG_name}.bif'
# 获取贝叶斯网络的邻接矩阵
model = bn.import_DAG(bifile,verbose=0)
adj_matrix = model['adjmat']   # DataFrame
node_list = model['model'].nodes()  # 节点列表  

# 创建有向图
DAG = nx.DiGraph()
DAG.add_nodes_from(node_list)  # 添加节点
# 添加边
for source in node_list:
    for target in node_list:
        if adj_matrix.loc[source, target] == True:
            DAG.add_edge(source, target)

# 进行拓扑排序
topological_order = list(nx.topological_sort(DAG))
print(f'tyep(topological_order):{type(topological_order)},topological_order:{topological_order}')
order_DAG = nx.relabel_nodes(DAG, {node: i for i, node in enumerate(topological_order)})

# 根据拓扑排序的顺序重新排列邻接矩阵
order_adjacency_matrix = adj_matrix.loc[topological_order, topological_order]
print(f"order_adjacency_matrix:{order_adjacency_matrix},type(order_adjacency_matrix):{type(order_adjacency_matrix)}")
save_path = os.path.join(r'data_bif', f'{DAG_name}_graph.csv')
order_adjacency_matrix.to_csv(save_path, index=True, header=True)

load_path = os.path.join(r'data_bif', f'{DAG_name}_graph.csv')
# 读取 CSV 文件 
df = pd.read_csv(load_path, index_col=0) # index_col=0 表示将第一列作为索引
# 打印 DataFrame 的内容
print(df)