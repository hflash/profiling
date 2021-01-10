# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:45:03 2020

https://gitlab.umiacs.umd.edu/amchilds/arct/-/commit/d74ac309e7a6f88324e7ba427951c9dca4346f8a
"""

from Childs.general import ApproximateTokenSwapper
import pandas as pd
import os
import networkx as nx
from typing import Mapping

num_q = 20
res_num_swaps = []
res_num_cnot = []

# AG
AG = nx.Graph()
vertex = list(range(20))
edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(1,6),(2,7),(3,8),(4,9),(1,7),(2,6),(3,9),(4,8),\
         (5,6),(6,7),(7,8),(8,9),(5,10),(6,11),(7,12),(8,13),(9,14),(5,11),(6,10),(7,13),(8,12),\
         (10,11),(11,12),(12,13),(13,14),(10,15),(11,16),(12,17),(13,18),(14,19),(11,17),(12,16),(13,19),(14,18),\
         (15,16),(16,17),(17,18),(18,19)]
AG.add_nodes_from(vertex)
AG.add_edges_from(edges)

# read qasm file names
path = "D:/anaconda3/envs/quantum/Lib/site-packages/circuittransform/inputs/QASM example/Tan_UCLA_Q20" 
#pre_name = "Tan_UCLA_Q20/"
QASM_files = os.listdir(path) #得到文件夹下的所有文件名称

#读取工作簿和工作簿中的工作表
solution_path = 'D:/Users/zxz58/Documents/Python Scripts/QUEKO-benchmark-master/meta/'
num_files = 0
for qasm_name in QASM_files:
    num_files += 1
    print('number of files is', num_files)
    solution_file_name =  qasm_name[0:-5] + '_solution'
    data_frame = pd.read_csv(solution_path+solution_file_name+'.csv',
                             header=None)
    log_2_phy = data_frame.values
    # create target mapping
    tokens = {}
    for q_log in range(num_q):
        q_phy = log_2_phy[q_log][0]
        tokens[q_log] = q_phy
    # peform approximate token swapping
    token_swapping = ApproximateTokenSwapper(AG)
    swaps = token_swapping.map(tokens, trials=5000)
    num_swaps = len(swaps)
    res_num_swaps.append(num_swaps)
    res_num_cnot.append(num_swaps*3)
    
