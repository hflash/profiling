import os
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

a = [[1,2,3], [4,5,6],[7,8,9]]

b = [[1,1,1], [2,2,2],[3,4,5]]

a_name = "aname"
b_name = "bname"

output = [{"name": a_name, "matrix": a}, {"name": b_name, "matrix": b}]

s = json.dumps(output)

# f = open("test.json", "w")

# f.write(s)
# f.close()

ff = open("test.json", "r")
ss = ff.read()

content = json.loads(ss)
for item in content:
    n = item["name"]
    arr = item["matrix"]
    arr_np = np.array(arr)

    print(n)
    print(arr_np)

