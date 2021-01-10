import copy
import time

from qiskit import dagcircuit, transpile, QuantumRegister
from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
    converters,
    IBMQ)
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGNode
from qiskit.visualization import plot_histogram, plot_circuit_layout
import os
import random
from qiskit.transpiler import CouplingMap, Layout
import matplotlib.pyplot as plt


def get_extended_layer(dag, front_layer, layers, node_state_d):
    extended_layer = []
    f_layer = copy.deepcopy(front_layer)
    for i in range(layers):
        for node in f_layer:
            node_state_d[node._node_id] = 2
            # layer_identifier[node._node_id] = -1
            suc = dag.successors(node)
            for item in suc:
                if item.name == "cx":
                    pre = dag.predecessors(item)
                    for x in pre:
                        if x.name != "cx":
                            continue
                        if node_state_d[x._node_id] == 2:
                            node_state_d[item._node_id] = 3
                        else:
                            node_state_d[item._node_id] = 0
                            break
                    if node_state_d[item._node_id] == 3:
                        extended_layer.append(item)
        for node in extended_layer:
            node_state_d[node._node_id] = 2
            f_layer.append(node)
    temp = set(extended_layer)
    extended_layer = list(temp)
    print(f_layer)
    print(front_layer)
    return extended_layer


def transend_barrier(f_layer, dag):
    suc = []
    if f_layer[0].name == "barrier":
        suc = dag.successors(f_layer[0])
        f_layer.remove(f_layer[0])
    suc_dict = {}
    for node in suc:
        # suc_dict[node._node_id] = -1
        for pre in dag.predecessors(node):
            if pre != f_layer[0]:
                suc_dict[node._node_id] = 0
                break
            else:
                suc_dict[node._node_id] = 1
    for node in suc:
        if suc_dict[node._node_id] == 1:
            f_layer.append(node)
    return f_layer


if __name__ == '__main__':
    filename = '../examples/test.qasm'
    # dag = dagcircuit.DAGCircuit()

    qr = QuantumRegister(5)

    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.measure_all()
    circuit.draw()

    node_state_dic = {}
    layer_identifier = {}
    dag = converters.circuit_to_dag(circuit)
    exetended_layer_nodes = []

    for node in dag.gate_nodes():
        if node.name != "cx":
            dag.remove_op_node(node)
    front_layer = dag.front_layer()
    for node in dag.gate_nodes():
        node_state_dic[node._node_id] = 0
        # layer_identifier[node._node_id] = -1
        """layer_identifier
        -1 not identified yet
        0 front layer
        1 extended layer 1
        2 extended layer 2
        """

    exetend_layer_nodes = get_extended_layer(dag, front_layer, 3, node_state_dic)
    for node in front_layer:
        print(node.qargs[0], node.qargs[1])
    for node in exetend_layer_nodes:
        print(node.qargs[0], node.qargs[1])

    # for item in dag.gate_nodes():
    #     # print(len(item.qargs))
    #     if item.name != 'cx':
    #         dag.remove_op_node(item)
    # dag_copy = copy.deepcopy(dag)
    # # for item in dag_copy.gate_nodes():
    # #     # print(len(item.qargs))
    # #     if item.name != 'cx':
    # #         dag_copy.remove_op_node(item)
    #
    # for item in dag.gate_nodes():
    #     dag.remove_op_node(item)
    # # for _ in range(len(dag_copy.gate_nodes())):
    # #     item=dag_copy.gate_nodes()[0]
    # #     dag_copy.remove_op_node(item)
    # for item in dag_copy.gate_nodes():
    #     dag_copy.remove_op_node(item)
    #
    # nodes = dag.topological_nodes()
    # for node in nodes:
    #     print(node.name)
    #     print(node.qargs)
    # f_layer = copy.deepcopy(front_layer)
    #
    # node_state_dict = {}
    # for node in dag.gate_nodes():
    #     node_state_dict[node._node_id] = 0
    # for node in f_layer:
    #     node_state_dict[node._node_id] = 2
    # for node in f_layer:
    #     f_layer.remove(node)
    #     successor_gate_list = dag.quantum_successors(node)
    #     for node in successor_gate_list:
    #         # if len(node.qargs) == 2:
    #         if node.name == 'cx':
    #             # predecessor_gate_list = dag_circuit.quantum_predecessors(node)
    #             for pre in dag.predecessors(node):
    #                 if pre.name == 'cx':
    #                     if node_state_dict[pre._node_id] == 2:
    #                         node_state_dict[node._node_id] = 3
    #                     else:
    #                         node_state_dict[node._node_id] = 0
    #                         break
    #
    #             if node_state_dict[node._node_id] == 3:
    #                 f_layer.append(node)
    #                 node_state_dict[node._node_id] = 1
    # temp_set_frontlayer = set(front_layer)
    # front_layer = list(temp_set_frontlayer)
