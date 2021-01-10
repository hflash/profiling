import math
import os

from qiskit import QuantumCircuit, converters, QuantumRegister, transpile, IBMQ
from qiskit.test.mock import FakeMelbourne

# import sys
from qiskit.transpiler import Layout
from qiskit.transpiler.passes.routing import sabre_swap

slice_layers = 10


# sys.path.append('D:/pythonProject/profiling/circuittransform_modified')


if __name__ == '__main__':
    filename = './examples/qft_n14.qasm'
    circuit = QuantumCircuit.from_qasm_file(filename)
    qubit_num = len(circuit.qubits)
    qr = QuantumRegister(qubit_num)
    print("qubit number,", qubit_num)
    dag = converters.circuit_to_dag(circuit)

    # # delete the node of dag if the node is not cx gate
    # for node in dag.gate_nodes():
    #     if node.name != "cx":
    #         dag.remove_op_node(node)
    #         # print("remove")

    circuits_num = math.ceil(dag.depth() / slice_layers)
    circuits_slices = []
    # initial_mappings = []
    # for i in range(circuits_num):
    #     circuits_slices.append(QuantumCircuit(qubit_num))
    # print("append")

    # dag.draw()
    # plt.show()
    # for i in range(circuits_num):
    #     for j in range(slice_layers):
    #         for node in dag.front_layer():
    #             if node.name == "cx":
    #                 circuits_slices[i].cx(node.qargs[0].index, node.qargs[1].index)
    #             dag.remove_op_node(node)
    #
    # # write the slices to files
    # for i in range(circuits_num):
    #     filename = './circuit_slices_qft_14/slices_qft_' + str(i) + '.qasm'
    #     circuits_slices[i].qasm(formatted=True, filename=filename)

    # # test of slices to circuits
    # # circuits_slices.append(QuantumCircuit(qubit_num))
    # # circuits_slices[0] = QuantumCircuit.from_qasm_file('./circuit_slices_qft_15/slices_qft_0.qasm')
    # # print(circuits_slices[0])
    #
    # directly from slices to circuits
    for item in os.listdir('./circuit_slices_qft_14'):
        circuits_slices.append(QuantumCircuit.from_qasm_file('./circuit_slices_qft_14/' + item))

    # filedir = './circuit_slices_qft_14/'
    # # '''QASM input control'''
    # # QASM_files = os.listdir(filedir)
    # #
    # # print('QASM file is', QASM_files)
    # # '''output control'''
    # results = get_initial_mapping(filedir)
    # print(results)
    # for i in range(len(results)):
    #     filename = 'slices_qft_' + str(i)
    #     initial_mappings.append(results[filename]['initial map'])
    # print(initial_mappings)

    # initial_mappings of 14 qubit melborne
    initial_mappings = [[2, 12, 1, 13, 11, 3, 10, 9, 7, 0, 8, 4, 6, 5],
                        [9, 4, 3, 13, 12, 2, 11, 10, 5, 6, 7, 0, 8, 1],
                        [9, 6, 10, 3, 12, 1, 0, 13, 2, 11, 4, 5, 8, 7],
                        [7, 8, 6, 5, 9, 10, 3, 2, 13, 1, 0, 12, 11, 4],
                        [11, 0, 13, 12, 1, 2, 7, 3, 8, 6, 10, 4, 5, 9],
                        [3, 4, 11, 2, 5, 9, 12, 1, 10, 6, 8, 13, 0, 7]]

    #
    # initial_layouts = []
    # for item in initial_mappings:
    #     i = 0
    #     initial_layouts.append(Layout.from_intlist(item, qr))
    #
    # print(initial_layouts)

    backend1 = FakeMelbourne()
    for item in circuits_slices:
        i = 0
        # print(item)
        nc = transpile(item, initial_layout=initial_mappings[i], backend=backend1, routing_method="sabre")
        i += 1
        print(nc.depth())
        print(nc.count_ops())

    nc = transpile(circuit, initial_layout=[10, 4, 3, 2, 1, 13, 11, 12, 0, 5, 6, 9, 8, 7], backend=backend1,
                   layout_method="sabre")




    # print(nc.depth())
    # print(nc.count_ops())



