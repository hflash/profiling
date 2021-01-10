from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def main():
    filename = './openqasm/adder.qasm'
    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.draw()
    plt.show()


if __name__ == '__main__':
    main()