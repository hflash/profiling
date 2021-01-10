from qiskit import execute, QuantumCircuit, IBMQ, Aer

filename = "./qasm/qft_10.qasm"
circuit = QuantumCircuit.from_qasm_file(filename)
circuit.measure_all()
IBMQ.load_account()
backend = Aer.get_backend()
execute()
