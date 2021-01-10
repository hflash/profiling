from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
    converters,
    IBMQ)
from qiskit.transpiler import CouplingMap, Layout

provider = IBMQ.load_account()

#simulated_backend = provider.get_backend('ibmq_qasm_simulator')
simulated_backend = provider.get_backend('ibmqx5')
coupling_map_list = simulated_backend.configuration().coupling_map  # Get coupling map from backend
coupling_map = CouplingMap(coupling_map_list)
print(coupling_map_list)