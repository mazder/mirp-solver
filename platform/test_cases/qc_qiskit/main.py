from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
import matplotlib.pyplot as plt  # <-- Make sure this is imported

# Create a new circuit with two qubits
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Generate and save the circuit diagram
fig = qc.draw("mpl")  # Save the figure returned by draw
fig.savefig("circuit.png")  # Save it to file
