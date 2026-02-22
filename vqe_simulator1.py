import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import efficient_su2
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver


# -----------------------------
# 1. Define Market Hamiltonian
# -----------------------------

paulis = [
    "IIII",
    "ZIII", "XXII", "YYII", "XIXI", "YIYI", "XIIX", "YIIY",
    "IZII", "IXXI", "IYYI", "IXIX", "IYIY",
    "IIZI", "IIXX", "IIYY",
    "IIIZ"
]

coeffs = [
    0.0,
    0.5, -0.15, -0.15, -0.1, -0.1, -0.05, -0.05,
    0.6, -0.125, -0.125, -0.075, -0.075,
    0.4, -0.1, -0.1,
    0.7
]

ham = SparsePauliOp(paulis, coeffs)

n_qubits = ham.num_qubits

print("\nHamiltonian (market operator):")
print(ham)


# ----------------------------------
# 2. Exact Classical Ground State
# ----------------------------------

exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(ham)

print("\nExact ground state energy: {:.6f}".format(
    exact_result.eigenvalue.real
))


# ----------------------------------
# 3. Build Ansatz (modern API)
# ----------------------------------

ansatz = efficient_su2(
    num_qubits=n_qubits,
    reps=2,
    entanglement="full"
)

print("\nNumber of ansatz parameters:", ansatz.num_parameters)


# ----------------------------------
# 4. Setup VQE
# ----------------------------------

estimator = Estimator()  # Aer-backed primitive
optimizer = COBYLA(maxiter=200)

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer
)


# ----------------------------------
# 5. Run VQE
# ----------------------------------

vqe_result = vqe.compute_minimum_eigenvalue(ham)

print("\nVQE ground state energy: {:.6f}".format(
    vqe_result.eigenvalue.real
))

print("\nEnergy difference (VQE - Exact): {:.6f}".format(
    vqe_result.eigenvalue.real - exact_result.eigenvalue.real
))