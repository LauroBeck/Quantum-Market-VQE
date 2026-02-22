# ============================================================
# Probabilistic Nasdaq / Bloomberg-Style VQE Simulator
# Qiskit 2.x Compatible – Correct Result Handling
# Each qubit = one equity
# Output = Buy / Hold / Sell confidence
# ============================================================

import numpy as np

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA


# -------------------------------------------------
# 1. Define Market Hamiltonian
# -------------------------------------------------

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


# -------------------------------------------------
# 2. Exact Classical Reference
# -------------------------------------------------

exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(ham)

exact_energy = exact_result.eigenvalue.real

print("\nExact ground state energy: {:.6f}".format(exact_energy))


# -------------------------------------------------
# 3. Ansatz (Modern Builder)
# -------------------------------------------------

ansatz = efficient_su2(
    num_qubits=n_qubits,
    reps=2,
    entanglement="full"
)

print("\nNumber of ansatz parameters:", ansatz.num_parameters)


# -------------------------------------------------
# 4. VQE Setup (Primitive-Based)
# -------------------------------------------------

estimator = StatevectorEstimator()
optimizer = COBYLA(maxiter=500)

# Optional deterministic start (improves stability)
initial_point = np.zeros(ansatz.num_parameters)

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=initial_point
)


# -------------------------------------------------
# 5. Run VQE
# -------------------------------------------------

vqe_result = vqe.compute_minimum_eigenvalue(ham)

vqe_energy = vqe_result.eigenvalue.real

print("\nVQE ground state energy: {:.6f}".format(vqe_energy))
print("Energy difference (VQE - Exact): {:.6f}".format(
    vqe_energy - exact_energy
))


# -------------------------------------------------
# 6. Reconstruct Statevector (Qiskit 2.x Correct)
# -------------------------------------------------

optimal_params = vqe_result.optimal_point

bound_circuit = ansatz.assign_parameters(optimal_params)

state = Statevector.from_instruction(bound_circuit)
probs = state.probabilities()


# -------------------------------------------------
# 7. Extract Probabilistic Signals
# -------------------------------------------------

print("\n--- Probabilistic Equity Signals ---")

equities = ["AAPL", "MSFT", "GOOG", "AMZN"]

signals = []

for qubit in range(n_qubits):

    p0 = 0.0   # |0> → Hold
    p1 = 0.0   # |1> → Buy

    for idx, p in enumerate(probs):
        if ((idx >> qubit) & 1):
            p1 += p
        else:
            p0 += p

    # Finance-style Sell = low conviction model
    sell = (1.0 - max(p0, p1)) * 100

    signals.append({
        "Buy": 100 * p1,
        "Hold": 100 * p0,
        "Sell": sell
    })


# -------------------------------------------------
# 8. Display Results
# -------------------------------------------------

for eq, s in zip(equities, signals):
    print(f"{eq}: "
          f"Buy {s['Buy']:.2f}% | "
          f"Hold {s['Hold']:.2f}% | "
          f"Sell {s['Sell']:.2f}%")