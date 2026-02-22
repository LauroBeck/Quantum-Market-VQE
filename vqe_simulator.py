# ============================================================
# Probabilistic Nasdaq/Bloomberg Equity Flip VQE Simulator
# Python 3.13 + Qiskit 2.3.0 + qiskit-algorithms 0.4.0
# Each qubit = one stock
# Output = confidence % for Buy / Hold / Sell
# Using SparsePauliOp for correlations and VQE optimization
# ============================================================

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# -----------------------------
# Stocks (qubits)
# -----------------------------
equities = ["AAPL", "MSFT", "GOOG", "AMZN"]
n_qubits = len(equities)

# -----------------------------
# Market parameters (example)
# -----------------------------
# t_matrix = correlation-driven transition probability between stocks
t_matrix = np.array([
    [0.0, 0.3, 0.2, 0.1],
    [0.3, 0.0, 0.25, 0.15],
    [0.2, 0.25, 0.0, 0.2],
    [0.1, 0.15, 0.2, 0.0]
])

# U_values = individual stock volatility/risk
U_values = [0.5, 0.6, 0.4, 0.7]

# -----------------------------
# Build Hamiltonian as market operator
# -----------------------------
ham = SparsePauliOp.from_list([("I"*n_qubits, 0.0)])  # initialize zero operator

for i in range(n_qubits):
    # On-site volatility (Z_i)
    term = "I"*i + "Z" + "I"*(n_qubits-i-1)
    ham += U_values[i] * SparsePauliOp.from_list([(term, 1.0)])
    
    # Off-diagonal correlations (X_i X_j + Y_i Y_j)
    for j in range(i+1, n_qubits):
        xx_term = ["I"]*n_qubits
        yy_term = ["I"]*n_qubits
        xx_term[i], xx_term[j] = 'X','X'
        yy_term[i], yy_term[j] = 'Y','Y'
        xx_term_str = ''.join(xx_term)
        yy_term_str = ''.join(yy_term)
        ham += t_matrix[i,j] * SparsePauliOp.from_list([(xx_term_str, -0.5), (yy_term_str, -0.5)])

print("\nHamiltonian (market operator):")
print(ham)

# -----------------------------
# Exact diagonalization for reference
# -----------------------------
exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(ham)
print(f"\nExact ground state energy: {exact_result.eigenvalue.real:.6f}")

# -----------------------------
# VQE setup
# -----------------------------
ansatz = EfficientSU2(num_qubits=n_qubits, reps=2)
optimizer = COBYLA(maxiter=200)

use_ibmq = False  # True = IBM Quantum Runtime
if use_ibmq:
    service = QiskitRuntimeService(channel="ibm_quantum")
    sampler = Sampler(session=service)
    vqe_backend = sampler
else:
    vqe_backend = AerSimulator()

vqe = VQE(vqe_backend, ansatz=ansatz, optimizer=optimizer)
vqe_result = vqe.compute_minimum_eigenvalue(ham)
print(f"VQE ground state energy: {vqe_result.eigenvalue.real:.6f}")

# -----------------------------
# Probabilistic Buy/Hold/Sell mapping
# -----------------------------
ground_state = vqe_result.eigenstate  # statevector amplitudes
measurements = []

for i in range(n_qubits):
    p0 = 0.0  # probability qubit = 0
    p1 = 0.0  # probability qubit = 1
    for idx, amp in enumerate(ground_state):
        prob = abs(amp)**2
        if ((idx >> i) & 1):
            p1 += prob
        else:
            p0 += prob
    # Map 0 -> Hold, 1 -> Buy, simple Sell probability = residual (example)
    # Sell probability could also be derived from historical thresholds
    p_hold = p0
    p_buy = p1
    p_sell = max(0.0, 1.0 - (p0 + p1))  # small residual if probabilities < 1

    # Normalize to sum to 100%
    total = p_hold + p_buy + p_sell
    measurements.append({
        "Hold": p_hold/total*100,
        "Buy": p_buy/total*100,
        "Sell": p_sell/total*100
    })

# -----------------------------
# Print probabilistic recommendations
# -----------------------------
print("\n--- Probabilistic Nasdaq/Bloomberg Equity Flip Recommendations ---")
for eq, probs in zip(equities, measurements):
    print(f"{eq}: Buy {probs['Buy']:.1f}% | Hold {probs['Hold']:.1f}% | Sell {probs['Sell']:.1f}%")

# -----------------------------
# Energy error
# -----------------------------
energy_error = abs(vqe_result.eigenvalue.real - exact_result.eigenvalue.real)
print(f"\nEnergy error (VQE vs exact): {energy_error:.6f}")