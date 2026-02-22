import numpy as np
from itertools import product
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# Market Hamiltonian (same structure you already use)
# ============================================================

n_qubits = 4

paulis = [
    "IIII",
    "ZIII", "IZII", "IIZI", "IIIZ",
    "XXII", "YYII",
    "XIXI", "YIYI",
    "XIIX", "YIIY",
    "IXXI", "IYYI",
    "IXIX", "IYIY",
    "IIXX", "IIYY"
]

coeffs = [
    0.0,
    0.5, 0.6, 0.4, 0.7,
    -0.15, -0.15,
    -0.10, -0.10,
    -0.05, -0.05,
    -0.125, -0.125,
    -0.075, -0.075,
    -0.10, -0.10
]

ham = SparsePauliOp(paulis, coeffs)

print("\nHamiltonian (market operator):")
print(ham)

# ============================================================
# Energy of computational basis state
# ============================================================

def bitstring_energy(bitstring):
    """Compute <s|H|s> for computational basis state."""
    energy = 0.0

    for pauli, coeff in zip(paulis, coeffs):

        if "X" in pauli or "Y" in pauli:
            # Off-diagonal → expectation = 0 for basis states
            continue

        term = coeff

        for i, p in enumerate(pauli):
            if p == "Z":
                term *= (1 if bitstring[i] == "0" else -1)

        energy += term

    return energy


# ============================================================
# Gibbs Distribution
# ============================================================

def gibbs_distribution(T):
    states = [''.join(bits) for bits in product('01', repeat=n_qubits)]

    energies = np.array([bitstring_energy(s) for s in states])

    weights = np.exp(-energies / T)
    probs = weights / np.sum(weights)

    return states, energies, probs


# ============================================================
# Signal Mapping
# ============================================================

def signal_from_prob(p0):
    """
    Convert probability of qubit=0 into Buy/Hold/Sell style signal.
    """
    buy = p0
    sell = 1 - p0

    # Simple neutral band
    hold = 1 - abs(buy - sell)

    # Normalize
    norm = buy + sell + hold
    return buy / norm, hold / norm, sell / norm


# ============================================================
# Run Simulation
# ============================================================

T = 0.7  # Try changing this (0.1 → trending, 2.0 → chaotic)

states, energies, probs = gibbs_distribution(T)

print(f"\nTemperature (market volatility): {T}")

# Marginal qubit probabilities
p_qubits = np.zeros(n_qubits)

for s, p in zip(states, probs):
    for i, bit in enumerate(s):
        if bit == "0":
            p_qubits[i] += p

# ============================================================
# Display Signals
# ============================================================

labels = ["AAPL", "MSFT", "GOOG", "AMZN"]

print("\n--- Finite-Temperature Probabilistic Signals ---")

for label, p0 in zip(labels, p_qubits):

    buy, hold, sell = signal_from_prob(p0)

    print(f"{label}: "
          f"Buy {buy*100:6.2f}% | "
          f"Hold {hold*100:6.2f}% | "
          f"Sell {sell*100:6.2f}%")

# ============================================================
# Shock Event Example
# ============================================================

shock_strength = 0.3
T_shock = T + shock_strength

states_s, energies_s, probs_s = gibbs_distribution(T_shock)

p_qubits_s = np.zeros(n_qubits)

for s, p in zip(states_s, probs_s):
    for i, bit in enumerate(s):
        if bit == "0":
            p_qubits_s[i] += p

print("\n--- After Volatility Shock ---")

for label, p0 in zip(labels, p_qubits_s):

    buy, hold, sell = signal_from_prob(p0)

    print(f"{label}: "
          f"Buy {buy*100:6.2f}% | "
          f"Hold {hold*100:6.2f}% | "
          f"Sell {sell*100:6.2f}%")