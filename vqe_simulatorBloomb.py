# -------------------------------------------------
# Run VQE
# -------------------------------------------------

vqe_result = vqe.compute_minimum_eigenvalue(ham)

print("\nVQE ground state energy: {:.6f}".format(
    vqe_result.eigenvalue.real
))

print("\nEnergy difference (VQE - Exact): {:.6f}".format(
    vqe_result.eigenvalue.real - exact_result.eigenvalue.real
))


# -------------------------------------------------
# Extract Probabilities
# -------------------------------------------------

state = vqe_result.eigenstate
probs = state.probabilities()

print("\n--- Probabilistic Equity Signals ---")

equities = ["AAPL", "MSFT", "GOOG", "AMZN"]

signals = []

for qubit in range(n_qubits):

    p0 = 0.0
    p1 = 0.0

    for idx, p in enumerate(probs):
        if ((idx >> qubit) & 1):
            p1 += p
        else:
            p0 += p

    signals.append({
        "Buy": 100 * p1,
        "Hold": 100 * p0,
        "Sell": 0.0
    })

for eq, s in zip(equities, signals):
    print(f"{eq}: Buy {s['Buy']:.2f}% | Hold {s['Hold']:.2f}% | Sell {s['Sell']:.2f}%")