"""Benchmark T-count reduction across several Qiskit circuits.

Reuses the PyZX pipeline from demo.py
"""

import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*GroverOperator.*")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*MCXGate.*")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*QuantumCircuit\.mcx.*")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    VBERippleCarryAdder,
    WeightedAdder,
)
from qiskit.circuit.library.grover_operator import GroverOperator
from qiskit.quantum_info import Operator

import pyzx as zx

from demo import BASIS, simplify, to_pyzx, to_qiskit


def as_circuit(obj):
    """Qiskit's *Gate classes are not circuits; wrap them so they decompose."""
    if isinstance(obj, QuantumCircuit):
        return obj
    qc = QuantumCircuit(obj.num_qubits)
    qc.append(obj, range(obj.num_qubits))
    return qc


def grover_oracle(n):
    """A small unitary oracle for GroverOperator to diffuse."""
    qc = QuantumCircuit(n, name="oracle")
    qc.h(range(n))
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(range(n))
    return qc

CIRCUITS = [
    ("CDKMRippleCarryAdder(3)", "adder", lambda: CDKMRippleCarryAdder(3, "full")),
    ("CDKMRippleCarryAdder(4)", "adder", lambda: CDKMRippleCarryAdder(4, "full")),
    ("VBERippleCarryAdder(3)", "adder", lambda: VBERippleCarryAdder(3)),
    ("WeightedAdder(3)", "adder", lambda: WeightedAdder(3)),
    ("GroverOperator(3)", "search", lambda: GroverOperator(grover_oracle(3))),
]

def measure(factory):
    """Run one circuit through the same pipeline demo.py uses."""
    decomposed = as_circuit(factory()).decompose(reps=10)
    non_unitary = {op for op in ("reset", "initialize", "measure")
                   if op in decomposed.count_ops()}
    if non_unitary:
        raise ValueError(f"circuit is not unitary: {sorted(non_unitary)}")

    tq = transpile(decomposed, basis_gates=BASIS, optimization_level=0)
    stray = set(tq.count_ops()) - set(BASIS)
    if stray:
        raise ValueError(f"not Clifford+T after transpile: {sorted(stray)}")

    circ = to_pyzx(tq)
    _, _, opt = simplify(circ)

    t_before, t_after = zx.tcount(circ), zx.tcount(opt)

    assert circ.verify_equality(opt), "pyzx verify_equality failed"
    assert Operator(tq) == Operator(to_qiskit(opt, tq)), "unitary mismatch"

    reduction = 100 * (t_before - t_after) / t_before if t_before else 0.0
    return {
        "name": None,           # filled in by the caller
        "qubits": tq.num_qubits,
        "gates": tq.size(),
        "depth": tq.depth(),
        "t_before": t_before,
        "t_after": t_after,
        "saved": t_before - t_after,
        "reduction": reduction,
    }


def format_rows(rows):
    header = f"{'Circuit':26s} {'Family':10s} {'Qubits':>6s} {'T before':>8s} " \
             f"{'T after':>8s} {'Saved':>6s} {'Reduction':>9s}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{str(r['name']):26s} {str(r['family']):10s} {r['qubits']:6d} "
            f"{r['t_before']:8d} {r['t_after']:8d} {r['saved']:6d} "
            f"{r['reduction']:8.1f}%"
        )
    return "\n".join(lines)


def main():
    rows = []
    for name, family, factory in CIRCUITS:
        started = time.perf_counter()
        try:
            row = measure(factory)
        except Exception as exc:                      # keep going on failure
            print(f"{name:26s} FAILED {type(exc).__name__}: {str(exc)[:60]}")
            continue
        row["name"], row["family"] = name, family
        rows.append(row)
        print(f"{name:26s} q={row['qubits']:2d}  T {row['t_before']:5d} -> "
              f"{row['t_after']:5d}  (-{row['saved']:4d}, "
              f"{row['reduction']:5.1f}%)  {time.perf_counter() - started:5.1f}s",
              flush=True)

    if not rows:
        raise SystemExit("no circuits measured")

    print()
    print(format_rows(rows))


if __name__ == "__main__":
    main()