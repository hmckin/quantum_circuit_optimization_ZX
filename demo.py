"""Minimal adder demo: build in Qiskit, simplify with PyZX, report T-count.

Run:  python demo.py
"""

import io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import numpy as np
from qiskit import qasm2, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.quantum_info import Operator
from qiskit.qasm2 import dumps

import pyzx as zx

zx.settings.mode = "notebook"

BASIS = ["h", "cx", "t", "tdg", "s", "sdg", "z", "x"]
N = 3

OUT = "out"
ZX_ROW_IN = 0.45
ZX_DPI = 110
ZX_COLS_PER_PANEL = 48
PANEL_PAD = 0.9
ZX_COLS_PER_ROW = 64
ZX_STRIP_PAD = 1.4                # qubit of blank space between wrapped rows


# ===========================
# Core pipeline: PyZX logic
# ===========================
def build_and_transpile():
    qc = CDKMRippleCarryAdder(N, kind="full")
    qc = qc.decompose(reps=10)
    tq = transpile(qc, basis_gates=BASIS, optimization_level=0)
    return tq


def to_pyzx(tq):
    return zx.Circuit.from_qasm(dumps(tq))


def to_qiskit(opt, reference):
    """Turn the extracted PyZX circuit back into a Qiskit QuantumCircuit.
    """
    qc = qasm2.loads(opt.to_qasm(),
                     custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    overlap = Operator(reference).data @ Operator(qc).data.conj().T
    qc.global_phase = np.angle(np.diag(overlap)[0])
    return qc


def simplify(circ):
    g_before = circ.to_graph()
    g = g_before.copy()
    zx.full_reduce(g)
    opt = zx.extract_circuit(g.copy())
    return g_before, g, opt


# ===========================================================================
# Images: everything below renders diagrams to PNG
# ===========================================================================
def compact_columns(g):
    """Drop empty columns so the drawing has no wasted horizontal space.
    """
    out = g.copy()
    occupied = sorted({out.row(v) for v in out.vertices()})
    remap = {col: i for i, col in enumerate(occupied)}
    for v in out.vertices():
        out.set_row(v, remap[out.row(v)])
    return out


def zx_panel_pngs(g, stem, cols_per_panel=ZX_COLS_PER_PANEL):
    """Draw `g` as one PNG per block of columns; return the paths.
    """
    rows = [g.row(v) for v in g.vertices()]
    qubits = [g.qubit(v) for v in g.vertices()]
    lo, hi = min(rows), max(rows)
    minq, maxq = min(qubits), max(qubits)

    blocks = list(range(lo, hi + 1, cols_per_panel or (hi - lo + 1)))
    paths = []
    for idx, start in enumerate(blocks):
        end = min(start + cols_per_panel - 1, hi)
        if not any(start <= r <= end for r in rows):
            continue

        fig = zx.draw_matplotlib(g, rows=(start, end), figsize=(4, 2))
        ax = fig.axes[0]
        x0, x1 = start - PANEL_PAD, end + 1.0 + PANEL_PAD
        y0, y1 = -maxq - 0.8, -minq + 0.8
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        fig.set_size_inches(ZX_ROW_IN * (x1 - x0), ZX_ROW_IN * (y1 - y0))
        if len(blocks) > 1:
            ax.set_title(f"columns {start}-{end}", fontsize=9)

        p = f"{OUT}/{stem}_{idx}.png"
        fig.savefig(p, dpi=ZX_DPI)
        paths.append(p)
        plt.close(fig)
    return paths


def wrapped_zx_png(g, out_path, cols_per_row=ZX_COLS_PER_ROW):
    """Render `g` as one PNG that wraps onto further rows, Qiskit-style.
    """
    qubits = [g.qubit(v) for v in g.vertices()]
    qmin, qmax = min(qubits), max(qubits)
    n_qubit_rows = qmax - qmin + 1
    stride = n_qubit_rows + ZX_STRIP_PAD

    lo = min(g.row(v) for v in g.vertices())
    hi = max(g.row(v) for v in g.vertices())

    # One window reused by every strip: identical span -> identical scale.
    x0, x1 = -PANEL_PAD, cols_per_row + PANEL_PAD
    y0, y1 = -(n_qubit_rows - 1) - 0.8, 0.8
    w_in, h_in = ZX_ROW_IN * (x1 - x0), ZX_ROW_IN * (y1 - y0)

    strips = []
    for k, start in enumerate(range(lo, hi + 1, cols_per_row)):
        end = min(start + cols_per_row - 1, hi)
        strip = g.copy()
        outside = [v for v in strip.vertices()
                   if not start <= strip.row(v) <= end]
        if outside:
            strip.remove_vertices(outside)
        if not strip.num_vertices():
            continue

        for v in strip.vertices():
            strip.set_row(v, strip.row(v) - start)
            strip.set_qubit(v, strip.qubit(v) - qmin + k * stride)

        fig = zx.draw_matplotlib(strip, figsize=(4, 2))
        ax = fig.axes[0]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0 - k * stride, y1 - k * stride)
        ax.set_aspect("equal", adjustable="box")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.set_size_inches(w_in, h_in)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=ZX_DPI)
        plt.close(fig)
        strips.append(Image.open(buf).convert("RGB"))

    if not strips:
        raise ValueError("nothing to draw")

    canvas = Image.new("RGB",
                       (max(s.width for s in strips),
                        sum(s.height for s in strips)), "white")
    y = 0
    for s in strips:
        canvas.paste(s, (0, y))
        y += s.height
    canvas.save(out_path)
    return out_path, len(strips)


def save_drawings(tq, g_before, g, opt_qc):
    """Render the diagrams to PNG.
    """
    Path(OUT).mkdir(exist_ok=True)

    f1 = tq.draw("mpl", fold=100, idle_wires=False)
    f1.suptitle(f"Qiskit CDKMRippleCarryAdder({N}) - transpiled, unoptimised")
    p1 = f"{OUT}/1_qiskit_circuit.png"
    f1.savefig(p1, dpi=100, bbox_inches="tight")
    plt.close(f1)

    f4 = opt_qc.draw("mpl", fold=60, idle_wires=False)
    f4.suptitle(f"Optimised adder, extracted from ZX and rebuilt in Qiskit "
                f"(T-count {opt_qc.count_ops().get('t', 0) + opt_qc.count_ops().get('tdg', 0)}"
                f" once re-transpiled to Clifford+T)")
    p4 = f"{OUT}/4_qiskit_optimised.png"
    f4.savefig(p4, dpi=100, bbox_inches="tight")
    plt.close(f4)

    p2 = zx_panel_pngs(g_before, "2_zx_before")
    p2_all, strips = wrapped_zx_png(g_before, f"{OUT}/2_zx_before_all.png")
    p3 = zx_panel_pngs(compact_columns(g), "3_zx_after")
    return p1, p2, p2_all, strips, p3, p4


def display(qiskit_png, before_pngs, after_pngs, opt_png, t_before, t_after):
    """Compose the diagrams into one overview and show it."""
    n_before, n_after = len(before_pngs), len(after_pngs)
    entries = [(qiskit_png, f"1. Qiskit circuit, Clifford+T (T-count {t_before})"),
               (opt_png, f"2. Rebuilt in Qiskit from the optimised ZX "
                         f"(T-count {t_after})")]
    entries += [(p, f"3. ZX before full_reduce  {i + 1}/{n_before}")
                for i, p in enumerate(before_pngs)]
    entries += [(p, f"4. ZX after full_reduce  {i + 1}/{n_after}")
                for i, p in enumerate(after_pngs)]

    p, _ = compose(entries, f"{OUT}/0_all.png", full_width=(0, 1))
    print(f"\nSaved {p}")
    plt.show()
    return p


def compose(entries, out_path, ncols=3, target_w=3000, full_width=(0,),
            gap_frac=0.012):
    """Lay images out on a grid at their native aspect ratio and save.
    """
    imgs = [plt.imread(p) for p, _ in entries]
    nat = [(im.shape[1], im.shape[0]) for im in imgs]
    rest = [i for i in range(len(imgs)) if i not in full_width]
    grid = [[i] for i in full_width]
    grid += [rest[r:r + ncols] for r in range(0, len(rest), ncols)]

    row_w = [sum(nat[j][0] for j in row) for row in grid]
    row_h = [max(nat[j][1] for j in row) for row in grid]
    s = 1.0 if target_w is None else target_w / max(row_w)
    total_w = max(row_w) * s

    # Only reserve title/headroom room if something actually has a title.
    has_titles = any(t for _, t in entries)
    title_h = 0.030 * total_w if has_titles else 0.0
    head = 0.026 * total_w if has_titles else 0.0
    gap = gap_frac * total_w
    total_h = head + sum(h * s for h in row_h) + len(grid) * title_h \
        + (len(grid) - 1) * gap

    fig = plt.figure(figsize=(total_w / 100, total_h / 100))
    y = 1.0 - head / total_h
    for row, rh in zip(grid, row_h):
        x = 0.0
        for j in row:
            w, h = nat[j]
            bottom = y - (h * s) / total_h
            ax = fig.add_axes([x, bottom, (w * s) / total_w, (y - bottom)])
            ax.imshow(imgs[j], aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if entries[j][1]:
                ax.set_title(entries[j][1], fontsize=9, pad=3)
            x += (w * s) / total_w
        y = bottom - title_h / total_h - gap / total_h
    fig.savefig(out_path, dpi=100)
    return out_path, fig


def main():
    tq = build_and_transpile()
    print(f"Qiskit circuit: {tq.num_qubits} qubits, {tq.size()} gates")
    print(f"  op counts: {dict(sorted(tq.count_ops().items()))}")

    circ = to_pyzx(tq)
    print(f"\nPyZX circuit:\n{circ.stats()}")

    g_before, g, opt = simplify(circ)

    t_before = zx.tcount(g_before)
    t_after = zx.tcount(opt)
    print(f"\nAfter full_reduce + extract_circuit:\n{opt.stats()}")

    same = circ.verify_equality(opt)
    print(f"\nverify_equality(circ, opt): {same}")
    assert same, "PyZX rewrite is not equivalent to the original circuit"

    opt_qc = to_qiskit(opt, tq)
    print(f"\nRebuilt in Qiskit from the optimised ZX "
          f"({opt_qc.size()} gates):")
    print(f"  {dict(sorted(opt_qc.count_ops().items()))}"
          f"  global_phase {opt_qc.global_phase:.4f}")
    exact = Operator(tq) == Operator(opt_qc)
    print(f"  unitary identical to the original circuit: {exact}")
    assert exact, "ZX round trip does not reproduce the original circuit"

    opt_ct = transpile(opt_qc, basis_gates=BASIS, optimization_level=0)
    t_qiskit = opt_ct.count_ops().get("t", 0) + opt_ct.count_ops().get("tdg", 0)
    print(f"  re-transpiled to Clifford+T: T-count {t_qiskit} "
          f"(PyZX reported {t_after})")

    # --- image writing: ------------------------
    qiskit_png, before_pngs, before_all, strips, after_pngs, opt_png = save_drawings(
        tq, g_before, g, opt_ct)
    print(f"\nDiagrams written to {OUT}/:")
    print(f"  {before_all}  - ZX before, one connected diagram wrapped over "
          f"{strips} row(s)")
    print(f"  {after_pngs[0]}{'' if len(after_pngs) == 1 else ' ...'}  - ZX after "
          f"(columns compacted to {len(after_pngs)} block(s))")
    print(f"  {qiskit_png}  - Qiskit circuit")
    print(f"  {opt_png}  - optimised circuit, rebuilt from ZX")
    print(f"  {before_pngs[0]} ... {before_pngs[-1]}  - the individual ZX before "
          f"blocks, if you want to zoom a region")

    print(f"\nT-count before: {t_before}")
    print(f"T-count after:  {t_after}")
    print(f"Reduction:      {t_before - t_after} "
          f"({100 * (t_before - t_after) / t_before:.1f}%)")

    display(qiskit_png, before_pngs, after_pngs, opt_png, t_before, t_after)
    # --- end image writing ---------------------------------------------------


if __name__ == "__main__":
    main()
