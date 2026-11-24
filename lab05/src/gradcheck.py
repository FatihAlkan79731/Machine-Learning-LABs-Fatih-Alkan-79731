"""Step 4: verify the engine against PyTorch autograd and central finite differences.

Main test (from the lab manual):  f = ((a*b + c)**2).relu().sum()
with a, b, c of shape (4, 5), float64, drawn from N(0, 1) with seed 42.

Extra tests:
  * reuse:  e = a*b; f = sum(e*e + e + relu(e - c))   -> every node is used more than once,
            so a wrong (overwriting) backward would fail here
  * all ops: f = sum(sigmoid(a) * log(b**2 + 1) + exp(-c) / (a**2 + 1) - c**3)

Pass criterion: max |engine - torch| < 1e-6 and max |engine - numerical| < 1e-6.

Outputs:
  results/gradcheck.csv          - one row per input element of the main test
  results/gradcheck_summary.csv  - max absolute differences for every test
  results/graph_example.dot      - graph of the main expression for one element (Graphviz)
  results/graph_stats.txt        - number of nodes / edges / ops of each test graph
"""
import math

import numpy as np
import pandas as pd

from common import RESULTS, SEED
from engine import Value, graph_stats, to_dot, vsum

try:
    import torch
except ImportError:  # the engine part still runs; the torch column is left empty
    torch = None
    print("WARNING: torch is not installed - the PyTorch comparison is skipped")

SHAPE = (4, 5)
TOL = 1e-6
H = 1e-6  # step of the central difference


# --------------------------------------------- the three test functions, written three times:
# 1) on plain floats (for the numerical gradient), 2) on Values, 3) on torch tensors
def f_main_float(a, b, c):
    return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))


def f_main_value(a, b, c):
    return vsum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))


def f_main_torch(a, b, c):
    return ((a * b + c) ** 2).relu().sum()


def f_reuse_float(a, b, c):
    out = 0.0
    for ai, bi, ci in zip(a, b, c):
        e = ai * bi
        out += e * e + e + max(e - ci, 0.0)
    return out


def f_reuse_value(a, b, c):
    terms = []
    for ai, bi, ci in zip(a, b, c):
        e = ai * bi
        terms.append(e * e + e + (e - ci).relu())
    return vsum(terms)


def f_reuse_torch(a, b, c):
    e = a * b
    return (e * e + e + (e - c).relu()).sum()


def _sig(x):
    return 1 / (1 + math.exp(-x))


def f_ops_float(a, b, c):
    return sum(_sig(ai) * math.log(bi ** 2 + 1) + math.exp(-ci) / (ai ** 2 + 1) - ci ** 3
               for ai, bi, ci in zip(a, b, c))


def f_ops_value(a, b, c):
    return vsum(ai.sigmoid() * (bi ** 2 + 1).log() + (-ci).exp() / (ai ** 2 + 1) - ci ** 3
                for ai, bi, ci in zip(a, b, c))


def f_ops_torch(a, b, c):
    return (torch.sigmoid(a) * torch.log(b ** 2 + 1) + torch.exp(-c) / (a ** 2 + 1) - c ** 3).sum()


TESTS = {
    "main: ((a*b+c)**2).relu().sum()": (f_main_float, f_main_value, f_main_torch),
    "reuse: sum(e*e + e + relu(e-c)), e=a*b": (f_reuse_float, f_reuse_value, f_reuse_torch),
    "all ops: sigmoid, log, exp, div, pow": (f_ops_float, f_ops_value, f_ops_torch),
}


def numerical_grad(f, inputs):
    """Central difference (f(x+h) - f(x-h)) / 2h for every element of every input."""
    grads = []
    for k in range(len(inputs)):
        g = np.zeros_like(inputs[k])
        for i in range(inputs[k].size):
            plus = [x.copy() for x in inputs]
            minus = [x.copy() for x in inputs]
            plus[k][i] += H
            minus[k][i] -= H
            g[i] = (f(*plus) - f(*minus)) / (2 * H)
        grads.append(g)
    return grads


def engine_grad(f, inputs):
    vals = [[Value(x, label=f"{n}[{i}]") for i, x in enumerate(arr)] for n, arr in zip("abc", inputs)]
    out = f(*vals)
    out.backward()
    return out, [np.array([v.grad for v in vs]) for vs in vals]


def torch_grad(f, inputs):
    if torch is None:
        return float("nan"), [np.full_like(x, np.nan) for x in inputs]
    ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    out = f(*ts)
    out.backward()
    return out.item(), [t.grad.numpy().copy() for t in ts]


def main():
    rng = np.random.default_rng(SEED)
    inputs = [rng.standard_normal(SHAPE).ravel() for _ in range(3)]   # a, b, c flattened

    summary, stats_lines = [], []
    for name, (f_float, f_value, f_torch) in TESTS.items():
        out_e, g_e = engine_grad(f_value, inputs)
        out_t, g_t = torch_grad(f_torch, [x.reshape(SHAPE) for x in inputs])
        g_t = [g.ravel() for g in g_t]
        g_n = numerical_grad(f_float, inputs)

        d_et = max(np.max(np.abs(e - t)) for e, t in zip(g_e, g_t))
        d_en = max(np.max(np.abs(e - n)) for e, n in zip(g_e, g_n))
        d_tn = max(np.max(np.abs(t - n)) for t, n in zip(g_t, g_n))
        ok = bool(d_et < TOL and d_en < TOL) if torch is not None else bool(d_en < TOL)
        summary.append({"test": name, "f_engine": out_e.data, "f_torch": out_t,
                        "max_abs_diff_engine_vs_torch": d_et,
                        "max_abs_diff_engine_vs_numerical": d_en,
                        "max_abs_diff_torch_vs_numerical": d_tn,
                        "tolerance": TOL, "pass": ok})
        st = graph_stats(out_e)
        stats_lines.append(f"{name}\n  nodes={st['nodes']} edges={st['edges']} ops={st['ops']}")
        print(f"{name:<45} engine-torch={d_et:.2e}  engine-numerical={d_en:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

        if name.startswith("main"):
            rows = []
            for n, ge, gt, gn, x in zip("abc", g_e, g_t, g_n, inputs):
                for i in range(len(x)):
                    rows.append({"input": n, "index": i, "value": x[i], "grad_engine": ge[i],
                                 "grad_torch": gt[i], "grad_numerical": gn[i],
                                 "abs_diff_engine_torch": abs(ge[i] - gt[i]),
                                 "abs_diff_engine_numerical": abs(ge[i] - gn[i])})
            pd.DataFrame(rows).to_csv(RESULTS / "gradcheck.csv", index=False, float_format="%.12g")

    pd.DataFrame(summary).to_csv(RESULTS / "gradcheck_summary.csv", index=False, float_format="%.3e")

    # one-element version of the main expression, small enough to draw
    a, b, c = Value(inputs[0][0], label="a"), Value(inputs[1][0], label="b"), Value(inputs[2][0], label="c")
    f = ((a * b + c) ** 2).relu()
    f.label = "f"
    f.backward()
    (RESULTS / "graph_example.dot").write_text(to_dot(f), encoding="utf-8")
    stats_lines.append(f"\nsingle element: f = relu((a*b + c)**2) = {f.data:.6f}\n"
                       f"  df/da = {a.grad:.6f}  (2(ab+c)·b = {2 * (a.data * b.data + c.data) * b.data:.6f})\n"
                       f"  df/db = {b.grad:.6f}  (2(ab+c)·a = {2 * (a.data * b.data + c.data) * a.data:.6f})\n"
                       f"  df/dc = {c.grad:.6f}  (2(ab+c)   = {2 * (a.data * b.data + c.data):.6f})")
    (RESULTS / "graph_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    print("\n".join(stats_lines[-1:]))


if __name__ == "__main__":
    main()
