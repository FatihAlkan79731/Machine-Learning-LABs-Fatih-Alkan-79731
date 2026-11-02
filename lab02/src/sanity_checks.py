"""Steps 3-5: forward-pass checks, initial loss check and numerical gradient check.

Outputs:
  results/sanity_checks.txt  - shape / softmax / initial-loss checks
  results/grad_check.csv     - 10 random weights per width, analytic vs numerical slope

The gradient check uses the first 128 training samples. With all 1257 samples a
+-1e-5 nudge often moves some ReLU input across 0 for at least one sample; the
loss is not differentiable there and the finite difference no longer matches
the analytic slope. The column `kink_crossed` records when this happens.
"""
import numpy as np
import pandas as pd

from common import RESULTS, load_data, set_seeds
from mlp import MLP

WIDTHS = [16, 64]
EPS = 1e-5
N_CHECK = 10
TOL = 1e-6
N_GC_SAMPLES = 128


def rel_error(a, n):
    denom = max(abs(a), abs(n))
    return 0.0 if denom == 0 else abs(a - n) / denom


def relu_pattern(model):
    """Which ReLU units are active for each sample in the last forward pass."""
    _, z1, _, z2, _, _ = model.cache
    return np.concatenate([z1 > 0, z2 > 0], axis=1)


def gradient_check(model, X, Y, rng):
    model.loss(X, Y)
    grads = model.backward(Y)

    # pick 10 random weight entries across W1, W2, W3 (uniform over all weights)
    names = ["W1", "W2", "W3"]
    sizes = np.array([model.params[n].size for n in names])
    flat_ids = rng.choice(sizes.sum(), size=N_CHECK, replace=False)

    rows = []
    for fid in flat_ids:
        layer = int(np.searchsorted(np.cumsum(sizes), fid, side="right"))
        name = names[layer]
        local = fid - (sizes[:layer].sum() if layer else 0)
        idx = np.unravel_index(local, model.params[name].shape)

        w = model.params[name]
        old = w[idx]
        w[idx] = old + EPS
        loss_plus = model.loss(X, Y)
        signs_plus = relu_pattern(model)
        w[idx] = old - EPS
        loss_minus = model.loss(X, Y)
        signs_minus = relu_pattern(model)
        w[idx] = old

        numerical = (loss_plus - loss_minus) / (2 * EPS)
        analytic = grads[name][idx]
        err = rel_error(analytic, numerical)
        rows.append({
            "param": name,
            "index": str(tuple(int(i) for i in idx)),
            "analytic": analytic,
            "numerical": numerical,
            "rel_error": err,
            "kink_crossed": not np.array_equal(signs_plus, signs_minus),
            "pass": err < TOL,
        })
    return rows


def main():
    set_seeds()
    X_train, X_test, Y_train, Y_test = load_data()
    lines = [f"train: {X_train.shape}, test: {X_test.shape}"]
    all_rows = []

    for width in WIDTHS:
        model = MLP(width, rng=np.random.default_rng(42))
        lines.append(f"\n--- width {width} ({model.n_params()} parameters) ---")
        for k, v in model.params.items():
            lines.append(f"  {k}: {v.shape}")

        # step 3: forward-pass checks
        p = model.forward(X_train[:32])
        row_err = np.abs(p.sum(axis=1) - 1).max()
        lines.append(f"output shape for batch of 32: {p.shape} (expected (32, 10))")
        lines.append(f"max |row sum - 1|: {row_err:.2e} -> {'OK' if row_err < 1e-6 else 'FAIL'}")

        # step 4: initial loss
        loss0 = model.loss(X_train, Y_train)
        lines.append(f"untrained loss: {loss0:.4f} (ln(10) = {np.log(10):.4f})")

        # step 5: gradient check (float64, first 128 training samples)
        Xg, Yg = X_train[:N_GC_SAMPLES], Y_train[:N_GC_SAMPLES]
        rows = gradient_check(model, Xg, Yg, np.random.default_rng(42 + width))
        n_pass = sum(r["pass"] for r in rows)
        lines.append(f"gradient check: {n_pass}/{N_CHECK} weights with relative error < {TOL:g} "
                     f"-> {'PASS' if n_pass >= 9 else 'FAIL'}")
        for r in rows:
            r["width"] = width
        all_rows += rows

    text = "\n".join(lines)
    print(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sanity_checks.txt").write_text(text + "\n", encoding="utf-8")

    df = pd.DataFrame(all_rows)[["width", "param", "index", "analytic", "numerical", "rel_error", "kink_crossed", "pass"]]
    df.to_csv(RESULTS / "grad_check.csv", index=False, float_format="%.6e")
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
