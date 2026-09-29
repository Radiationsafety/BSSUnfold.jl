"""Generate the shared benchmark problem for Julia/Python cross-checks.

Writes problem.json with: E_MeV (n), A (m x n), x_true (n), b (m), x0 (n).
Deterministic: numpy default_rng(seed).
"""
import json
import sys

import numpy as np


def build(seed: int = 0, m: int = 14, n: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    E = np.logspace(-9, 1.30103, n)  # 1e-9 .. 20 MeV

    # Synthetic detector responses: broad log-Gaussian bumps at fixed centroids
    centroids = np.logspace(-9.5, 1.2, m)
    A = np.zeros((m, n))
    for i, ec in enumerate(centroids):
        s = 0.45 + 0.25 * rng.random()  # width in log10(E)
        mu = np.log10(ec) + 0.1 * rng.normal()
        g = np.exp(-0.5 * ((np.log10(E) - mu) / s) ** 2)
        A[i] = g * (1.0 + 0.15 * rng.random())
    A += 1e-4
    A /= A.sum(axis=1, keepdims=True)

    # Reference spectrum: thermal peak + 1 MeV Maxwell + fast fission tail
    lE = np.log10(E)
    x_true = (
        50.0 * np.exp(-0.5 * ((lE - (-8.5)) / 0.4) ** 2)
        + 30.0 * np.exp(-0.5 * ((lE - (-1.5)) / 0.7) ** 2)
        + 15.0 * np.exp(-0.5 * ((lE - 0.3) / 0.9) ** 2)
        + 1.0
    )
    flux = 1.0
    x0 = np.full(n, x_true.mean() * flux)

    b = A @ (x_true * flux)
    b = b * (1.0 + 0.02 * rng.standard_normal(m))

    return {
        "E_MeV": E.tolist(),
        "A": A.tolist(),
        "x_true": x_true.tolist(),
        "b": b.tolist(),
        "x0": x0.tolist(),
        "seed": seed,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bsscmp/problem.json"
    data = build()
    with open(out, "w") as fh:
        json.dump(data, fh)
    print("wrote", out)
