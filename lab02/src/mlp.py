"""Two-hidden-layer MLP in plain NumPy.

    x (B,64) -> [W1 (64,W), b1 (W,)] -> ReLU -> [W2 (W,W), b2 (W,)] -> ReLU
             -> [W3 (W,10), b3 (10,)] -> softmax -> p (B,10)

Loss: mean cross-entropy over the batch, L = -1/B * sum(y * log p).

Backward pass (chain rule, B = batch size):
    dz3 = (p - y) / B                      softmax + cross-entropy combined
    dW3 = a2^T dz3      db3 = sum(dz3)
    da2 = dz3 W3^T      dz2 = da2 * 1[z2 > 0]
    dW2 = a1^T dz2      db2 = sum(dz2)
    da1 = dz2 W2^T      dz1 = da1 * 1[z1 > 0]
    dW1 = x^T dz1       db1 = sum(dz1)
"""
import numpy as np

PARAM_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(p, y, eps=1e-12):
    return -np.sum(y * np.log(p + eps)) / y.shape[0]


class MLP:
    def __init__(self, width, n_in=64, n_out=10, std=0.1, rng=None):
        rng = rng if rng is not None else np.random.default_rng(42)
        self.params = {
            "W1": rng.normal(0.0, std, size=(n_in, width)),
            "b1": np.zeros(width),
            "W2": rng.normal(0.0, std, size=(width, width)),
            "b2": np.zeros(width),
            "W3": rng.normal(0.0, std, size=(width, n_out)),
            "b3": np.zeros(n_out),
        }
        self.cache = None

    def n_params(self):
        return sum(v.size for v in self.params.values())

    def forward(self, x):
        P = self.params
        z1 = x @ P["W1"] + P["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ P["W2"] + P["b2"]
        a2 = np.maximum(z2, 0.0)
        z3 = a2 @ P["W3"] + P["b3"]
        p = softmax(z3)
        self.cache = (x, z1, a1, z2, a2, p)
        return p

    def loss(self, x, y):
        return cross_entropy(self.forward(x), y)

    def backward(self, y):
        x, z1, a1, z2, a2, p = self.cache
        P = self.params
        B = y.shape[0]

        dz3 = (p - y) / B
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ P["W3"].T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ P["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def step(self, grads, lr):
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def predict(self, x):
        return self.forward(x).argmax(axis=1)
