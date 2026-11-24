"""A tiny MLP built from `Value` scalars (2 -> 8 -> 1 in step 5)."""
import random

from engine import Value, vsum


class Linear:
    def __init__(self, n_in, n_out, rng):
        bound = 1.0 / n_in ** 0.5          # same range as torch.nn.Linear's default init
        self.w = [[Value(rng.uniform(-bound, bound)) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [Value(rng.uniform(-bound, bound)) for _ in range(n_out)]

    def __call__(self, x):
        return [vsum(wi * xi for wi, xi in zip(row, x)) + bi for row, bi in zip(self.w, self.b)]

    def parameters(self):
        return [w for row in self.w for w in row] + self.b


class MoonsMLP:
    """x (2) -> Linear(2, 8) -> ReLU -> Linear(8, 1) -> sigmoid -> p(y = 1)."""

    def __init__(self, hidden=8, seed=42):
        rng = random.Random(seed)
        self.l1 = Linear(2, hidden, rng)
        self.l2 = Linear(hidden, 1, rng)

    def __call__(self, x):
        h = [v.relu() for v in self.l1(x)]
        return self.l2(h)[0].sigmoid()

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def state(self):
        """Weights as nested lists, ordered like torch: (W1 [8x2], b1 [8], W2 [1x8], b2 [1])."""
        return ([[w.data for w in row] for row in self.l1.w], [b.data for b in self.l1.b],
                [[w.data for w in row] for row in self.l2.w], [b.data for b in self.l2.b])


def bce(p, y):
    """Binary cross-entropy for one sample: -(y log p + (1 - y) log(1 - p)).

    Only the term that is not multiplied by 0 is built, so the graph stays small.
    """
    return -p.log() if y == 1 else -(1 - p).log()
