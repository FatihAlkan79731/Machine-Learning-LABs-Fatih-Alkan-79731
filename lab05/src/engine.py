"""A scalar automatic-differentiation engine (reverse mode).

Every arithmetic operation on `Value` objects creates a new node and records
- the parent nodes it was computed from (`_parents`), and
- a local derivative rule (`_backward`) that pushes the node's gradient to its parents.

So the forward pass builds the computation graph as a side effect. `backward()`
sorts that graph topologically and applies the local rules in reverse order
(chain rule). Gradients are *accumulated* with `+=`, because a node that is used
by several later nodes receives one gradient contribution from each of them.
"""
import math


class Value:
    __slots__ = ("data", "grad", "_parents", "_backward", "_op", "label")

    def __init__(self, data, _parents=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(_parents)
        self._backward = lambda: None   # local derivative rule, set by the op that created the node
        self._op = _op                  # name of that op, only for inspection / printing
        self.label = label

    # ------------------------------------------------------------------ basic ops
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant (int/float) exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # d(a^k)/da = k * a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # d relu(a)/da = 1 if a > 0 else 0 (the sub-gradient 0 is used at a = 0, as in PyTorch)
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------- extra ops needed for training (step 5)
    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        # numerically stable for large |x|
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # --------------------------------------- derived ops (built from the ones above)
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    # ------------------------------------------------------------------ backward pass
    def topological_order(self):
        """All nodes of the graph that ends in `self`, parents before children.

        Iterative depth-first search (a recursive one hits Python's recursion
        limit on the long graphs created during training).
        """
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        return order

    def backward(self):
        order = self.topological_order()
        for node in order:          # start from a clean state
            node.grad = 0.0
        self.grad = 1.0             # d out / d out
        for node in reversed(order):  # children before parents
            node._backward()        # accumulates (+=) into the parents
        return order

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"


# ---------------------------------------------------------------------- helpers
def vsum(values):
    """Sum a list of Values. Starts from the first element so no extra constant node is created."""
    values = list(values)
    total = values[0]
    for v in values[1:]:
        total = total + v
    return total


def graph_stats(root):
    """Number of nodes, number of edges and count per operation of the graph ending in `root`."""
    order = root.topological_order()
    ops = {}
    for n in order:
        ops[n._op or "leaf"] = ops.get(n._op or "leaf", 0) + 1
    return {"nodes": len(order), "edges": sum(len(n._parents) for n in order), "ops": ops}


def to_dot(root):
    """Graphviz DOT text of a (small) graph, for the report."""
    lines = ["digraph G {", "  rankdir=LR;", '  node [shape=record, fontname="Helvetica"];']
    for n in root.topological_order():
        name = f"n{id(n)}"
        lbl = f"{n.label + ' | ' if n.label else ''}data {n.data:.4f} | grad {n.grad:.4f}"
        lines.append(f'  {name} [label="{{ {lbl} }}"];')
        if n._op:
            lines.append(f'  {name}_op [label="{n._op}", shape=circle];')
            lines.append(f"  {name}_op -> {name};")
            for p in n._parents:
                lines.append(f"  n{id(p)} -> {name}_op;")
    lines.append("}")
    return "\n".join(lines)
