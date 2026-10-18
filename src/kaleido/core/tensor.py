"""Reverse-mode autodiff with immutable float64 values and explicit VJPs."""

from numbers import Real

import numpy as np

from kaleido.numerics import sigmoid


def _unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """Immutable values; gradients accumulate until zero_grad() on each tensor.

    Parameters are updated by constructing new leaf tensors, so existing graphs
    always refer to the values used in their forward pass. No in-place operators,
    GPU, batched matmul, or higher-order differentiation. P2 adds explicit indexing VJPs.
    """

    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False):
        if np.iscomplexobj(data):
            raise TypeError("Tensor supports real values only")
        self._data = np.array(data, dtype=np.float64, copy=True)
        if not np.isfinite(self._data).all():
            raise ValueError("Tensor values must be finite")
        self._data.flags.writeable = False
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = ()

    @property
    def data(self):
        return self._data.view()

    @property
    def shape(self):
        return self._data.shape

    @staticmethod
    def _as_tensor(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    @staticmethod
    def _result(data, *parents):
        active = tuple((node, vjp) for node, vjp in parents if node.requires_grad)
        result = Tensor(data, requires_grad=bool(active))
        result._parents = active
        return result

    def __add__(self, other):
        other = self._as_tensor(other)
        return self._result(
            self._data + other._data,
            (self, lambda g: _unbroadcast(g, self.shape)),
            (other, lambda g: _unbroadcast(g, other.shape)),
        )

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._as_tensor(other))

    def __rsub__(self, other):
        return self._as_tensor(other) - self

    def __mul__(self, other):
        other = self._as_tensor(other)
        return self._result(
            self._data * other._data,
            (self, lambda g: _unbroadcast(g * other._data, self.shape)),
            (other, lambda g: _unbroadcast(g * self._data, other.shape)),
        )

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if not isinstance(exponent, Real) or not np.isfinite(exponent):
            raise TypeError("power requires a finite real scalar exponent")
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            data = self._data**exponent
        # x**0 has derivative zero, including x=0; never evaluate 0 * x**-1.
        return self._result(
            data,
            (
                self,
                lambda g: (
                    np.zeros_like(g)
                    if exponent == 0
                    else g * exponent * self._data ** (exponent - 1)
                ),
            ),
        )

    def __truediv__(self, other):
        return self * self._as_tensor(other) ** -1

    def __rtruediv__(self, other):
        return self._as_tensor(other) / self

    def __matmul__(self, other):
        other = self._as_tensor(other)
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("P0 matmul supports exactly two rank-2 matrices")
        return self._result(
            self._data @ other._data,
            (self, lambda g: g @ other._data.T),
            (other, lambda g: self._data.T @ g),
        )

    def __rmatmul__(self, other):
        return self._as_tensor(other) @ self

    def sum(self, axis=None, keepdims=False):
        data = self._data.sum(axis=axis, keepdims=keepdims)
        axes = (
            tuple(range(len(self.shape)))
            if axis is None
            else (axis if isinstance(axis, tuple) else (axis,))
        )
        axes = tuple(a % len(self.shape) for a in axes) if self.shape else ()

        def vjp(g):
            if not keepdims and axes:
                g = np.expand_dims(g, axis=axes)
            return np.broadcast_to(g, self.shape)

        return self._result(data, (self, vjp))

    def mean(self, axis=None, keepdims=False):
        total = self.sum(axis=axis, keepdims=keepdims)
        axes = (
            tuple(range(len(self.shape)))
            if axis is None
            else (axis if isinstance(axis, tuple) else (axis,))
        )
        count = int(np.prod([self.shape[a] for a in axes])) if self.shape else 1
        if count == 0:
            raise ValueError("mean of an empty reduction is undefined")
        return total / count

    def reshape(self, *shape):
        return self._result(self._data.reshape(*shape), (self, lambda g: g.reshape(self.shape)))

    def __getitem__(self, index):
        # Freeze array indices as well as values: changing an index cannot rewrite a graph.
        def freeze(item):
            if isinstance(item, (np.ndarray, list)):
                return np.array(item, copy=True)
            return item

        index = tuple(freeze(i) for i in index) if isinstance(index, tuple) else freeze(index)

        def vjp(g):
            gradient = np.zeros_like(self._data)
            np.add.at(gradient, index, g)
            return gradient

        return self._result(self._data[index], (self, vjp))

    def tanh(self):
        data = np.tanh(self._data)
        return self._result(data, (self, lambda g: g * (1 - data * data)))

    def clip(self, low, high):
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("clip requires finite low < high")
        mask = (self._data >= low) & (self._data <= high)
        return self._result(np.clip(self._data, low, high), (self, lambda g: g * mask))

    def minimum(self, other):
        other = self._as_tensor(other)
        left = self._data < other._data
        equal = self._data == other._data
        return self._result(
            np.minimum(self._data, other._data),
            (self, lambda g: _unbroadcast(g * (left + 0.5 * equal), self.shape)),
            (
                other,
                lambda g: _unbroadcast(g * ((self._data > other._data) + 0.5 * equal), other.shape),
            ),
        )

    def logsumexp(self, axis=-1, keepdims=False):
        maximum = self._data.max(axis=axis, keepdims=True)
        shifted = np.exp(self._data - maximum)
        total = shifted.sum(axis=axis, keepdims=True)
        data = maximum + np.log(total)
        axes = (
            tuple(range(len(self.shape)))
            if axis is None
            else (axis if isinstance(axis, tuple) else (axis,))
        )

        def vjp(g):
            if not keepdims and axes:
                g = np.expand_dims(g, axis=axes)
            return g * shifted / total

        return self._result(data if keepdims else np.squeeze(data, axis=axis), (self, vjp))

    def softmax(self, axis=-1, *, mask=None):
        allowed = np.ones(self.shape, dtype=bool) if mask is None else np.asarray(mask)
        if allowed.dtype != bool:
            raise ValueError("softmax mask must be boolean (True means allowed)")
        allowed = np.broadcast_to(allowed, self.shape).copy()
        if not allowed.any(axis=axis).all():
            raise ValueError("softmax requires an allowed element in every row")
        masked = np.where(allowed, self._data, -np.inf)
        exp = np.exp(masked - masked.max(axis=axis, keepdims=True))
        data = exp / exp.sum(axis=axis, keepdims=True)
        return self._result(
            data, (self, lambda g: data * (g - (g * data).sum(axis=axis, keepdims=True)))
        )

    @property
    def T(self):
        return self._result(self._data.T, (self, lambda g: g.T))

    def exp(self):
        with np.errstate(over="raise"):
            data = np.exp(self._data)
        return self._result(data, (self, lambda g: g * data))

    def log(self):
        if np.any(self._data <= 0):
            raise ValueError("log requires strictly positive values")
        return self._result(np.log(self._data), (self, lambda g: g / self._data))

    def relu(self):
        return self._result(np.maximum(self._data, 0), (self, lambda g: g * (self._data > 0)))

    def softplus(self):
        return self._result(np.logaddexp(0, self._data), (self, lambda g: g * sigmoid(self._data)))

    def sigmoid(self):
        data = sigmoid(self._data)
        return self._result(data, (self, lambda g: g * data * (1 - data)))

    def backward(self, gradient=None):
        """Accumulate a vector-Jacobian product; non-scalars need an explicit seed."""
        if not self.requires_grad:
            raise ValueError("backward requires a tensor with requires_grad=True")
        if gradient is None:
            if self.shape != ():
                raise ValueError("non-scalar backward requires an explicit gradient")
            gradient = np.ones_like(self._data)
        seed = np.array(gradient, dtype=np.float64, copy=True)
        if seed.shape != self.shape or not np.isfinite(seed).all():
            raise ValueError("gradient must be finite and exactly match the output shape")

        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent, _ in node._parents)
        # Fresh adjoints per call prevent old intermediate gradients being propagated again.
        adjoints = {self: seed}
        for node in reversed(order):
            incoming = adjoints[node]
            if not np.isfinite(incoming).all():
                raise FloatingPointError("non-finite gradient in backward")
            for parent, vjp in node._parents:
                contribution = vjp(incoming)
                if not np.isfinite(contribution).all():
                    raise FloatingPointError("non-finite gradient in backward")
                adjoints[parent] = adjoints.get(parent, 0) + contribution
        # Publish only after every VJP succeeds; a failed backward leaves gradients intact.
        for node in order:
            incoming = adjoints[node]
            node.grad = incoming.copy() if node.grad is None else node.grad + incoming

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self._data)

    def __repr__(self):
        return f"Tensor({self._data!r}, requires_grad={self.requires_grad})"
