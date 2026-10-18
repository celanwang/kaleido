"""Stable parameter identities wrap immutable Tensor snapshots."""

import numpy as np

from kaleido.core.tensor import Tensor


class Parameter:
    def __init__(self, data):
        self.tensor = Tensor(data, requires_grad=True)

    @property
    def data(self):
        return self.tensor.data

    @property
    def grad(self):
        return self.tensor.grad

    def replace(self, data):
        value = Tensor(data, requires_grad=True)
        if value.shape != self.tensor.shape:
            raise ValueError("parameter shape cannot change")
        self.tensor = value


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self):
        seen = set()

        def visit(value, prefix):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield prefix, value
            elif isinstance(value, Module):
                for name, child in vars(value).items():
                    yield from visit(child, f"{prefix}.{name}" if prefix else name)
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    yield from visit(child, f"{prefix}.{i}")

        yield from visit(self, "")

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def state_dict(self):
        return {name: parameter.data.copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state):
        current = dict(self.named_parameters())
        if set(state) != set(current):
            raise ValueError("state keys must exactly match model parameters")
        values = {name: Tensor(value).data for name, value in state.items()}
        if any(values[name].shape != current[name].data.shape for name in current):
            raise ValueError("state shapes must exactly match model parameters")
        for name, value in values.items():
            current[name].replace(value)


class Linear(Module):
    def __init__(self, in_features, out_features, *, rng, bias=True):
        if (
            type(in_features) is not int
            or type(out_features) is not int
            or min(in_features, out_features) < 1
        ):
            raise ValueError("linear dimensions must be positive integers")
        self.weight = Parameter(
            rng.normal(scale=in_features**-0.5, size=(in_features, out_features))
        )
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        result = x @ self.weight.tensor
        return result + self.bias.tensor if self.bias is not None else result


class MLP(Module):
    def __init__(self, sizes, *, rng):
        if len(sizes) < 2:
            raise ValueError("MLP requires at least input/output dimensions")
        self.layers = [Linear(a, b, rng=rng) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
