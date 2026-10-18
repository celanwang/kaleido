"""Transactional parameter updates, coupled L2 decay, per-parameter Adam clocks."""

import numpy as np

from kaleido.numerics import nonnegative_float


class SGD:
    def __init__(self, parameters, *, lr=0.01, momentum=0.0, weight_decay=0.0):
        nonnegative_float(lr, "lr", positive=True)
        nonnegative_float(momentum, "momentum")
        nonnegative_float(weight_decay, "weight_decay")
        if momentum >= 1:
            raise ValueError("momentum must be < 1")
        self.parameters = list(parameters)
        if not self.parameters or len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("optimizer requires unique non-empty parameters")
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.state = {}

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.tensor.zero_grad()

    def _update(self, index, parameter, gradient):
        velocity = self.momentum * self.state.get(index, np.zeros_like(gradient)) + gradient
        return parameter.data - self.lr * velocity, velocity

    def step(self):
        updates = []
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad)
            if gradient.shape != parameter.data.shape or not np.isfinite(gradient).all():
                raise ValueError("gradient must be finite and match parameter shape")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value, state = self._update(
                    index, parameter, gradient + self.weight_decay * parameter.data
                )
            if not np.isfinite(value).all():
                raise FloatingPointError("optimizer update is not finite")
            updates.append((index, parameter, value, state))
        for index, parameter, value, state in updates:
            gradient = parameter.grad.copy()
            parameter.replace(value)
            parameter.tensor.grad = gradient
            self.state[index] = state


class Adam(SGD):
    def __init__(self, parameters, *, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)
        if len(betas) != 2 or not all(np.isfinite(b) and 0 <= b < 1 for b in betas):
            raise ValueError("Adam betas must lie in [0,1)")
        nonnegative_float(eps, "eps", positive=True)
        self.betas, self.eps = tuple(betas), eps

    def _update(self, index, parameter, gradient):
        t, mean, square = self.state.get(
            index, (0, np.zeros_like(gradient), np.zeros_like(gradient))
        )
        b1, b2 = self.betas
        t += 1
        mean = b1 * mean + (1 - b1) * gradient
        square = b2 * square + (1 - b2) * gradient**2
        direction = (mean / (1 - b1**t)) / (np.sqrt(square / (1 - b2**t)) + self.eps)
        return parameter.data - self.lr * direction, (t, mean, square)
