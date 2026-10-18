import numpy as np

from kaleido.core.nn import Linear, Module, Parameter
from kaleido.core.ops import conv2d, stack
from kaleido.core.tensor import Tensor


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, *, rng):
        if any(type(v) is not int or v < 1 for v in (in_channels, out_channels, kernel_size)):
            raise ValueError("convolution dimensions must be positive integers")
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter(
            rng.normal(scale=(in_channels * kernel_size**2) ** -0.5, size=shape)
        )
        self.bias = Parameter(np.zeros(out_channels))

    def forward(self, x):
        return conv2d(x, self.weight.tensor) + self.bias.tensor.reshape(1, -1, 1, 1)


class DigitCNN(Module):
    def __init__(self, *, rng):
        self.conv = Conv2d(1, 4, 3, rng=rng)
        self.classifier = Linear(4 * 6 * 6, 10, rng=rng)

    def forward(self, x):
        hidden = self.conv(x).relu()
        return self.classifier(hidden.reshape(x.shape[0], -1))


class SimpleRNN(Module):
    def __init__(self, input_size, hidden_size, *, rng):
        self.input = Linear(input_size, hidden_size, rng=rng)
        self.recurrent = Linear(hidden_size, hidden_size, rng=rng, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x, initial=None):
        if len(x.shape) != 3 or x.shape[1] < 1:
            raise ValueError("RNN requires non-empty (batch,time,features)")
        hidden = Tensor(np.zeros((x.shape[0], self.hidden_size))) if initial is None else initial
        if hidden.shape != (x.shape[0], self.hidden_size):
            raise ValueError("initial hidden shape does not match batch/hidden dimensions")
        history = []
        for t in range(x.shape[1]):
            hidden = (self.input(x[:, t, :]) + self.recurrent(hidden)).tanh()
            history.append(hidden)
        return stack(history, axis=1)


class MemoryClassifier(Module):
    def __init__(self, *, rng):
        self.rnn = SimpleRNN(1, 8, rng=rng)
        self.output = Linear(8, 2, rng=rng)

    def forward(self, x):
        return self.output(self.rnn(x)[:, -1, :])


class GaussianVAE(Module):
    """Diagonal Gaussian latent posterior, unit Gaussian prior, Bernoulli decoder."""

    def __init__(self, input_size, hidden_size, latent_size, *, rng):
        self.encoder = Linear(input_size, hidden_size, rng=rng)
        self.mean = Linear(hidden_size, latent_size, rng=rng)
        self.logvar = Linear(hidden_size, latent_size, rng=rng)
        self.decoder = Linear(latent_size, hidden_size, rng=rng)
        self.output = Linear(hidden_size, input_size, rng=rng)

    def encode(self, x):
        hidden = self.encoder(x).relu()
        return self.mean(hidden), self.logvar(hidden).clip(-10, 10)

    def decode(self, z):
        return self.output(self.decoder(z).relu())

    def forward(self, x, epsilon):
        mean, logvar = self.encode(x)
        epsilon = Tensor(epsilon)
        if epsilon.shape != mean.shape:
            raise ValueError("epsilon must exactly match posterior mean shape")
        z = mean + (logvar * 0.5).exp() * epsilon
        return self.decode(z), mean, logvar


def vae_loss(logits, target, mean, logvar, *, beta=1.0):
    from kaleido.numerics import nonnegative_float

    nonnegative_float(beta, "beta")
    if logits.shape != target.shape or len(logits.shape) != 2 or not all(logits.shape):
        raise ValueError("VAE reconstruction requires matching non-empty rank-2 targets")
    if np.any(target.data < 0) or np.any(target.data > 1):
        raise ValueError("Bernoulli decoder targets must lie in [0,1]")
    if mean.shape != logvar.shape or len(mean.shape) != 2 or mean.shape[0] != logits.shape[0]:
        raise ValueError("posterior shape mismatch")
    reconstruction = (logits.softplus() - target * logits).sum(axis=1).mean()
    kl = (mean**2 + logvar.exp() - logvar - 1).sum(axis=1).mean() * 0.5
    return reconstruction + beta * kl, reconstruction, kl
