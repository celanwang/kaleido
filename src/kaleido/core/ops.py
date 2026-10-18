"""Only the extra operators required by the selected CNN/RNN cases."""

import numpy as np

from kaleido.core.tensor import Tensor


def stack(tensors, axis=0):
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    data = np.stack([t.data for t in tensors], axis=axis)
    parents = [(t, lambda g, i=i: np.take(g, i, axis=axis)) for i, t in enumerate(tensors)]
    return Tensor._result(data, *parents)


def conv2d(x, weight):
    """NCHW cross-correlation, OIHW kernels, stride one, valid padding."""
    if len(x.shape) != 4 or len(weight.shape) != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError("conv2d requires compatible NCHW and OIHW tensors")
    n, c, h, w = x.shape
    out, _, kh, kw = weight.shape
    if min(n, c, h, w, out, kh, kw) < 1 or kh > h or kw > w:
        raise ValueError("conv2d requires non-empty inputs and kernels fitting the image")
    oh, ow = h - kh + 1, w - kw + 1
    windows = np.lib.stride_tricks.sliding_window_view(x.data, (kh, kw), axis=(2, 3))
    data = np.einsum("nchwij,ocij->nohw", windows, weight.data)

    def input_vjp(g):
        result = np.zeros(x.shape)
        for i in range(kh):
            for j in range(kw):
                result[:, :, i : i + oh, j : j + ow] += np.einsum(
                    "nohw,oc->nchw", g, weight.data[:, :, i, j]
                )
        return result

    return Tensor._result(
        data, (x, input_vjp), (weight, lambda g: np.einsum("nohw,nchwij->ocij", g, windows))
    )
