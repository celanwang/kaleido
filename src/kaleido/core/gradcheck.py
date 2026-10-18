"""Central finite-difference validation for smooth scalar functions."""

import numpy as np

from kaleido.core.tensor import Tensor


def gradcheck(function, inputs, *, eps=1e-6, atol=1e-6, rtol=1e-4):
    """Check every coordinate, raising on mismatch; return the largest absolute error.

    Avoid non-differentiable points (e.g. ReLU at zero). This checks first derivatives
    at supplied points, not correctness over the entire input domain.
    """
    if not np.isfinite([eps, atol, rtol]).all() or eps <= 0 or atol < 0 or rtol < 0:
        raise ValueError("eps must be positive and tolerances non-negative and finite")
    values = [np.array(value, dtype=np.float64, copy=True) for value in inputs]
    tensors = [Tensor(value, requires_grad=True) for value in values]
    result = function(*tensors)
    if result.shape != ():
        raise ValueError("gradcheck requires a scalar output")
    result.backward()
    max_error = 0.0
    for argument, (value, tensor) in enumerate(zip(values, tensors, strict=True)):
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + eps
            plus = float(function(*(Tensor(v) for v in values)).data)
            value[index] = original - eps
            minus = float(function(*(Tensor(v) for v in values)).data)
            value[index] = original
            numerical[index] = (plus - minus) / (2 * eps)
        analytical = np.zeros_like(value) if tensor.grad is None else tensor.grad
        np.testing.assert_allclose(
            analytical,
            numerical,
            atol=atol,
            rtol=rtol,
            err_msg=f"gradient mismatch for input {argument}",
        )
        if numerical.size:
            max_error = max(max_error, float(np.max(np.abs(analytical - numerical))))
    return max_error
