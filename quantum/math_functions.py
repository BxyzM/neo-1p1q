"""
Differentiable loss functions.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import pennylane as qml
import pennylane.numpy as np


def mean_squared_error(predictions,targets):
    """Return squared prediction errors averaged over all array elements."""
    return qml.math.mean((predictions-targets)**2)

def binary_cross_entropy_with_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    """Return stable mean binary cross-entropy from unnormalized logits.

    Labels must lie in [0, 1] and have the same element count as logits;
    they are reshaped to match the logits before averaging over all elements.
    qml.math dispatches reshape/mean correctly for both the autograd and jax
    interfaces, but its logaddexp does not trace correctly under either one
    in this PennyLane version -- use each interface's own logaddexp instead.
    """
    labels = qml.math.reshape(labels, qml.math.shape(logits))
    if qml.math.get_interface(logits) == 'jax':
        import jax.numpy as jnp
        stable = jnp.logaddexp(0.0, logits)
    else:
        stable = np.logaddexp(0.0, logits)
    return qml.math.mean(stable - labels * logits)
