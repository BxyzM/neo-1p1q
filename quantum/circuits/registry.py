"""
Circuit registry: maps a circuit_type string to its Circuit implementation.
Replaces the if/else string switch in QuantumClassifier.set_circuit().
Author: Aritra Bal (ETP)
Date: 2026-09-09
"""
from .base import Circuit
from .vqc import VQCCircuit, VQCExperimental001, VQCExperimental002

_REGISTRY = {
    'normal': VQCCircuit,
    'experimental_001': VQCExperimental001,
    'experimental_002': VQCExperimental002,
}


def get(circuit_type: str, num_layers: int) -> Circuit:
    """Instantiate the registered circuit for `circuit_type`.

    Args:
        circuit_type: registered circuit name ('normal', 'experimental_001',
            or 'experimental_002').
        num_layers: number of layers to construct the circuit with.

    Returns:
        A new instance of the registered Circuit implementation.

    Raises:
        ValueError: if `circuit_type` is not registered.
    """
    try:
        return _REGISTRY[circuit_type](num_layers=num_layers)
    except KeyError:
        raise ValueError(f"Unknown circuit_type '{circuit_type}'. Registered: {list(_REGISTRY)}")
