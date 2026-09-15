"""
Weight types and the Circuit protocol shared by all quantum circuit implementations.
Defines the rotation-tensor/aux-weight representation and the encode/entangle/rotate/
measure/build contract a circuit must satisfy to plug into QuantumClassifier.
Author: Aritra Bal (ETP)
2026-09-05
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import pennylane.numpy as np


@dataclass(frozen=True)
class RotationShape:
    """
    Declared shape of a circuit's trainable rotation tensor.

    Attributes:
        L: number of layers.
        N: qubits addressed independently per layer (1 if a layer's rotation
           parameters are shared/broadcast across all active wires instead of
           being distinct per qubit).
        R: trainable rotation ops per qubit per layer.
    """
    L: int
    N: int
    R: int

    def validate(self, layer: int, wires: List[int]) -> None:
        """Raise if `layer`/`wires` exceed what this shape was declared for."""
        if layer >= self.L:
            raise ValueError(f"layer {layer} exceeds declared L={self.L}")
        if self.N > 1 and len(wires) > self.N:
            raise ValueError(f"{len(wires)} wires requested, declared N={self.N}")


@dataclass
class CircuitWeights:
    """
    A circuit's full trainable state: a shaped rotation tensor plus named
    auxiliary scalars (scale factor, loss bias, Hamiltonian coefficients, ...).

    Attributes:
        rot: rotation weights, shape (L, N, R).
        aux: named auxiliary weights, keyed by name (e.g. 'scale_factor').
    """
    rot: np.ndarray
    aux: Dict[str, Any] = field(default_factory=dict)


class Circuit(Protocol):
    """
    Contract every circuit implementation must satisfy. `build()` has a usable
    default (see CircuitBase) for the common "repeat every layer, measure once"
    shape; override it only when the loop structure itself differs, not the
    gates inside it.
    """

    num_layers: int
    operations_per_qubit: int
    aux_defaults: Dict[str, float]
    aux_per_wire_names: tuple = ()
    """Names in aux_defaults that hold one independent trainable value per
    wire; resolve_aux_weights() broadcasts their scalar default/override to
    one entry per wire."""
    aux_per_pair_names: tuple = ()
    """Names in aux_defaults that hold one independent trainable value per
    unordered wire pair; resolve_aux_weights() broadcasts their scalar
    default/override to one entry per pair (n_wires * (n_wires - 1) / 2)."""

    def rotation_shape(self, n_qubits: int, n_layers: int) -> RotationShape:
        """(L, N, R) for this circuit's trainable rotation tensor."""
        ...

    def encode(self, weights: CircuitWeights, inputs: np.ndarray, layer: int, wires: List[int]) -> None:
        """State preparation from data. Called however many times build() decides to."""
        ...

    def entangle(self, wires: List[int]) -> None:
        """Fixed (non-parameterized) entangling gates over the given wires."""
        ...

    def rotate(self, weights: CircuitWeights, layer: int, wires: List[int]) -> None:
        """Trainable single-qubit rotations for this layer's active wires."""
        ...

    def measure(self, weights: CircuitWeights, wires: List[int]) -> Any:
        """Terminal measurement: an expval (scalar or Hamiltonian), or a list of expvals."""
        ...

    def build(
        self,
        weights: CircuitWeights,
        inputs: np.ndarray,
        wires: List[int],
        measure_override: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """Full circuit body; the QNode is built directly from this method."""
        ...


class CircuitBase:
    """Default rotation_shape()/build() for the common per-qubit, fixed-loop case."""

    num_layers: int
    operations_per_qubit: int
    aux_defaults: Dict[str, float]
    aux_per_wire_names: tuple = ()
    aux_per_pair_names: tuple = ()

    def rotation_shape(self, n_qubits: int, n_layers: int) -> RotationShape:
        return RotationShape(L=n_layers, N=n_qubits, R=self.operations_per_qubit)

    def build(
        self,
        weights: CircuitWeights,
        inputs: np.ndarray,
        wires: List[int],
        measure_override: Optional[Callable[..., Any]] = None,
    ) -> Any:
        shape = self.rotation_shape(len(wires), self.num_layers)
        for layer in range(self.num_layers):
            shape.validate(layer, wires)
            self.encode(weights, inputs, layer=layer, wires=wires)
            self.entangle(wires)
            self.rotate(weights, layer=layer, wires=wires)
        measure = measure_override or self.measure
        return measure(weights, wires)
