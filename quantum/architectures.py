"""
Configure quantum classifiers and manage training, inference, and checkpoints.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

# pylint: disable=maybe-no-member
from itertools import combinations
import os
import pathlib
import pickle
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import helpers.utils as ut
from helpers.utils import getIndex
from quantum.circuits import registry
from quantum.circuits.base import Circuit, CircuitWeights

_jax_x64_enabled = False


def _enable_jax_x64_once() -> None:
    """Enable float64 in JAX, once, before any JAX array is created.

    Must run before the jax backend touches a single array: JAX defaults to
    float32, and enabling x64 later leaves already-created arrays at float32,
    silently diverging from the autograd path's float64 baseline. Importing
    jax stays conditional on this backend being selected.
    """
    global _jax_x64_enabled
    if _jax_x64_enabled:
        return
    import jax
    jax.config.update("jax_enable_x64", True)
    _jax_x64_enabled = True


def to_jax_pytree(weights: CircuitWeights) -> Dict[str, Any]:
    """Convert CircuitWeights into a flat dict pytree of JAX arrays.

    The narrow seam between the backend-agnostic CircuitWeights used for
    initialization/checkpointing and the jax training step, which needs a
    plain pytree to differentiate and jit through.
    """
    import jax.numpy as jnp
    pytree = {'rot': jnp.array(weights.rot)}
    pytree.update({name: jnp.array(value) for name, value in weights.aux.items()})
    return pytree


def from_jax_pytree(pytree: Dict[str, Any]) -> CircuitWeights:
    """Convert a flat dict pytree of (JAX or plain) arrays back into CircuitWeights."""
    rot = np.array(pytree['rot'], requires_grad=True)
    aux = {name: np.array(value, requires_grad=True) for name, value in pytree.items() if name != 'rot'}
    return CircuitWeights(rot=rot, aux=aux)


def _make_jax_train_step(
    circuit: qml.QNode, optimizer: Any, loss_type: str, quantum_loss: Callable
) -> Callable:
    """Build one jax.jit-compiled Adam step, closing over the fixed (non-traced)
    circuit/optimizer/loss_type. jax.jit retraces on argument shape change --
    data/labels have at most two distinct shapes per run (a regular batch and a
    smaller final partial batch), so this compiles at most twice per run, not
    once per epoch.
    """
    import jax
    import optax

    def cost_fn(params: Dict[str, Any], data: Any, labels: Any) -> Any:
        weights = CircuitWeights(
            rot=params['rot'],
            aux={key: value for key, value in params.items() if key != 'rot'},
        )
        return quantum_loss(
            weights, inputs=data, labels=labels,
            quantum_circuit=circuit, loss_type=loss_type,
        )

    @jax.jit
    def train_step(params, opt_state, data, labels):
        cost, grads = jax.value_and_grad(cost_fn)(params, data, labels)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, cost

    return train_step


class QuantumClassifier:
    """
    A Quantum Classifier that uses quantum circuits for classification tasks.
    
    This class encapsulates all the functionality needed to create, configure,
    and run quantum classification circuits using PennyLane.
    
    Args:
        wires: Number of qubits to use in the circuit
        layers: Number of layers in the quantum circuit
        shots: Number of shots for measurements on quantum states
        dev_name: Name of the quantum device to use
        use_ancilla: Whether to use an ancilla qubit
        backend_name: Backend for the QNode (e.g., 'autograd', 'torch', 'jax')
        test: If True, sets the circuit immediately for testing
        random_seed: Seed for finite-shot sampling
    """
    
    def __init__(
        self, 
        wires: int = 4,
        layers: int = 1,
        shots: Optional[int] = 5000,
        dev_name: str = 'default.qubit',
        use_ancilla: bool = False,
        backend_name: str = 'autograd',
        test: bool = False,
        random_seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        if backend_name == 'jax':
            _enable_jax_x64_once()
        # Circuit configuration
        self.n_qubits = wires
        self.num_layers = layers
        self.use_ancilla = use_ancilla
        self.backend = backend_name
        self.n_trash_qubits = -1
        self.separate_ancilla = False
        # Initialize wire configurations
        self._initialize_wires()
        
        # Set up device
        self.device = self._set_device(
            shots=shots, device_name=dev_name, random_seed=random_seed
        )
        
        # Circuit and weights
        self._impl: Optional[Circuit] = None
        self.circuit: Optional[qml.QNode] = None
        self.current_weights: Optional[CircuitWeights] = None
        
        # Data indices for particle properties
        self.index = {
            'eta': getIndex('particle', 'eta'),
            'phi': getIndex('particle', 'phi'), 
            'pt': getIndex('particle', 'pt')
        }
        
        if test:
            self.set_circuit()
    
    def _initialize_wires(self) -> None:
        """
        Initialize wire (qubit) indices and create necessary combinations.
        """
        total_qubits = self.n_qubits

        if self.use_ancilla:
            total_qubits += 1
            self.truth_wire = total_qubits - 1  # Last wire is the truth wire
        else:
            self.truth_wire = None

        self.all_wires = list(range(total_qubits))
        self.auto_wires = list(range(self.n_qubits))
        self.two_comb_wires = list(combinations(range(self.n_qubits), 2))

    @classmethod
    def from_config(cls, cfg: Any, circuit_registry: Any = registry) -> "QuantumClassifier":
        """Build the same classifier from explicit training or saved run settings."""
        required = ('wires', 'shots', 'device_name', 'num_layers', 'backend',
                    'circuit_type', 'operations_per_qubit')
        missing = [key for key in required if key not in cfg or cfg[key] is None]
        if missing:
            raise ValueError(f"Missing circuit settings: {', '.join(missing)}")
        model = cls(
            wires=cfg.wires,
            shots=cfg.shots if cfg.shots > 0 else None,
            dev_name=cfg.device_name,
            layers=cfg.num_layers,
            backend_name=cfg.backend,
            random_seed=cfg.get('random_seed'),
            test=False,
        )
        model.set_circuit(
            cfg.circuit_type,
            operations_per_qubit=cfg.operations_per_qubit,
            circuit_registry=circuit_registry,
            # 'best' (PennyLane's own QNode default) reproduces the historical
            # behaviour for configs saved before diff_method was configurable.
            diff_method=cfg.get('diff_method', 'best'),
        )
        return model
    
    def _set_device(
        self,
        shots: Optional[int],
        device_name: str,
        random_seed: Optional[int],
    ) -> "qml.devices.Device":
        """
        Set up the quantum device for simulation/execution.
        
        Args:
            shots: Number of shots for each measurement, or None for analytic execution
            device_name: Name of the quantum device
            random_seed: Seed for finite-shot sampling, or None for legacy behavior
            
        Returns:
            Initialized PennyLane device
        """
        device_options = {} if random_seed is None else {'seed': random_seed}
        device = qml.device(
            device_name, wires=len(self.all_wires), shots=shots, **device_options
        )
        print(f"Device initialized: {device}")
        return device
    
    def print_training_params(self) -> None:
        """
        Print initialized training parameters for verification.
        Includes a pause to allow user review.
        """
        print("\n Sanity check: \n")
        print('all_wires:', self.all_wires)
        print('auto_wires:', self.auto_wires)  
        print('two_comb_wires:', self.two_comb_wires)
        print('no. of layers:', self.num_layers)
        print('index:', self.index)
        print('Using truth wire (if None, then unused):', self.truth_wire)
        print('\n' + '#' * 46 + '\n')
        print("Sleep on it for 3s")
        print("Maybe you want to change something?")
        print("Then press CTRL-C")
        print('\n' + '#' * 50 + '\n')
        time.sleep(3)
        print("LETS GOOOOOOOOOOOOO")
        time.sleep(1)
    
    def set_circuit(
        self,
        circuit_type: str = 'normal',
        operations_per_qubit: Optional[int] = None,
        circuit_registry: Any = registry,
        diff_method: str = 'backprop',
    ) -> None:
        """
        Configure the QNode circuit from the circuit registry.

        Args:
            circuit_type: registered circuit name (see quantum.circuits.registry).
            operations_per_qubit: overrides the circuit's default rotation ops
                per qubit per layer (R in its RotationShape), if given.
            circuit_registry: registry module used to resolve circuit_type.
            diff_method: PennyLane QNode differentiation method (e.g. 'backprop',
                'parameter-shift', 'adjoint'). Not every method works with every
                device or shot count -- see the raised error for guidance.
        """
        self._impl = circuit_registry.get(circuit_type, self.num_layers)
        if operations_per_qubit is not None:
            self._impl.operations_per_qubit = operations_per_qubit
        if self.backend == 'jax' and self.device.shots:
            raise ValueError(
                f"backend='jax' only supports analytic execution (shots<=0), but this "
                f"device has shots={self.device.shots.total_shots}. Finite-shot sampling "
                "under the jax backend is not implemented; use backend='autograd' for "
                "finite-shot runs, or set shots<=0 for an analytic jax run."
            )
        try:
            qnode = qml.QNode(
                lambda weights, inputs: self._impl.build(weights, inputs, self.auto_wires),
                self.device,
                interface=self.backend,
                diff_method=diff_method,
            )
        except qml.exceptions.QuantumFunctionError as error:
            raise ValueError(
                f"diff_method='{diff_method}' is not compatible with device {self.device}: {error}\n"
                "Suggestion: 'parameter-shift' works on any device and any shot count, "
                "including finite shots and Hamiltonian-coefficient training (slower per "
                "step). 'backprop' is fast but only works with an analytic (shots=None), "
                "backprop-capable simulator such as default.qubit. 'adjoint' is fast on "
                "statevector simulators like lightning.qubit but cannot differentiate "
                "Hamiltonian/observable coefficients such as aux_weights.hamiltonian_coeffs."
            ) from error
        # Expand before differentiation so finite-shot parameter-shift supports
        # broadcast inputs whose encoded angles also contain trainable values.
        self.circuit = qml.transforms.broadcast_expand(qnode)

    def fetch_circuit(self) -> qml.QNode:
        """
        Get the quantum circuit for inference or training.

        Returns:
            Configured quantum node (QNode) circuit
        """
        if self.circuit is None:
            self.set_circuit()
        return self.circuit

    def fetch_backend(self) -> str:
        """
        Get the backend being used for the QNode.
        
        Returns:
            Name of the backend
        """
        return self.backend
    
    def load_weights(self, model_path: str, train: bool = False, rearrange: bool = False) -> None:
        """
        Load pre-trained weights for the classifier.
        
        Args:
            model_path: Path to the file containing the pre-trained model
            train: If True, enables gradients for the weights
            rearrange: Reorder legacy flat rotation weights from axis blocks
                into interleaved Z/Y/X order; ignored for CircuitWeights
        """
        dictionary = ut.Unpickle(model_path)
        saved_weights = dictionary['weights']
        if isinstance(saved_weights, CircuitWeights):
            # D7: split-weight checkpoints carry their own rot/aux structure --
            # just re-enable gradients on each trainable piece, nothing to rearrange.
            self.current_weights = CircuitWeights(
                rot=np.array(saved_weights.rot, requires_grad=train),
                aux={k: np.array(v, requires_grad=train) for k, v in saved_weights.aux.items()},
            )
            return
        self.current_weights = np.array(saved_weights, requires_grad=train)
        if rearrange:
            block_size = self.n_qubits * self.num_layers
            x_weights = self.current_weights[:block_size]
            y_weights = self.current_weights[block_size:2 * block_size]
            z_weights = self.current_weights[2 * block_size:3 * block_size]
            self.current_weights = np.zeros_like(self.current_weights)

            self.current_weights[2:3 * block_size:3] = x_weights
            self.current_weights[:3 * block_size:3] = z_weights
            self.current_weights[1:3 * block_size:3] = y_weights

    def print_weights(self) -> None:
        """Print the current weights of the quantum classifier."""
        print('Current weights: \n\n', self.current_weights)
    
    def run_inference(
        self, 
        dataloader: DataLoader, 
        loss_fn: Callable,
        loss_type: str = 'BCE'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on the classifier circuit using loaded weights and a dataloader.
        
        Args:
            dataloader: DataLoader yielding input data and labels. Inference runs
                one jet at a time regardless of how the loader groups them.
            loss_fn: Loss function to calculate the quantum cost (should be VQC_cost)
            loss_type: Type of loss function to use
            
        Returns:
            Tuple of (costs, scores, labels), each with shape (N_inputs,)
            
        Raises:
            ValueError: If weights are not initialized
        """
        if self.current_weights is None:
            raise ValueError(
                'Weights not initialized. Load a model first by calling load_weights(model_path)'
            )
        
        all_costs = []
        all_scores = []
        all_labels = []
        # Inference never needs batching: walk the loader's output jet by jet so
        # loss_fn always sees a single sample, whatever grouping the loader uses.
        for chunk_inputs, chunk_labels in tqdm(dataloader, desc="Running inference"):
            for i in range(len(chunk_labels)):
                cost, score = loss_fn(
                    self.current_weights,
                    inputs=chunk_inputs[i:i + 1],   # shape (1, n_qubits, 3)
                    labels=chunk_labels[i:i + 1],   # shape (1,)
                    quantum_circuit=self.circuit,
                    return_scores=True,
                    loss_type=loss_type
                )

                all_costs.append(float(cost))
                all_scores.append(float(np.reshape(score, (-1,))[0]))
                all_labels.append(float(chunk_labels[i]))
        # Convert to numpy arrays
        costs = np.array(all_costs)    # Shape: (N_inputs,)
        scores = np.array(all_scores)  # Shape: (N_inputs,)
        labels = np.array(all_labels)  # Shape: (N_inputs,)
        
        print("Inference completed")
        return costs, scores, labels


class QuantumTrainer:
    """
    A class for training quantum classification circuits.
    
    Args:
        model: The VQC circuit to be trained
        lr: Learning rate for the optimizer
        optimizer: The optimizer used for training
        loss_fn: The loss function used for optimization
        save: Whether to save the trained model and checkpoints
        train_max_n: Maximum number of training samples
        valid_max_n: Maximum number of validation samples  
        epochs: Number of training epochs
        improv: Minimum improvement threshold for early stopping
        min_epochs: Minimum number of training epochs before learning-rate decay
        decay_rate: Factor applied to the learning rate after insufficient improvement
        decay_patience: Number of learning-rate reductions allowed before early stopping
        wandb: Weights & Biases logger instance
        loss_type: Type of loss function ('BCE', etc.)
        checkpoint_config: Resolved run configuration stored in each checkpoint
        circuit_signature: Saved circuit fingerprints stored in each checkpoint
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        model: QuantumClassifier,
        lr: float = 0.001,
        optimizer: Optional[Callable] = None,
        loss_fn: Optional[Callable] = None,
        save: bool = True,
        train_max_n: int = 10000,
        valid_max_n: int = 2000,
        epochs: int = 20,
        improv: float = 0.01,
        min_epochs: int = 10,
        decay_rate: float = 0.5,
        decay_patience: int = 3,
        wandb: Optional[Any] = None,
        loss_type: str = 'MSE',
        checkpoint_config: Optional[Dict[str, Any]] = None,
        circuit_signature: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        self.model = model
        self.circuit = model.fetch_circuit()
        self.backend = model.fetch_backend()
        
        # Training parameters
        self.init_weights = kwargs.get('init_weights')
        self.batch_size = kwargs.get('batch_size', 1000)
        self.logger = kwargs.get('logger')
        
        # Training configuration
        self.train_max_n = train_max_n
        self.valid_max_n = valid_max_n
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.decay_rate = decay_rate
        self.decay_patience = decay_patience
        self.saving = save
        self.loss_type = loss_type
        self.improv = improv
        self.is_evictable = False
        self.wandb = wandb
        self.checkpoint_config = checkpoint_config
        self.circuit_signature = circuit_signature
        
        # Training state
        self.current_weights = self.init_weights
        self.current_epoch = 0
        self.optim = optimizer
        self.quantum_loss = loss_fn
        self.history: Dict[str, List[float]] = {'train': [], 'val': [], 'auc': []}
        self.n_decays = 0

        # jax backend: current_weights stays the single source of truth (read by
        # validation, checkpointing, and final certification, all backend-agnostic);
        # jax_params/jax_opt_state are the differentiable/optax-native mirror used
        # only inside iteration()'s training step. restore_checkpoint() re-derives
        # both from a restored current_weights on resume. _jax_train_step is jitted
        # once here (not per-call) so jax.jit's compilation cache actually applies --
        # a freshly redefined closure every call would defeat it, since jit's cache
        # key includes function identity, not just argument shapes.
        self.jax_params: Optional[Dict[str, Any]] = None
        self.jax_opt_state: Optional[Any] = None
        self._jax_train_step: Optional[Callable] = None
        if self.backend == 'jax' and self.current_weights is not None:
            self.jax_params = to_jax_pytree(self.current_weights)
            self.jax_opt_state = self.optim.init(self.jax_params)
            self._jax_train_step = _make_jax_train_step(
                self.circuit, self.optim, self.loss_type, self.quantum_loss
            )
        
        # Directories (to be set later)
        self.save_dir: Optional[str] = None
        self.checkpoint_dir: Optional[str] = None
        self.seed: Optional[Any] = None
        
        print(f'Performing optimization with: {self.optim} | Setting Learning rate: {lr}')
        print('Backend:', self.backend, '\n')
    
    def iteration(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        train: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Perform a single training or validation iteration.
        
        Args:
            data: Batch of input data
            labels: Corresponding labels
            train: Whether to perform training (True) or validation (False)
            
        Returns:
            Training loss (if train=True) or tuple of (validation loss, scores)
        """
        if train and self.backend == 'jax':
            self.jax_params, self.jax_opt_state, cost = self._jax_train_step(
                self.jax_params, self.jax_opt_state, data, labels
            )
            # self.current_weights must stay in sync on every call: validation,
            # save_checkpoint(), and train.py's final save_trained_run() all read
            # it directly and have no reason to know a jax branch exists.
            self.current_weights = from_jax_pytree(self.jax_params)
            return float(cost)
        if train:
            # step_and_cost needs each trainable piece as its own argument (a
            # CircuitWeights isn't itself differentiable) -- see D7. cost_fn
            # repacks rot + aux back into a CircuitWeights for the real loss.
            aux_keys = list(self.current_weights.aux.keys())

            def cost_fn(rot: np.ndarray, *aux_values: np.ndarray) -> np.ndarray:
                weights = CircuitWeights(rot=rot, aux=dict(zip(aux_keys, aux_values)))
                return self.quantum_loss(
                    weights,
                    inputs=data,
                    labels=labels,
                    quantum_circuit=self.circuit,
                    loss_type=self.loss_type
                )

            updated_args, cost = self.optim.step_and_cost(
                cost_fn,
                self.current_weights.rot,
                *(self.current_weights.aux[k] for k in aux_keys)
            )
            new_rot, *new_aux_values = updated_args
            self.current_weights = CircuitWeights(rot=new_rot, aux=dict(zip(aux_keys, new_aux_values)))
            return float(cost)
        cost, scores = self.quantum_loss(
            self.current_weights,
            inputs=data,
            labels=labels,
            quantum_circuit=self.circuit,
            return_scores=True,
            loss_type=self.loss_type
        )
        return float(cost), np.reshape(np.array(scores, requires_grad=False), (-1,))
    
    def is_evictable_job(self, seed: Optional[Any] = None) -> None:
        """
        Mark the current job as evictable and enable checkpoint copying to EOS.
        
        Args:
            seed: Random seed for checkpoint saving
        """
        self.is_evictable = True
        self.seed = seed
    
    def run_training_loop(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Execute the full training loop, including training and validation.
        
        Args:
            train_loader: DataLoader for the training dataset
            val_loader: DataLoader for the validation dataset
            
        Returns:
            Training losses, validation losses, and validation ROC AUC values.
            Validation includes epoch zero, before any training updates.
        """
        self.print_params('Initial weights: ')
        complete = False
        
        epoch_progress = tqdm(
            range(self.current_epoch, self.epochs + 1),
            desc="Epochs",
            unit="epoch",
        )
        for n_epoch in epoch_progress:
            sample_counter = 0
            batch_yield = 0
            self.current_epoch = n_epoch
            losses = 0.0
            
            # Allow the configured minimum training period before decay or stopping.
            if n_epoch > self.min_epochs and len(self.history['auc']) >= 3:
                recent_val_metrics = self.history['auc'][-2:]
                previous_val_metric = self.history['auc'][-3]
                improvement = np.mean(recent_val_metrics) - previous_val_metric
                
                if improvement < self.improv:
                    if self.n_decays < self.decay_patience:
                        self.n_decays += 1
                        if self.backend == 'jax':
                            # optax.inject_hyperparams keeps the learning rate as a
                            # mutable opt_state field, so decay is a plain field
                            # replace -- no optimizer reconstruction, no recompile.
                            hyperparams = self.jax_opt_state.hyperparams
                            new_lr = hyperparams['learning_rate'] * self.decay_rate
                            self.jax_opt_state = self.jax_opt_state._replace(
                                hyperparams={**hyperparams, 'learning_rate': new_lr}
                            )
                            current_stepsize = float(new_lr)
                        else:
                            self.optim.stepsize *= self.decay_rate
                            current_stepsize = self.optim.stepsize
                        self.logger.info(
                            f'No improvement observed over last 3 epochs. \n'
                            f'Learning rate decayed to {current_stepsize} at epoch {n_epoch}'
                        )
                    else:
                        self.logger.info(
                            f"\n\nNo improvement over last 3 epochs and {self.decay_patience} "
                            f"decay steps. Early stopping!\n\n"
                        )
                        complete = True
                        break
            
            # Training phase
            if n_epoch > 0:
                start = round(time.time(), 2)

                train_progress = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {n_epoch}/{self.epochs} training",
                    unit="batch",
                    leave=False,
                )
                for data, labels in train_progress:
                    batch_samples = data.shape[0]
                    sample_counter += batch_samples
                    batch_yield += 1
                    loss = self.iteration(data, labels=labels, train=True)
                    losses += loss * batch_samples
                    running_train_loss = losses / sample_counter
                    train_progress.set_postfix(loss=f"{running_train_loss:.4f}")

                    if self.wandb is not None:
                        self.wandb.log({'train_loss': running_train_loss})

                end = round(time.time(), 2)
                train_loss = losses / sample_counter
                self.print_params('Current weights: \n\n')

            # Validation phase
            val_loss = 0.0
            val_sample_counter = 0
            val_score = []
            val_labels = []

            val_progress = tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {n_epoch}/{self.epochs} validation",
                unit="batch",
                leave=False,
            )
            for data, labels in val_progress:
                batch_samples = data.shape[0]
                loss, score = self.iteration(data, labels=labels, train=False)
                val_loss += loss * batch_samples
                val_sample_counter += batch_samples
                val_score.append(score)
                val_labels.append(np.reshape(labels, (-1,)))
                val_progress.set_postfix(
                    loss=f"{val_loss / val_sample_counter:.4f}"
                )

            val_loss = val_loss / val_sample_counter
            val_labels = np.concatenate(val_labels)
            val_score = np.concatenate(val_score)
            val_auc = roc_auc_score(val_labels, val_score)
            val_std = np.std(val_score)
            val_score_mean = np.mean(val_score)

            epoch_metrics = {
                'val_loss': f"{val_loss:.4f}",
                'val_auc': f"{val_auc:.4f}",
            }
            if n_epoch > 0:
                epoch_metrics['train_loss'] = f"{train_loss:.4f}"
            epoch_progress.set_postfix(epoch_metrics)
            
            if self.wandb is not None:
                is_logits = self.loss_type == 'BCE'
                val_probability = ut.sigmoid(val_score) if is_logits else val_score
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.hist(val_probability[val_labels == 1], bins=30, alpha=0.5, density=True, label='signal')
                ax.hist(val_probability[val_labels == 0], bins=30, alpha=0.5, density=True, label='background')
                ax.set_xlabel('Predicted probability' if is_logits else 'Classifier score', size=16)
                ax.set_ylabel('Density', size=16)
                ax.legend(prop={'size': 14})
                self.wandb.log({
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'score_distribution': self.wandb.Image(fig),
                })
                plt.close(fig)
            
            # Logging
            if n_epoch > 0:
                self.logger.info(
                    f'Epoch {n_epoch}: Network with {len(self.model.auto_wires)} input qubits '
                    f'trained on {sample_counter} samples in {batch_yield} batches'
                )
                self.logger.info(
                    f'Epoch {n_epoch}: Train Loss = {train_loss:.3f} | Val loss = {val_loss:.3f} | '
                    f'Val preds mean and std = {val_score_mean:.3f}, {val_std:.3f} | '
                    f'Val AUC = {val_auc:.3f} | Time taken = {end-start:.3f} seconds\n\n'
                )
                self.history['train'].append(train_loss)
            else:
                self.logger.info('Initial validation pass completed')
                self.logger.info(
                    f'Epoch {n_epoch} (No training performed): Val loss = {val_loss:.3f} | '
                    f'Val AUC = {val_auc:.3f} | Val preds mean and std = {val_score_mean:.3f}, {val_std:.3f}\n\n'
                )
            
            self.history['val'].append(val_loss)
            self.history['auc'].append(val_auc)
            
            # Saving
            if self.saving:
                checkpoint_path = self.save_checkpoint()
                
                if self.is_evictable and n_epoch > 0:
                    print('Will copy over checkpoints')
                    checkpoint_name = checkpoint_path.name
                    try:
                        tmpfile = (
                            f"{os.environ['EOS_MGM_URL']}://eos/user/"
                            f"{os.environ['CERN_USERNAME'][0]}/{os.environ['CERN_USERNAME']}/"
                            f"QML/checkpoint_dumps/{self.seed}/{checkpoint_name}"
                        )
                        exec_path = os.path.join(os.environ['BELLE2_EXEC'], 'xrdcp')
                        subprocess.call(
                            f'{exec_path} {os.path.join(self.checkpoint_dir, checkpoint_name)} {tmpfile}',
                            shell=True
                        )
                    except Exception:
                        print("Failed to copy over checkpoints")
        
        if not complete:
            ut.Pickle(self.history, 'history.pickle', path=self.save_dir)
        
        return self.history
    
    def print_params(self, prefix: Optional[str] = None) -> None:
        """
        Print the current parameters (weights) of the quantum classifier.
        
        Args:
            prefix: Optional prefix to print before the parameters
        """
        if prefix is not None:
            print(prefix)
        print('autograd weights:', self.current_weights, '\n')
    
    def _optimizer_checkpoint_payload(self) -> Dict[str, Any]:
        """Build the backend-appropriate 'optimizer' checkpoint block."""
        if self.backend == 'jax':
            import jax
            import numpy as onp
            return {
                'name': 'optax_adam',
                # Leaves converted to plain NumPy: JAX array pickling is
                # device/backend-dependent and not the documented portable path.
                'opt_state': jax.tree_util.tree_map(onp.asarray, self.jax_opt_state),
            }
        return {
            'name': type(self.optim).__name__,
            'stepsize': self.optim.stepsize,
            'beta1': self.optim.beta1,
            'beta2': self.optim.beta2,
            'eps': self.optim.eps,
            'accumulation': self.optim.accumulation,
        }

    def save_checkpoint(self) -> pathlib.Path:
        """Atomically save all state needed to resume after the current epoch."""
        if self.checkpoint_dir is None:
            raise RuntimeError('Checkpoint directory has not been configured.')
        if self.checkpoint_config is None or self.circuit_signature is None:
            raise RuntimeError('Checkpoint provenance has not been configured.')
        payload = {
            'weights': self.current_weights,
            'optimizer': self._optimizer_checkpoint_payload(),
            'training': {
                'config': self.checkpoint_config,
                'implementation': self.circuit_signature,
                'completed_epoch': self.current_epoch,
                'history': self.history,
                'n_decays': self.n_decays,
            },
        }
        path = pathlib.Path(self.checkpoint_dir) / f'ep{self.current_epoch:04d}.pickle'
        temporary = path.with_suffix('.pickle.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(payload, stream)
        temporary.replace(path)
        return path

    def restore_checkpoint(self, payload: Dict[str, Any]) -> None:
        """Restore weights, optimizer state, history, and the next epoch."""
        optimizer = payload['optimizer']
        training = payload['training']
        self.current_weights = payload['weights']
        if self.backend == 'jax':
            import jax
            import jax.numpy as jnp
            self.jax_opt_state = jax.tree_util.tree_map(jnp.asarray, optimizer['opt_state'])
            # jax_params must be re-derived from the just-restored current_weights,
            # not left at whatever __init__ initialized it to -- otherwise a
            # resumed run would restore the optimizer's momentum/step-count
            # correctly but keep training from the wrong (stale/initial) weights.
            self.jax_params = to_jax_pytree(self.current_weights)
        else:
            self.optim.stepsize = optimizer['stepsize']
            self.optim.beta1 = optimizer['beta1']
            self.optim.beta2 = optimizer['beta2']
            self.optim.eps = optimizer['eps']
            self.optim.accumulation = optimizer['accumulation']
        self.history = training['history']
        self.n_decays = training.get('n_decays', 0)
        self.current_epoch = training['completed_epoch'] + 1
    
    def set_directories(self, save_dir: str) -> None:
        """
        Set up directories for saving model checkpoints and logs.
        
        Args:
            save_dir: The directory where model and checkpoints will be saved
        """
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def fetch_history(self) -> Dict[str, List[float]]:
        """
        Get the history of training losses, validation losses, and ROC AUC.
        
        Returns:
            Dictionary containing the history of training and validation metrics
        """
        return self.history
