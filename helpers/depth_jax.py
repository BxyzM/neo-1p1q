"""JIT-compiled numerical backend for the matched depth/readout protocol.

PennyLane owns all quantum evolution. This module owns JAX device placement,
sample-weighted microbatch accumulation, Optax AdamW, and portable state.
"""

import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from helpers.depth_metrics import classification_metrics
from quantum.circuits.reupload import JaxReuploadingClassifier


class JaxBackend:
    def __init__(self, config, depth, readout, seed):
        # Match the Torch path's promoted PennyLane state precision. Parameter,
        # input, readout and optimizer dtypes still follow training.precision.
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        self.training = config["training"]
        requested = self.training["jax_device"]
        platform, _, index = requested.partition(":")
        try:
            devices = jax.devices(platform)
            self.device = devices[int(index or 0)]
        except (RuntimeError, IndexError) as error:
            raise RuntimeError(f"Requested JAX device {requested!r} is unavailable; no CPU fallback. "
                               "Check the JAX CUDA installation or explicitly select training.jax_device=cpu.") from error
        self.model = JaxReuploadingClassifier(
            config["circuit"]["wires"], depth, readout, seed,
            config["circuit"]["input_angle_scale"], self.training["precision"], self.device)
        self.params = self.model.params
        self.dtype = np.dtype(self.training["precision"])
        self.parameter_count = sum(value.size for value in self.params.values())
        mask = {key: key in ("bias", "readout_weights") for key in self.params}
        # Unit learning rate lets the compiled step accept the epoch LR as a
        # dynamic scalar, without rebuilding Optax or recompiling each epoch.
        self.optimizer = optax.adamw(1.0, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0,
                                    weight_decay=self.training["weight_decay"], mask=mask)
        with jax.default_device(self.device):
            self.opt_state = self.optimizer.init(self.params)
        self._step_jit = jax.jit(self._step)
        self._predict_jit = jax.jit(self.model.forward)
        self.compilation = {}

    def _step(self, params, opt_state, angles, labels, mask, lr):
        """One optimizer update; arrays have (microbatches, microbatch, ...).

        The mask excludes padded jets from both loss and gradients. Each real
        jet has weight 1 / real_batch_size, including a partial final batch.
        lax.scan accumulates gradients, not one giant backward graph over the
        optimizer batch. No sample is dropped or duplicated in the objective.
        """
        count = jnp.sum(mask)

        def loss_fn(parameters, x, y, valid):
            logits = self.model.forward(parameters, x)
            bce = jnp.logaddexp(jnp.zeros_like(logits), logits) - y * logits
            return jnp.sum(bce * valid) / count

        def accumulate(carry, batch):
            loss_sum, grads_sum = carry
            loss, grads = jax.value_and_grad(loss_fn)(params, *batch)
            return (loss_sum + loss, jax.tree_util.tree_map(jnp.add, grads_sum, grads)), None

        initial = (jnp.zeros((), dtype=labels.dtype), jax.tree_util.tree_map(jnp.zeros_like, params))
        (loss, grads), _ = jax.lax.scan(accumulate, initial, (angles, labels, mask))
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        updates = jax.tree_util.tree_map(lambda update: lr * update, updates)
        params = optax.apply_updates(params, updates)
        diagnostics = jnp.stack((loss, jnp.linalg.norm(grads["rotations"]),
                                 jnp.abs(grads["deformation"]), jnp.abs(grads["bias"]),
                                 jnp.linalg.norm(grads["readout_weights"]) if "readout_weights" in grads else jnp.zeros_like(loss)))
        finite = jnp.all(jnp.isfinite(diagnostics))
        for leaf in jax.tree_util.tree_leaves((grads, params, opt_state)):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return params, opt_state, diagnostics, finite

    def _put(self, array):
        return jax.device_put(np.asarray(array, dtype=self.dtype), self.device)

    def _training_batch(self, split=None, indices=None):
        micro = self.training["microbatch_size"]
        padded = math.ceil(self.training["batch_size"] / micro) * micro
        shape = (padded, self.params["rotations"].shape[1], 2)
        angles = np.zeros(shape, dtype=self.dtype)
        labels, mask = np.zeros(padded, dtype=self.dtype), np.zeros(padded, dtype=self.dtype)
        if split is not None:
            count = len(indices)
            angles[:count], labels[:count], mask[:count] = split.angles[indices], split.labels[indices], 1
        return (self._put(angles.reshape(-1, micro, *shape[1:])),
                self._put(labels.reshape(-1, micro)), self._put(mask.reshape(-1, micro)))

    def prepare(self):
        """Compile once per run/session, separately from timed training epochs."""
        if self.compilation:
            return
        batch = self._training_batch()
        lr = self._put(self.training["initial_lr"])
        print("JAX: compiling training step (may take time at large depth)...", flush=True)
        start = time.perf_counter()
        self._compiled_step = self._step_jit.lower(self.params, self.opt_state, *batch, lr).compile()
        train_seconds = time.perf_counter() - start
        inputs = self._put(np.zeros((self.training["eval_batch_size"], self.params["rotations"].shape[1], 2), dtype=self.dtype))
        start = time.perf_counter()
        self._compiled_predict = self._predict_jit.lower(self.params, inputs).compile()
        predict_seconds = time.perf_counter() - start
        self.compilation = {"train_compile_seconds": train_seconds, "predict_compile_seconds": predict_seconds,
                            "jit_compile_seconds": train_seconds + predict_seconds}
        print(f"JAX: compilation complete in {train_seconds + predict_seconds:.2f}s on {self.device}", flush=True)

    def predict(self, split, batch_size):
        self.prepare()
        if batch_size != self.training["eval_batch_size"]:
            raise ValueError("Prediction batch size must match the compiled configuration")
        start = time.perf_counter()
        scores = np.empty(len(split.labels), dtype=np.float64)
        for offset in range(0, len(scores), batch_size):
            count = min(batch_size, len(scores) - offset)
            inputs = np.zeros((batch_size, self.params["rotations"].shape[1], 2), dtype=self.dtype)
            inputs[:count] = split.angles[offset:offset + count]
            # device_get synchronizes execution; padding never enters metrics.
            scores[offset:offset + count] = np.asarray(self._compiled_predict(self.params, self._put(inputs)))[:count]
        inference_seconds = time.perf_counter() - start
        metric_start = time.perf_counter()
        metrics = classification_metrics(split.labels, scores)
        metrics.update(inference_seconds=inference_seconds, metric_seconds=time.perf_counter() - metric_start)
        return metrics, scores

    def train_epoch(self, split, training, seed, lr):
        self.prepare()
        start = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(split.labels), generator=generator).numpy()
        totals, updates = np.zeros(5, dtype=np.float64), 0
        for offset in range(0, len(order), training["batch_size"]):
            indices = order[offset:offset + training["batch_size"]]
            self.params, self.opt_state, diagnostics, finite = self._compiled_step(
                self.params, self.opt_state, *self._training_batch(split, indices), self._put(lr))
            diagnostics, finite = jax.device_get((diagnostics, finite))
            if not bool(finite):
                raise FloatingPointError("Nonfinite JAX BCE, gradient, parameters or optimizer state")
            totals[0] += float(diagnostics[0]) * len(indices)
            totals[1:] += diagnostics[1:]
            updates += 1
        elapsed = time.perf_counter() - start
        return {"train_online_bce": totals[0] / len(order), "train_seconds": elapsed,
                "train_events_per_second": len(order) / elapsed, "optimizer_updates": updates,
                **dict(zip(("rotation_grad_norm", "deformation_grad_abs", "bias_grad_abs", "readout_grad_norm"),
                           (totals[1:] / updates).tolist()))}

    def model_state(self):
        return jax.tree_util.tree_map(lambda value: np.array(value, copy=True), self.params)

    def optimizer_state(self):
        return {"name": "optax_adamw", "state": jax.tree_util.tree_map(np.asarray, self.opt_state)}

    def _restore_tree(self, saved, template):
        if jax.tree_util.tree_structure(saved) != jax.tree_util.tree_structure(template):
            raise ValueError("JAX checkpoint state structure mismatch")
        for value, expected in zip(jax.tree_util.tree_leaves(saved), jax.tree_util.tree_leaves(template)):
            value = np.asarray(value)
            if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
                raise ValueError("JAX checkpoint state shape, dtype or finiteness mismatch")
        return jax.tree_util.tree_map(lambda value: jax.device_put(value, self.device), saved)

    def load_model(self, state):
        self.params = self._restore_tree(state, self.params)

    def load_optimizer(self, state):
        if state.get("name") != "optax_adamw":
            raise ValueError("JAX continuation requires Optax AdamW state, not Torch checkpoints")
        self.opt_state = self._restore_tree(state["state"], self.opt_state)

    @property
    def probability_dtype(self):
        return self.model.probability_dtype

    def diagnostics(self):
        return {"deformation_factor": float(1 + 2 * math.pi * jax.nn.sigmoid(self.params["deformation"])),
                "bias": float(self.params["bias"]),
                "readout_weight_norm": float(jnp.linalg.norm(self.params["readout_weights"])) if "readout_weights" in self.params else 1.0}

    def memory(self):
        stats = self.device.memory_stats() or {}
        return {"peak_cuda_allocated_mib": stats["peak_bytes_in_use"] / 2 ** 20 if "peak_bytes_in_use" in stats else None,
                "peak_cuda_reserved_mib": stats["peak_pool_bytes"] / 2 ** 20 if "peak_pool_bytes" in stats else None}

    def runtime_info(self):
        stats = self.device.memory_stats() or {}
        return {"interface": "jax", "jax_device": str(self.device), "jax_platform": self.device.platform,
                "jax_layer_execution": "lax.scan over PennyLane default.qubit layers with differentiable unnormalized state hand-off",
                "jax_device_kind": self.device.device_kind,
                "torch_cpu_rng_threads": torch.get_num_threads(),
                "thread_semantics": "Configured limits apply to Torch/BLAS; XLA manages its own compilation/execution thread pools.",
                "jax_platform_version": getattr(self.device.client, "platform_version", None),
                "jax_enable_x64": bool(jax.config.jax_enable_x64), "jax_matmul_precision": str(jax.config.jax_default_matmul_precision),
                "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "gpu": self.device.device_kind if self.device.platform == "gpu" else None,
                "gpu_allocator_limit_mib": stats["bytes_limit"] / 2 ** 20 if "bytes_limit" in stats else None,
                "memory_semantics": "JAX device allocator high-water marks when available; null if unsupported. Not Torch allocator measurements."}
