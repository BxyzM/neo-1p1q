# Configuration

`base.yaml` is the default configuration for the JetClass variational quantum classifier. Both `train.py` and `evaluate.py` use the same schema. Training saves the resolved configuration with the model so that evaluation can rebuild the same circuit and data selection.

## Loading and overriding values

The project uses OmegaConf to read YAML and apply command line overrides. YAML provides the defaults, while arguments in `key=value` form replace individual entries:

```bash
python train.py \
  --config configs/base.yaml \
  seed=run_001 \
  random_seed=42 \
  data_dir=/path/to/JetClass \
  save_dir=/path/to/saved_models \
  epochs=40 \
  shots=-1
```

OmegaConf parses the value type from the command line. For example, `epochs=40` is an integer, `flat=true` is a Boolean, and `lr=0.01` is a float. Nested entries use dot notation:

```bash
python train.py aux_weights.bias=0.0 aux_weights.scale_factor=1.5
```

An entry that is not already in the YAML can also be supplied. This is useful for supported optional settings such as an explicit particle count:

```bash
python train.py seed=run_002 num_particles=12
```

Use `--print-config` to inspect the merged configuration without starting training:

```bash
python train.py --config configs/base.yaml epochs=5 --print-config
```

Avoid spaces around `=`. Quote an entire override when its value contains spaces or shell characters.

## Run and storage entries

| Entry | Purpose |
| --- | --- |
| `seed` | Names the experiment directory under `save_dir`. |
| `random_seed` | Integer controlling weight initialization, data ordering, and finite-shot sampling. Artifacts are saved under `<save_dir>/<seed>/<random_seed>/`. |
| `data_dir` | Root of the JetClass dataset. Relative paths are resolved when training starts. |
| `save_dir` | Parent directory for saved experiments and their random-seed runs. |
| `dump` | Parent directory for evaluation results. Evaluation writes to `<dump>/<seed>/<random_seed>/`. |
| `desc` | Short run description passed to Weights & Biases. |
| `save` | Enables checkpoints, history, and final model output. Keep this `true` for resumption or evaluation. |
| `resume` | Requests resumption from the latest checkpoint. Prefer the `--resume` flag described below. |

The default paths are relative to the directory where the command is run. Override them when the data or model storage is elsewhere.

## Repeated runs

`run_experiments.py` launches fresh runs, each with its own randomly generated `random_seed`:

```bash
python run_experiments.py \
  --config configs/base.yaml \
  --number-of-runs 10 \
  --num-processes 4 \
  --gpu-id 0,1 \
  seed=experiment_001
```

`--number-of-runs`, `--num-processes`, and `--gpu-id` belong to the launcher, not the YAML, and are not saved with a model. `--number-of-runs` and `--num-processes` default to 1; `--gpu-id` defaults to `0`. `--num-processes` is the maximum number of simultaneous training processes, independent of GPU count in either direction: it can exceed the number of GPUs listed in `--gpu-id` (processes then share a GPU's compute -- safe, just slower per run) or be smaller than it (some listed GPUs go unused that round). `--gpu-id` only affects `backend='jax'` runs (this pipeline has no GPU-accelerated `autograd` path); each launched process is assigned one GPU from the list in round-robin order via `CUDA_VISIBLE_DEVICES`, so it only ever sees and uses that one GPU. The launcher does not resume runs; resume an individual saved configuration with `train.py --resume`. `random_seed` is generated per run and rejected as an override on `run_experiments.py`; set it directly only when calling `train.py`.

## Data entries

| Entry | Purpose |
| --- | --- |
| `signal` | JetClass sample directory assigned label 1. |
| `background` | JetClass sample directory assigned label 0. |
| `n_signal`, `n_background` | Number of signal and background jets requested for training. |
| `n_signal_val`, `n_background_val` | Number of jets requested for validation. |
| `n_signal_test`, `n_background_test` | Number of jets requested for evaluation. |
| `batch_size` | Number of jets in each training and validation batch. The last smaller batch is retained. |
| `flat` | Uses `flat_train`, `flat_val`, and `flat_test` when `true`; otherwise uses `train`, `val`, and `test`. |
| `norm_pt` | Divides constituent transverse momentum by the jet transverse momentum when `true`. When `false`, the loader applies its fixed scaling limits. |

The loader expects this directory structure:

```text
<data_dir>/<split>/<signal>/*.h5
<data_dir>/<split>/<background>/*.h5
```

Each HDF5 file must contain `jetConstituentsList` and `jetFeatures`. Constituents must be ordered by decreasing transverse momentum.

The circuit consumes `wires * num_layers` particles per jet by default. Set `num_particles` to load more, but it cannot be smaller than that product.

## Circuit entries

| Entry | Purpose |
| --- | --- |
| `wires` | Number of data qubits. |
| `num_layers` | Number of encoding, entangling, and trainable rotation layers. Each layer consumes another group of `wires` particles. |
| `shots` | Number of measurement shots. The default `-1` selects analytic expectation values. |
| `device_name` | PennyLane device name. The default is `default.qubit`. |
| `diff_method` | PennyLane QNode differentiation method. The default is `backprop`, which differentiates `aux_weights.hamiltonian_coeffs` correctly but only works with an analytic (`shots: -1`), backprop-capable device such as `default.qubit`. `parameter-shift` works on any device and any shot count (including finite shots and `lightning.qubit`) at the cost of more circuit evaluations per step. `adjoint` is fast on statevector simulators like `lightning.qubit` but cannot differentiate `hamiltonian_coeffs` -- it silently returns a zero gradient for them. An incompatible combination (e.g. `backprop` with finite shots or `lightning.qubit`) raises a `ValueError` naming the conflict and suggesting `parameter-shift`. |
| `backend` | PennyLane QNode interface. `configs/base.yaml`'s default is `jax` (GPU-accelerated via `jax.jit` and `optax.adam`); pass `backend=autograd` for the CPU fallback. Both run in the same `pennylane-gpu-jax-sep2026` conda env (pinned `jax==0.10.2` -- do not let it float to a newer version). `jax` supports analytic execution only (`shots` must be `<=0`, rejected otherwise); `autograd` is required whenever `shots>0`. The jax path trains with `optax.adam` (wrapped in `optax.inject_hyperparams` for learning-rate decay) instead of `qml.AdamOptimizer`, and has its own checkpoint schema (`optimizer.name == 'optax_adam'`). See `llm_summary.MD` for the full design and known pitfalls. |
| `circuit_type` | Circuit name from `quantum/circuits/registry.py`. The supported value is `normal`. |
| `operations_per_qubit` | Trainable rotation parameters per qubit and layer. The current VQC requires `3` for its RZ, RY, and RX rotations. |
| `aux_weights.scale_factor` | Initial trainable scale applied during angle encoding. |
| `aux_weights.bias` | Initial trainable bias added to the circuit output before the loss is calculated. |
| `aux_weights.hamiltonian_coeffs` | Starting value for each wire's Hamiltonian coefficient. Broadcast to one independent trainable value per wire; each trains separately from this shared starting point. |

These entries define the circuit and weight shapes. Changing them when evaluating an existing run would produce a different model, so evaluation does not accept overrides.

## Optimization entries

| Entry | Purpose |
| --- | --- |
| `epochs` | Maximum number of training epochs. Validation also runs once before the first update. |
| `lr` | Initial Adam learning rate. |
| `loss` | Classifier loss. Supported values are `MSE` and `BCE`. |
| `improv` | Minimum validation AUC improvement required by the stopping logic after its initial warmup. |
| `min_epochs` | Minimum number of training epochs completed before learning-rate decay or early stopping. |
| `decay_rate` | Factor applied to the Adam learning rate after insufficient improvement. |
| `decay_patience` | Number of learning-rate reductions allowed before early stopping. |

## Saved configurations

For a new run, training merges the selected YAML with every command line override, resolves the paths, and writes the result to:

```text
<save_dir>/<seed>/<random_seed>/config.yaml
```

The run directory also contains the circuit source used for that run, checkpoints, and `trained_model.pickle`. The saved YAML is part of the model record. Do not replace it with the current `configs/base.yaml` after the run has started.

Resume from the saved settings and latest checkpoint with:

```bash
python train.py --config /path/to/saved_models/run_001/42/config.yaml --resume
```

Resumption reloads the original saved configuration. Training option overrides supplied with `--resume` do not replace the recorded settings.

Evaluation accepts a saved run configuration directly:

```bash
python evaluate.py --config /path/to/saved_models/run_001/42/config.yaml
```

It can also reconstruct that path from the run seed and model directory:

```bash
python evaluate.py --seed run_001 --random-seed 42 --model-dir /path/to/saved_models
```

Evaluation verifies the saved configuration, circuit files, and final weights before running inference. It does not use an epoch checkpoint as a substitute for `trained_model.pickle`.

Evaluate all random-seed runs beneath one experiment directory with:

```bash
python evaluate.py \
  --experiment-dir /path/to/saved_models/run_001 \
  --num-cores 4
```

`--num-cores` defaults to 1 and is used only for experiment-directory evaluation. Each numeric subdirectory counts as one run. Incomplete and failed runs are recorded in `evaluation_summary.log` while the remaining runs continue. Successful results are combined into `roc_curve_summary.png`, whose band is one standard deviation across interpolated ROC curves. The printed and logged AUC error is the standard deviation across successful runs.

Saved runs created before `random_seed` was introduced retain the historical `<save_dir>/<seed>/` layout and can still be resumed or evaluated without adding the new field to their configurations.
