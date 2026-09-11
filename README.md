# 1P1Q

1P1Q trains and evaluates a variational quantum classifier on JetClass HDF5 data.

## Installation

Create the project environment and install its dependencies. This targets the recommended, GPU-accelerated JAX backend; PennyLane Autograd (CPU-only, needed for finite-shot runs) works in the same environment:

```bash
conda create -n pennylane-gpu-jax-sep2026 python=3.14 -y
conda activate pennylane-gpu-jax-sep2026
python -m pip install -r requirements.txt
```

Training uses Weights & Biases. Set `WANDB_API_KEY` for online logging or run locally with:

```bash
export WANDB_MODE=offline
```

## Data

Arrange the JetClass files by split and class:

```text
<data_dir>/
├── train/
│   ├── TTBar_/*.h5
│   └── ZJetsToNuNu/*.h5
├── val/
│   ├── TTBar_/*.h5
│   └── ZJetsToNuNu/*.h5
└── test/
    ├── TTBar_/*.h5
    └── ZJetsToNuNu/*.h5
```

Each file must contain `jetConstituentsList` and `jetFeatures`. Constituents must be ordered by decreasing transverse momentum. For flattened samples, use `flat_train`, `flat_val`, and `flat_test` and set `flat=true`.

## Training

Pass configuration changes as `key=value` arguments:

```bash
python train.py \
  --config configs/base.yaml \
  seed=run_001 \
  random_seed=42 \
  data_dir=/path/to/JetClass \
  save_dir=/path/to/saved_models
```

The run is saved under `<save_dir>/<seed>/<random_seed>/`. It contains the resolved `config.yaml`, a copy of the circuit source, epoch checkpoints, final weights, logs, history, and plots.

`backend` defaults to `jax` (GPU-accelerated) in `configs/base.yaml`; pass `backend=autograd` for the CPU fallback (also the only option for `shots>0`).

Launch repeated runs, each with a randomly generated seed, and bounded parallelism:

```bash
python run_experiments.py \
  --config configs/base.yaml \
  --number-of-runs 10 \
  --num-processes 4 \
  --gpu-id 0,1 \
  seed=experiment_001 \
  data_dir=/path/to/JetClass
```

This launches 10 runs, each with its own randomly generated `random_seed`, and keeps at most four one-thread training processes active. The launcher options are not training configuration fields and are not stored with individual models.

Resume the latest checkpoint for a run with:

```bash
python train.py --config /path/to/saved_models/run_001/42/config.yaml --resume
```

## Evaluation

Evaluate the final saved model with its run configuration:

```bash
python evaluate.py --config /path/to/saved_models/run_001/42/config.yaml
```

The same run can be selected by seed:

```bash
python evaluate.py --seed run_001 --random-seed 42 --model-dir /path/to/saved_models
```

Evaluation loads the saved circuit and `trained_model.pickle`. It writes `test_results.pickle` to `<dump>/<seed>/<random_seed>/` and `plots/roc_curve.png` to the run directory. The `dump` path comes from the saved configuration.

Evaluate every random-seed run in an experiment directory in parallel with:

```bash
python evaluate.py \
  --experiment-dir /path/to/saved_models/run_001 \
  --num-cores 4
```

Experiment mode examines each numeric subdirectory and logs incomplete or failed runs without stopping the remaining evaluations. It writes `evaluation_summary.log` and `roc_curve_summary.png` in the experiment directory. The ROC curve shows the mean true-positive rate with a one-standard-deviation band. The final report gives the total, successful, and failed run counts; jets per run and total jet evaluations; and mean AUC plus or minus its run-to-run standard deviation to four decimal places.

The same `random_seed` reproduces weight initialization, dataset ordering, and finite-shot sampling for clean runs with the same code, data, dependencies, backend, hardware, and thread settings. Finite-shot resumption does not reproduce an uninterrupted sampling stream because device RNG state is not checkpointed.
