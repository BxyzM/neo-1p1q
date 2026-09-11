# Same-particle depth versus readout study

`run_depth_readout.py` reproduces the **Top/QCD depth/readout protocol** inspected
in `/work/mbinder/src/polarization`, using PennyLane `default.qubit`, JAX or Torch
backpropagation and `shots=None`. All quantum gate execution and probabilities
come from PennyLane. There is no native simulator, fallback simulator, or
runtime import of `polarization`.

This study has a separate entry point because its cached-angle inputs,
initialization, AdamW schedule and validation selection differ from the existing
`train.py` JAX/Adam or Autograd/Adam workflow. They share the circuit layer loop, ordered
CNOT ring and trainable rotation implementation.

## Four parallel JAX launch commands

From the repository root, run each command in its own terminal/job. The intended
environment for these commands is upstream's `pennylane-gpu-jax-sep2026`, with
JAX/Optax and the project dependencies installed. Pulling code does not install
that environment. GPU numbering below refers to the two physical GPUs; each
process sees its selected GPU as logical `gpu:0`. `--jax` selects the JAX overlay
and a **new output directory**, `saved_models/depth_readout_jax`.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pennylane-gpu-jax-sep2026 python run_depth_readout.py --jax --dataset jetclass --shard 0/2 --resume
```

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n pennylane-gpu-jax-sep2026 python run_depth_readout.py --jax --dataset jetclass --shard 1/2 --resume
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pennylane-gpu-jax-sep2026 python run_depth_readout.py --jax --dataset jetgame --shard 0/2 --resume
```

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n pennylane-gpu-jax-sep2026 python run_depth_readout.py --jax --dataset jetgame --shard 1/2 --resume
```

The config assumes **two 48 GB CUDA GPUs**, with two training workers on each.
Each launcher runs one child at a time, with Torch/BLAS thread limits set to one;
XLA manages its own compilation and execution thread pools. A deterministic
partition balances the sum of circuit depths between the two shards of each
dataset; this is a cost estimate, not a guarantee of equal wall time. Together
the commands cover exactly 390 distinct runs (195 per dataset). They do not
launch 390 concurrent processes.

The launcher defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX,
including in child workers, so two processes do not each eagerly reserve most
of a GPU. Explicit environment settings are preserved; avoid overriding this
to `true` for shared GPUs. GPU selection fails clearly when unavailable; there
is no silent CPU fallback. To verify the environment before launching:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n pennylane-gpu-jax-sep2026 python -c "import jax; print(jax.devices('gpu'))"
```

Without `--jax`, the previous Torch config/output remains the default, using
`pennylane-gpu-sep2026` with CUDA-enabled Torch. The JAX requirements use CPU
Torch only for the matched RNG and artifact serialization. JAX and Torch runs
must not share output directories, checkpoints or aggregate tables. Existing
Torch results are left intact. Source changes from this port also mean old
Torch checkpoints require their matching pre-port checkout for continuation.

The optimizer batch is 1024 jets and the simulation microbatch is 64 jets
(16 accumulated backward passes per update). Validation batches are 1024.
JAX compiles both training and prediction. A compiled `lax.scan` accumulates
microbatch gradients before one Optax AdamW update. Zero-padded final batches
use explicit loss masks and their real event count; padded validation scores
are removed before metrics. No real event is discarded, and varying final-batch
sizes do not cause new compilations. The LR is a dynamic scalar, not a new
compiled function each epoch.

The depth loop also uses `lax.scan`: PennyLane initializes |0>, a compiled
PennyLane layer executes the shared gate sequence, and the full analytic state
is passed to the next layer with `qml.StatePrep(normalize=False,
validate_norm=False)`. Final probabilities are computed by PennyLane. The
state is never reset, renormalized, measured or detached between layers. This
keeps all quantum operations in `default.qubit` while avoiding a huge unrolled
compilation graph at L=128. It is a simulation execution strategy, not a change
to the scientific gate sequence or a claim of finite-shot/hardware execution.
Microbatching preserves the sample-weighted gradient mathematically, including
partial batches, but floating-point reduction order can change the last bits.
The microbatch choice is a starting point for the supplied GPU capacity, not
a measured CUDA memory guarantee. A changed microbatch is an explicit override,
for example `training.microbatch_size=32`; use the same override for all shards
and a new `output_dir` if the existing run configurations differ.

All four commands must use the same scientific config and output directory.
`CUDA_VISIBLE_DEVICES` selects hardware without changing the scientific config.
Do not assign one shard `jax_device=gpu:1` and another `gpu:0`; remap physical
devices with `CUDA_VISIBLE_DEVICES` as above instead.

`--resume` skips completed runs only after verifying config/source/data identity,
and resumes interrupted runs from their last completed epoch with AdamW state.
An incomplete epoch is repeated. Locks prevent two workers writing the same
run or concurrently building the same prepared subset. Source/config/dependency
changes are rejected for resumed checkpoints; select a new output directory
for a new protocol. Epoch checkpoints do not certify convergence.

Add `--dry-run` to any command to inspect its jobs without loading data, requiring
CUDA, or starting training. A small CPU integration smoke run is also possible:

```bash
conda run --no-capture-output -n pennylane-gpu-jax-sep2026 python run_depth_readout.py --jax \
  --dataset jetgame --shard 0/1 output_dir=./saved_models/depth_smoke \
  'depths=[1,2]' 'readouts=[all_z_zz]' 'seeds=[12345]' \
  datasets.jetgame.max_train_events=8 datasets.jetgame.max_valid_events=8 \
  training.epochs=1 training.warmup_epochs=1 training.batch_size=8 \
  training.microbatch_size=4 training.eval_batch_size=4 training.jax_device=cpu
```

## Fixed protocol and parameter semantics

The scientific defaults live in `configs/depth_readout.yaml`; `--jax` merges the
small `configs/depth_readout_jax.yaml` overlay, removing `training.torch_device`
and adding `training.jax_device`. The latter accepts `cpu`, `gpu`, or `gpu:N`,
where N is a nonnegative logical device index (default `gpu:0`). Torch retains
`training.torch_device=cpu|cuda[:N]`. Only the selected backend's field is valid.
Typed `key=value` overrides
use OmegaConf. Unrecognized config fields fail instead of being silently ignored.

Protocol evidence was checked against
`polarization/scripts/run_vqc_matched_q10.sh`,
`scripts/run_jetclass_vqc_readout_depth_scan.sh`,
`artifacts/vqc_native_progress_20260910/summary.json`, and the saved
`configuration.json` files under the JetClass `jetclass_readout_full_20260901`
and JetGame `jetgame_q10_matched_20260908` result directories. The matched script
currently defaults to batch 512, while the inspected historical saved runs use
1024; this configuration adopts the saved-run batch and schedule. JetClass's
full validation split is retained. Historical results without an execution
backend recorded in their config are not relabelled as PennyLane results.

| Setting | Value / meaning |
| --- | --- |
| Task | Top signal=1, QCD background=0; canonical cache raw labels 2 and 0 |
| Datasets | Full (not flattened) JetClass Top/QCD and JetGame 500 GeV Top/QCD |
| Selected particles / wires | 10 hardest constituents, fixed across depths |
| Depth grid | 1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128 |
| Seeds | 12345, 23456, 34567, 45678, 56789 |
| Training events | 200000 total per dataset, 100000 per class |
| Validation events | JetClass: full 1000000, 500000 per class; JetGame: 100000, 50000 per class |
| Training duration | 25 complete epochs; no early stopping |
| Loss | Mean BCE with logits, `logaddexp(0, score) - label*score` |
| Optimizer | Torch AdamW or Optax AdamW, betas (0.9, 0.999), epsilon 1e-8 outside the square root |
| Weight decay | 1e-4 for bias/readout coefficients; zero for rotations/deformation |
| LR schedule | 0.001 to 0.005 linearly over epochs 1–5; cosine to 0.001 at epoch 25 |
| Initialization | Torch local seeded float32 `U(0, pi)` stream for rotations then deformation; zero bias |
| Precision | Float32 parameters, inputs and readout; record PennyLane's actual probability output dtype |
| Simulator | `default.qubit`, `interface='torch'` or `'jax'`, `diff_method='backprop'`, `shots=None` |
| Model selection | Best validation ROC AUC among epochs 1–25; first epoch wins ties |
| Data I/O | At most `loader.block_rows=65536` rows of feature scratch storage per read, aligned to HDF5 chunks when possible |

`max_*_events` budgets are totals across two balanced classes, not per-class
counts. `null` reads the whole balanced task split. Insufficient samples,
malformed data and unbalanced full splits fail explicitly.

For each seed, subset selection matches `polarization` exactly: NumPy's local
generator uses seed+10000 for train, seed+10001 for validation, sampling QCD then
Top without replacement. Training shuffles with a local Torch generator seeded
by seed+zero-based epoch. The same seed/depth's three readouts start with matching
rotations/deformation and identical initial predictions up to numerical rounding.
Across depths the deformation draw follows a differently sized rotation tensor,
as it does in the reference. Across seeds both initialization and sampled events
change; the reported spread is not initialization-only uncertainty.

## Scientific model

The cache stores base angles `(z_i * delta_eta_i, z_i * delta_phi_i)`, with
`z_i = pt_i / jet_pt`, jet-relative eta, and phi wrapped to [-pi, pi).
Zero-pt padding has zero base angles. The input scale is `3.926990817`, the
reference scan's approximation to pi/0.8. A trainable global deformation
`f(w) = 1 + 2*pi*sigmoid(w)` multiplies both angles.

Every layer reuploads the **same** ten particles onto the same ten wires, without
resetting the quantum state:

1. Each wire receives `RY(f * scale * z*deta)`, then `RX(f * scale * z*dphi)`.
2. Apply sequential CNOTs 0→1, 1→2, …, 9→0 in that order.
3. Each wire receives `Rot(0, ry, rz)` followed by `RX(rx)`, with independent
   trainable parameters per wire and layer. Storage order is `(rz, ry, rx)`.

Measure once after the final layer. The logit is `<H> + bias`:

| Readout | Observable H | Parameters at 10 wires |
| --- | --- | --- |
| `z0` | Z0 | 30L + 2 |
| `all_z` | sum_i a_i Zi | 30L + 12 |
| `all_z_zz` | sum_i a_i Zi + sum_{i<j} b_ij ZiZj | 30L + 57 |

All 45 unordered pairs appear in the composite Hamiltonian, not just adjacent
pairs. Coefficients start at a_0=1 and all other a_i=b_ij=0. The PennyLane QNode
returns computational-basis probabilities once; classical parity signs evaluate
these commuting observables and their trainable weighted sum. This is exactly
the analytic Hamiltonian expectation, tested against `qml.expval(H)` for values
and gradients. It does not approximate probabilities with finite shots.

Parameter counts above are nominal: the final rotations on non-readout wires
cannot influence `z0` (27 rotation parameters at ten wires). This structural
limitation is retained from the reference and recorded. All-Z readouts also have
more trainable parameters and can learn a wider logit range: compare BCE together
with ROC AUC, calibration and parameter counts when interpreting improvements.

The [1P1Q paper](https://arxiv.org/html/2502.17301v2) establishes the particle
encoding. Reuploading and these trainable readout comparisons are project
extensions. This workflow reproduces the local `polarization` scan protocol;
it does not assert reproduction of the paper's complete published experiment.
The legacy repository's successive particle blocks and slightly offset affine
eta/phi rescaling are different from this cached reuploading experiment.
[PennyLane's backprop documentation](https://docs.pennylane.ai/en/stable/introduction/unsupported_gradients.html)
requires analytic shots for backpropagation.

One explicit numerical distinction from the current polarization Torch wrapper:
it casts the returned state amplitudes before calculating probabilities; here
PennyLane calculates probabilities and they are then cast for the readout.
Float64 forward/gradient parity is verified. Float32 rounding can differ, so
identical results across implementations or machines are not asserted.

For JAX, x64 is enabled so PennyLane can preserve the promoted state precision
seen on the Torch path. Input, parameter, readout and optimizer precision remains
the configured float32/float64. JAX matrix multiplication uses `highest` precision
to avoid backend-default reduced-precision products in this comparison. JIT and
different arithmetic/reduction ordering still preclude bitwise cross-interface
equivalence. Switching execution does not switch the initialization to JAX's RNG.

Implementation references: [PennyLane's JAX interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/jax.html),
[Optax AdamW](https://optax.readthedocs.io/en/stable/api/optimizers.html#optax.adamw),
and [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

## Saved evidence and statistical analysis

Runs are stored under
`<output_dir>/<dataset>/L<depth>/<readout>/<seed>/`:

- `configuration.json`: resolved protocol and implementation hashes.
- `source/`: source snapshot of circuit, training, configuration and loader code.
- `data.json`: source cache metadata, selected-data hashes, and prepared subset location.
- `initial_metrics.json`: before-training metrics on both training and validation data.
- `history.json`, `history.csv`: every epoch's online train BCE, validation metrics,
  LR, gradient norms, train seconds, validation inference/metric seconds, total
  epoch seconds, throughput, deformation, bias, and readout norm.
- `last_checkpoint.pt`: model, AdamW state, full history and best model, for exact
  epoch-boundary continuation in a matching environment (not bitwise guaranteed across hardware).
  JAX stores NumPy-converted parameter/Optax trees in the same Torch serialization
  container; restoring checks backend identity, shapes, dtypes and finite values.
- `models.pt`: best-validation-AUC and final trained weights.
- `validation_predictions.npz`: best-model logits, labels, row indices, event identifiers.
- `sessions.json`: per-session hardware, package versions, threads, start epoch,
  load time, and separate training/prediction JIT compilation seconds.
- `results.json`: completion marker and the full scientific result.

Final fixed-weight training BCE is evaluated again after training. This makes
the initial-to-final BCE drop comparable; the online loss within an epoch mixes
different intermediate model states. Results include absolute and percentage
BCE drops, final and minimum validation BCE, BCE at the best-AUC epoch, the
corresponding epochs, and final train/validation BCE gap.

Classification metrics include ROC AUC, average precision, accuracy/balanced
accuracy at logit=0, Brier score, confusion counts, class mean logits, score
spread, and empirical background rejection at signal efficiencies 30%, 50%,
and 80%. Thresholds and actual achieved efficiencies/counts are saved. Zero
accepted background gives `null` rejection rather than an unsupported infinity.

Training time includes batch transfer, circuit construction/execution,
backpropagation, optimizer updates and gradient diagnostics. `epoch_seconds`
adds validation inference and metrics but excludes checkpoint I/O and JIT compilation.
The runner reports compilation before initial evaluation, separately per session;
`total_jit_compile_seconds` includes recompilation after resume. Large-depth
compilation can take substantial time. A few-second epoch in the upstream
small-data workflow is not a prediction for this 200000-event / up-to-1M-validation
scan. Benchmark this exact workload on the target GPUs before extrapolating.
Initial/final
diagnostic passes and data preparation are separately timed. CUDA timers
synchronize at measurement boundaries (JAX results are materialized on the host).
Peak process RSS and device allocator statistics are reported. For JAX, the
legacy `peak_cuda_allocated_mib`/`peak_cuda_reserved_mib` columns contain
`peak_bytes_in_use`/`peak_pool_bytes` from JAX's device allocator, respectively,
or null if unsupported; these are not Torch allocator values or total GPU usage.
Timing runs sharing a GPU measure that concurrent workload,
not exclusive-GPU latency; retain the recorded hardware and thread settings.

Each finished launcher refreshes `<output_dir>/analysis/` under a
lock. The final launcher produces the complete combined report. It contains:

- `runs.csv`: one completed run per row, including all reporting metrics.
- `depth_readout.csv`: means, sample standard deviations (ddof=1), SEM, and
  Student-t 95% intervals of means across the available seeds.
- `paired_vs_z0.csv`: within-seed readout-minus-Z0 differences, with the same statistics.
- `coverage.json`: completed/expected counts and every missing run.
- `<dataset>_depth_readout.png` and `.pdf`: AUC, final BCE, and training time versus
  depth, with mean ± sample SD.

To refresh a JAX report while runs are in progress, use
`python run_depth_readout.py --jax --summarize` with the same config and overrides.
Incomplete groups remain clearly
marked; missing runs are not substituted with partial checkpoints. Different
code versions, dependencies or source caches are rejected when aggregating.

These are **validation-scan results**, matching the reference's validation-only
protocol. No test split is read. Selecting among depths/readouts on validation
and reporting the same validation optimum is not an unbiased test-performance
estimate. A final scientific claim needs a frozen model-selection rule and
untouched test evaluation. Five-seed intervals describe run variability, not
independent event-level coverage or significance after multiple comparisons.

## Loader comparison

The raw `case_reader.py` loader is already efficient at limiting leading event
counts and materializes each split once per run. However, it reads all stored
constituents before slicing, recomputes feature scaling for every run, and converts
NumPy → Torch float32 → PennyLane NumPy → Torch on the active loader path. It
validates descending pt order but records no event identities and only warns on
insufficient statistics. Its shuffle order is fixed across epochs.

`polarization` caches pt ordering, padding masks, wrapped jet-relative angles and
pt fractions once, supports both datasets, and preserves selected event IDs.
This is useful for a repeated scan. Its `hdf5_rows` reads an entire dataset when
more than 2048 rows are requested: the JetClass training angle array alone is
18,000,000 × 10 × 2 × 4 bytes = 1.44 GB even for a 200,000-event selection.

The new loader preserves those exact features and RNG selections while reading
bounded, chunk-aligned blocks. It prepares each dataset/seed once, atomically,
and shares immutable `.npy` memory maps across all depths/readouts. Only selected
batches are transferred to the selected Torch/JAX device. Larger source reads can still win on
single-pass wall time; this change prioritizes bounded memory and reuse, without
claiming every HDF5 access is faster. The cache still needs to be supplied if
running on another machine; this workflow does not rebuild raw feature caches.

JetGame event identifiers originate from audited content fingerprints. JetClass's
cache uses deterministic split/row IDs, so identity checks there cannot prove the
absence of duplicated physical events across source files.

## Local validation and its limits (2026-09-11)

- 34 targeted tests passed, including explicit-Hamiltonian value/gradient parity,
  a depth-128 backward pass, matching initial readouts, partial-batch gradient
  accumulation, source row order, seed sampling, checkpoint resume equivalence,
  metrics/statistics, report generation and legacy circuit regression tests.
- JAX-specific checks cover float32/float64 logits, BCE and gradients for all
  readouts, masked partial batches, AdamW updates with changing LR, exact CPU
  checkpoint continuation, rejection of unavailable GPUs/nonfinite values,
  four-shard coverage, and real subprocess launch with the JAX config overlay.
  Depth-128 parity was checked against the full unrolled Torch circuit at two
  wires/float64 and ten wires/float32, including rotation/deformation/head gradients.
- Full-suite result: 104 of 107 tests passed. Three original `test_saved_run`
  entry-point tests fail because this local interpreter lacks `python-dotenv`,
  which is already listed in project requirements. They need a rerun in the
  intended environment; this is not a fully passing production-stack suite.
- A ten-wire, depth-128 JAX CPU probe with batch 8 and microbatch 4 compiled
  training/prediction in 6.38 s and took 0.78/0.86 s for two subsequent AdamW
  updates. These are tiny synthetic measurements, not production epoch times
  or an apples-to-apples GPU speedup benchmark.
- A direct comparison with the current polarization PennyLane circuit at three
  wires and three layers found maximum float64 logit difference 5.6e-17 and
  matching rotation/deformation/bias gradients for all three readouts.
- A ten-wire, depth-128, eight-jet CPU forward/backward probe completed in
  12.64 s with finite gradients and 1777 MiB peak process RSS. Its PennyLane
  probability output was float64 despite float32 model parameters. This is a
  synthetic plumbing/memory probe, not a production timing result.
- Real JetClass and JetGame validation caches were read successfully; a one-epoch
  JAX smoke run with eight training/eight validation events completed for each
  dataset, including the CLI worker and combined report. Their BCE changes are
  not statistically useful performance results.
- Full configured data preparation and identifier checks passed for seed 12345:
  JetClass 200000 train / 1000000 validation, JetGame 200000 train / 100000
  validation. These were data checks only, not full training runs.
- On identical 200000-event JetClass training selections, the reference full
  angle-array read peaked at 1507 MiB process RSS; the bounded read at 150 MiB.
  Output SHA-256 hashes matched. Observed bounded times were 59.4 s on its first
  pass and 10.1 s on a repeat, with a 19.4 s reference pass in between. Cache
  warming and shared filesystem effects preclude a general speedup claim.

These checks used the existing ParT interpreter (PennyLane 0.42.3, Torch 2.9.1,
JAX 0.6.2, Optax 0.2.5, OmegaConf 2.3.1), because the named project environments
were unavailable in the checked local installations. No CUDA device was
available, so GPU execution, GPU memory at microbatch 64, and the full 390-run
study have not been verified here. No environments or dependencies were
installed/changed. Full intended-environment/GPU validation remains due.
