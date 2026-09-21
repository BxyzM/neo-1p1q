#!/bin/bash
# JetGame + experiment003, both tasks, 3 random seeds each.
#
# Array index 0 -> top_vs_qcd, 1 -> w_vs_qcd. Each array task runs
# run_experiments.py, which generates 3 independent random seeds and spreads
# them across the allocated GPUs. Total: 2 tasks x 3 seeds = 6 runs.
#
# num_layers>1 is what exercises same-particle reuploading: every circuit here
# re-encodes the inputs once per layer, so num_layers=1 uploads only once.
# tests/test_reuploading.py pins that behaviour.
#SBATCH --job-name=jetgame_exp003
#SBATCH --account=hk-project-p0027888
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24                   # 3 concurrent training processes
#SBATCH --mem=32G                            # cache is 694MB and reads are bounded;
                                             # this is headroom for 3 JAX/CUDA contexts
#SBATCH --time=08:00:00                      # UNVERIFIED for this circuit -- time one
                                             # seed first and adjust (see README)
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.log

set -euo pipefail

TASKS=(top_vs_qcd w_vs_qcd)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
NUMBER_OF_RUNS="${1:-3}"
shift $(( $# < 1 ? $# : 1 ))
EXTRA_OVERRIDES=("$@")                       # further key=value args, e.g. n_signal=100000

module load devel/cuda/12.4
# The cuda module puts $CUDA_HOME/lib64 on LD_LIBRARY_PATH but NOT cuPTI, which
# lives in extras/CUPTI/lib64. jax[cuda12_local] fails to load libcupti without
# it, its plugin init aborts, and JAX silently falls back to CPU.
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pennylane-gpu-jax-sep2026

cd "$HOME/aritra/neo-1p1q"
mkdir -p logs

# W&B credentials and TRAIN_WANDB_PROJECT/TRAIN_WANDB_ENTITY come from .env,
# which is a symlink to ../../neo-1p1q/.env, read by train.py's load_dotenv().
python run_experiments.py --config configs/jetgame_exp003.yaml \
  --number-of-runs "$NUMBER_OF_RUNS" --num-processes 3 --gpu-id 0,1,2 \
  task="$TASK" \
  seed="jetgame_exp003_${TASK}" \
  save_dir="$HOME/results/jetgame_exp003" \
  dump="$HOME/results/jetgame_exp003_dump" \
  "${EXTRA_OVERRIDES[@]}"

# -> runs land under $HOME/results/jetgame_exp003/jetgame_exp003_<task>/<random_seed>/
#
# Scale the event counts off configs/jetgame_exp003.yaml's defaults
# (10000/10000 train, 2000/2000 valid) by appending overrides, e.g.
#   sbatch slurm/jetgame_exp003.sbatch 3 n_signal=100000 n_background=100000
# The cache holds 500,000 jets per class in train and 50,000 in valid.
