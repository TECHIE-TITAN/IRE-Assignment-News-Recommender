#!/bin/bash
# SLURM job template for scripts/train_nrms.py on an ADA-style cluster.
#
# EVERYTHING marked <FILL IN> below is genuinely account/cluster-specific
# (partition name, module names/versions, scratch paths) -- these are not
# guessed placeholders dressed up as real values, they're intentionally
# left for you to fill in from `sinfo`/`module avail` on your own account,
# since a wrong guess here would just fail at submission with an unhelpful
# SLURM error rather than a Python traceback.
#
# Usage:
#   1. Copy the repo (and raw data -- see below) to your ADA scratch space.
#   2. Fill in the <FILL IN> lines.
#   3. sbatch scripts/train_nrms.sbatch mind      # or: sbatch scripts/train_nrms.sbatch ebnerd

#SBATCH --job-name=nrms-train
#SBATCH --partition=<FILL IN>                # e.g. `long`/`lovelace`/`ihub` -- run `sinfo` to see what's available on your account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00                       # generous first-run budget; tighten once you've timed one epoch
#SBATCH --output=logs/nrms_%x_%j.out
#SBATCH --error=logs/nrms_%x_%j.err

set -euo pipefail
mkdir -p logs

DATASET="${1:?usage: sbatch scripts/train_nrms.sbatch <mind|ebnerd>}"

# --- Environment ---------------------------------------------------------
# ADA's module system and available CUDA/conda versions change over time;
# run `module avail` yourself and swap these for what your account
# actually has. If you already built .venv locally with GPU-capable torch
# and copied it over wholesale, you may not need a `module load` for
# CUDA at all -- just activate that venv directly and skip these two lines.
# module load <FILL IN e.g. u20/cuda/11.8>
# module load <FILL IN e.g. u20/conda-python/3.9>

# If you did NOT copy an existing .venv, create one here instead (only
# needs to happen once per account, not once per job):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
source .venv/bin/activate

# --- Data staging ----------------------------------------------------------
# This script only reads data/processed/<dataset>/{articles,interactions}.parquet
# and data/models/<dataset>/{bm25,semantic,reranker}.pkl + config jsons --
# i.e. everything build_pipeline.py / build_indices.py / train_reranker.py
# already produced locally. Copy the `data/` directory's relevant
# subfolders over (not the raw MINDsmall_train.zip etc. -- those aren't
# needed for this script), e.g.:
#   rsync -av --include='processed/***' --include='models/***' --include='feature_store/***' \
#       --exclude='*' <local>/data/ <ada-scratch>/data/
# <FILL IN if your account needs a different path than the repo's own data/>

# --- Sanity run first (recommended) ---------------------------------------
# A tiny, fast CPU-or-GPU run to catch data/shape issues before spending a
# real GPU-hours budget on a full one -- takes well under a minute:
#   python scripts/train_nrms.py --dataset "$DATASET" --train_sample_size 500 \
#       --epochs 1 --eval_limit 500 --device cpu
# Uncomment to run it automatically as part of this job before the real run:
# python scripts/train_nrms.py --dataset "$DATASET" --train_sample_size 500 \
#     --epochs 1 --eval_limit 500 --device cpu

# --- Real run ---------------------------------------------------------------
python scripts/train_nrms.py \
    --dataset "$DATASET" \
    --train_sample_size 200000 \
    --epochs 5 \
    --batch_size 128 \
    --device auto

echo "Done. See data/reports/${DATASET}_nrms_eval.json"
