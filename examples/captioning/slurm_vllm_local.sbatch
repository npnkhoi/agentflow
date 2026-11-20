#!/usr/bin/env bash
# Caption the example images with Qwen2.5-VL-3B loaded inside the job (cls: vllm_local).
# Submit from the repo root (see docs/slurm.md for the one-time conda setup):
#   mkdir -p logs && sbatch examples/captioning/slurm_vllm_local.sbatch
#SBATCH --job-name=af-caption
#SBATCH --partition=h100,h200        # runs on whichever frees up first
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL      # goes to <netid>@utdallas.edu
set -euo pipefail

module load miniconda                                  # Juno: conda comes from this module
source "$(conda info --base)/etc/profile.d/conda.sh"   # makes `conda activate` work in batch jobs
conda activate "${CONDA_ENV:-vllm}"

cd "$SLURM_SUBMIT_DIR/examples/captioning"
python run.py configs/vllm_local_caption.yaml
echo "outputs in examples/captioning/output/test_vllm_local_caption/"
