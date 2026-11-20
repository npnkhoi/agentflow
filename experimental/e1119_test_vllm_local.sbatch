#!/usr/bin/env bash
# Slurm job that checks the in-process vLLM backend (`cls: vllm_local`) on a real GPU.
#
# Runs, from the repo root:
#   1. preflight: vLLM importable, CUDA visible, agentflow's dependencies installed
#   2. unit tests for the backend (fake vLLM; fast)
#   3. the GPU integration test: captions 2 images with Qwen2.5-VL-3B-Instruct
#      loaded inside the pytest process (~7 GB download on first run)
# AGENTFLOW_REQUIRE_VLLM=1 makes a missing vLLM/GPU a failure, not a silent skip.
#
# One-time setup: agentflow installed into the conda env that has vLLM:
#   conda activate vllm && pip install -e /path/to/agentflow pytest
#
# Usage (from the repo root; adjust partition/account to your cluster):
#   sbatch experimental/e1119_test_vllm_local.sbatch
#   sbatch --partition=gpu --export=ALL,CONDA_ENV=vllm experimental/e1119_test_vllm_local.sbatch
#   # pre-Ampere GPUs (e.g. Quadro RTX 6000, sm_75): no bf16, no FlashInfer prefill JIT
#   sbatch --export=ALL,VLLM_LOCAL_ENGINE_ARGS='{"attention_backend":"TRITON_ATTN","dtype":"half"}' \
#     experimental/e1119_test_vllm_local.sbatch
#
# Logs: logs/vllm_local_test_<jobid>.out (the logs/ dir must exist before submitting:
#   mkdir -p logs). Pipeline output lands in output/test_vllm_local_caption/.
#
#SBATCH --job-name=af-vllm-local-test
#SBATCH --partition=h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/vllm_local_test_%j.out
# Email on start and every termination (completed, failed, cancelled, timed out, requeued).
# With no --mail-user, Slurm mails the submitting user at the cluster's MailDomain.
# #SBATCH lines can't read env vars; to mail elsewhere without committing an address:
#   sbatch --mail-user="$SLURM_MAIL" experimental/e1119_test_vllm_local.sbatch
#SBATCH --mail-type=ALL

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vllm}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export AGENTFLOW_REQUIRE_VLLM=1
export VLLM_LOCAL_ENGINE_ARGS="${VLLM_LOCAL_ENGINE_ARGS:-}"
# Point at a shared cache if your home quota is small, e.g. HF_HOME=/scratch/$USER/hf
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# `conda activate` needs the shell hook in non-interactive batch shells.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) | env=$CONDA_ENV | repo=$REPO"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} | engine overrides: ${VLLM_LOCAL_ENGINE_ARGS:-none}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv || true

echo "== preflight"
python - <<'PY'
import importlib.util, sys, torch, vllm
print(f"python {sys.version.split()[0]} | vllm {vllm.__version__} | torch {torch.__version__}")
assert torch.cuda.is_available(), "torch sees no CUDA GPU"
print(f"GPU 0: {torch.cuda.get_device_name(0)} (sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
# agentflow itself is imported from the repo (cwd); its dependencies must be in this env.
missing = [m for m in ("openai", "pydantic", "yaml", "dotenv", "tqdm", "httpx", "pytest")
           if importlib.util.find_spec(m) is None]
assert not missing, f"missing in this env: {missing}; run `pip install -e . pytest` from the repo"
PY

echo "== unit tests"
python -m pytest test/unit/test_vllm_local.py -q

echo "== GPU integration test"
python -m pytest test/integration/test_vllm_local.py -v -rs

echo "== sample outputs"
for f in output/test_vllm_local_caption/SampleOutput/*/output.json; do
  echo "$f: $(cat "$f")"
done
echo "== PASSED"
