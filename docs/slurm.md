# Local models on Slurm

How to run a pipeline on a Slurm GPU node with the model loaded **inside the
job** (`cls: vllm_local`). There is no `vllm serve` to start, no port to pick,
and nothing left running when the job ends. Backend details are in
[models.md](models.md#vllmlocalllm-cls-vllm_local).

## 1. One-time environment setup

Do this in an interactive CPU session rather than on a login node, which may
kill long installs:

```bash
srun --partition=<cpu-partition> --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
conda create -n vllm python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate vllm
pip install vllm                         # first, so its torch/CUDA pins win
pip install -e /path/to/agentflow pytest
pip check
```

vLLM's wheels ship their own CUDA libraries, so `module load cuda` is not needed.
The node only needs a recent NVIDIA driver.

## 2. Config

```yaml
models:
  qwen:
    cls: vllm_local
    model_id: Qwen/Qwen2.5-VL-3B-Instruct
    engine_args:                 # -> vllm.LLM(...)
      max_model_len: 8192
      # tensor_parallel_size: 2  # match --gres=gpu:2
    generation_kwargs:           # -> vllm.SamplingParams(...)
      temperature: 0
```

On pre-Ampere GPUs (V100, RTX 6000, T4) also add `dtype: half` and
`attention_backend: TRITON_ATTN` to `engine_args`. Hopper and Ampere GPUs
(H200, H100, A100) need neither.

## 3. Entry point

```python
# run_my_pipeline.py
import agentflow  # registers built-in backends
from agentflow.pipeline import Pipeline

def main():
    Pipeline("configs/my_config.yaml", prompt_dir="prompts").execute_all()

if __name__ == "__main__":   # required: vLLM spawns a child process that re-imports this file
    main()
```

## 4. Job script

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-pipeline
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out          # mkdir -p logs before submitting
set -euo pipefail

# `conda activate` needs the shell hook in batch shells.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export HF_HOME=/path/to/scratch/hf        # model weights cache; keep it off a small home quota

cd "$SLURM_SUBMIT_DIR"
python run_my_pipeline.py
```

```bash
mkdir -p logs && sbatch run_my_pipeline.sbatch
```

To check that the setup works before running your own pipeline, submit
[`experimental/e1119_test_vllm_local.sbatch`](../experimental/e1119_test_vllm_local.sbatch).
It captions two images and ends with `== PASSED`.

## Things to know

- **Time limits are safe.** Every finished item and stage is cached under
  `output/`, so if a job times out, resubmit the same job and it continues where
  it stopped. Fully cached runs never load the model.
- **Compute nodes without internet:** download the model on a login node first
  (`hf download <model_id>`, with the same `HF_HOME`), then add
  `export HF_HUB_OFFLINE=1` to the job script.
- **Several models in one config:** each different model (or different
  `engine_args`) loads its own engine in the same process, so the weights must
  fit in GPU memory together. Lower `gpu_memory_utilization` in each one's
  `engine_args` so their shares add up to under 1.0, for example 0.45 each.
  Entries that differ only in `generation_kwargs` share one engine.
- **Multi-GPU:** request `--gres=gpu:N` and set `tensor_parallel_size: N`.
- **Fork errors:** AgentFlow sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, which
  avoids "Cannot re-initialize CUDA in forked subprocess". That is why the
  entry point needs the `__main__` guard.
- **Debugging interactively:** `srun --partition=<gpu-partition> --gres=gpu:1 --time=01:00:00 --pty bash`,
  then run the same commands by hand. Startup errors such as out of memory or a
  bad model ID appear as a normal Python traceback.
