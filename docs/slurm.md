# Local models on Slurm (Juno)

With `cls: vllm_local`, the model loads inside your Slurm job: no `vllm serve`,
no ports, nothing left running afterwards. Backend details are in
[models.md](models.md#vllmlocalllm-cls-vllm_local).

Partitions used here:

| partition | for | GPUs per node |
|---|---|---|
| `normal` | setup (CPU only) | none |
| `h200` | running models | 2 × H200 (141 GB) |
| `h100` | running models | 1–4 × H100 (80–94 GB) |

## 1. One-time setup

On a `normal` node, because installs can get killed on the login node:

```bash
srun -p normal --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
module load miniconda
conda create -n vllm python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate vllm
pip install vllm                          # install first, so its torch/CUDA versions win
pip install -e /path/to/agentflow
```

## 2. Try the example

From the repo root:

```bash
mkdir -p logs && sbatch examples/captioning/slurm_vllm_local.sbatch
```

This captions 2 images with Qwen2.5-VL-3B. Follow it with `tail -f logs/af-caption_<jobid>.out`.
The first run takes about 15 minutes, mostly downloading about 7 GB of weights
and compiling the model. When it ends, the log says `done: 2 ok, 0 failed` and
the captions are in `examples/captioning/output/test_vllm_local_caption/SampleOutput/*/output.json`.
The job runs on whichever of `h100` or `h200` has a free GPU first.

## 3. Your own pipeline

Copy [`run.py`](../examples/captioning/run.py) and
[`slurm_vllm_local.sbatch`](../examples/captioning/slurm_vllm_local.sbatch)
into your project. Then change the `cd` line and the config path, and set the
`--time` limit. In the config:

```yaml
models:
  qwen:
    cls: vllm_local
    model_id: Qwen/Qwen2.5-VL-3B-Instruct    # any Hugging Face model vLLM supports
    engine_args:                             # -> vllm.LLM(...)
      max_model_len: 8192
      # tensor_parallel_size: 2              # for a model too big for one GPU; also set --gres=gpu:2
    generation_kwargs:                       # -> vllm.SamplingParams(...)
      temperature: 0
```

## Things to know

- **Resubmitting is safe.** Finished items are cached under `output/`, so a job
  that hits its time limit continues where it stopped when you resubmit it.
- **Keep the `if __name__ == "__main__":` guard** in your entry script. vLLM's
  engine process re-imports the script, and without the guard it runs the pipeline again.
- **Check `done: N ok, M failed`.** Errors such as a wrong `model_id` or running
  out of GPU memory don't crash the run. Each item fails separately and the log
  shows `Stage 0 raised ...` (vLLM's real error is in the `ERROR` lines above it).
  `run.py` then exits with code 1, so Slurm marks the job FAILED and emails you.
- **Harmless noise in the log:** "FlashInfer kernels are disabled" and, at exit,
  `--- Logging error --- ... I/O operation on closed file`.
- **Large models:** weights are cached in `~/.cache/huggingface`. For big models,
  add `export HF_HOME=/work/$USER/hf` to the job script.
- **Debugging by hand:** `srun -p h200 --gres=gpu:1 --time=01:00:00 --pty bash`,
  then run the same commands as the job script (starting with `module load miniconda`).
