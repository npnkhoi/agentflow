"""Run one pipeline config over all items; exit 1 if any item failed.

Run from this directory (config paths are relative to it):
    python run.py                                # configs/vllm_local_caption.yaml
    python run.py configs/<other>.yaml
On Slurm, use slurm_vllm_local.sbatch (see docs/slurm.md).
"""
import sys

from agentflow.pipeline import Pipeline


def main() -> int:
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/vllm_local_caption.yaml"
    pipeline = Pipeline(config, prompt_dir="prompts")
    failed = [item_id for item_id in pipeline.item_ids if not pipeline.execute(item_id)]
    print(f"done: {len(pipeline.item_ids) - len(failed)} ok, {len(failed)} failed {failed or ''}")
    return 1 if failed else 0  # nonzero so Slurm marks the job FAILED


# Required: vLLM starts its engine with `spawn`, which re-imports this file in a child process.
if __name__ == "__main__":
    sys.exit(main())
