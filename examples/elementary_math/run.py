"""Run the elementary-math pipeline: figure + question -> calculations ->
calculator -> answer -> verdict.

Registers this example's output types and its `CalculatorProcessor`, then
executes the pipeline over `data/items.json` and prints per-item verdicts plus
the overall accuracy.

Config paths (`loader.source`, `image_dir`) are written relative to this
project directory and resolved here, so the script works from any cwd. Pipeline
output always lands in `./output/<pipeline-name>/` relative to the cwd.

Needs an OpenAI-compatible vision server matching `configs/math.yaml`:

    vllm serve Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8010 \
      --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85

Usage (as a module, from the repo root):
    python -m examples.elementary_math.run
    python -m examples.elementary_math.run --config math.yaml
"""

import argparse
import tempfile
from pathlib import Path

import yaml

from agentflow.pipeline import Pipeline
from examples.elementary_math.extensions import register

PROJECT_DIR = Path(__file__).resolve().parent


def resolved_config_path(config_name: str) -> str:
    """Rewrite the config's project-relative data paths to absolute ones and
    write the result to a temp file, whose path is returned."""
    raw = yaml.safe_load((PROJECT_DIR / "configs" / config_name).read_text(encoding="utf-8"))
    loader = raw["loader"]
    loader["source"] = str((PROJECT_DIR / loader["source"]).resolve())
    loader["kwargs"]["image_dir"] = str((PROJECT_DIR / loader["kwargs"]["image_dir"]).resolve())

    fd, path = tempfile.mkstemp(prefix="elementary_math_", suffix=f"_{config_name}")
    with open(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="math.yaml", help="config file name inside configs/ (default: math.yaml)")
    args = ap.parse_args()

    register()
    pipeline = Pipeline(resolved_config_path(args.config), prompt_dir=str(PROJECT_DIR / "prompts"))
    pipeline.execute_all()

    verdicts = pipeline.cache("Verdict")
    answers = pipeline.cache("Answer")
    if verdicts is None:
        return

    print()
    correct = 0
    for item_id in pipeline.item_ids:
        verdict = verdicts.load(item_id)
        answer = answers.load(item_id) if answers is not None else None
        if verdict is None:
            print(f"{item_id:>16}: no verdict (a stage failed; see output/*/{item_id}/run.log)")
            continue
        correct += verdict.correct
        mark = "PASS" if verdict.correct else "FAIL"
        print(f"{item_id:>16}: {mark}  answer={answer.value if answer else '?'}  {verdict.reason}")

    total = len(pipeline.item_ids)
    print(f"\naccuracy: {correct}/{total} = {correct / total:.1%}" if total else "\nno items")


if __name__ == "__main__":
    main()
