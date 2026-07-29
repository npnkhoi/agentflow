"""Shared helpers for integration tests that run real pipelines over the example
projects in ``examples/``.

Each example project (``examples/<name>/``) is self-contained:

    examples/<name>/
      configs/*.yaml   # pipeline configs, with data paths relative to the project dir
      data/            # items.json, images/, optional demo pools
      prompts/         # <Output>__<Inputs>.md prompt templates

``make_runner(project)`` returns a ``run_pipeline(config_name)`` closure bound to
one project; it resolves the config's project-relative data paths to absolute
paths (so tests can run from the repo root) and executes the pipeline.

Usage (inside a test module):
    from test.integration.harness import make_runner, output_path
    run_pipeline = make_runner("captioning")
    ...
    run_pipeline("caption.yaml")
"""

import tempfile
from pathlib import Path
from typing import Callable

import yaml

from agentflow.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
# Pipeline writes to ``output/<name>`` relative to the cwd (the repo root when
# running pytest from there); keep this in sync with that convention.
OUTPUT_DIR = Path("output")


def resolved_config_path(project_dir: Path, config_name: str) -> str:
    """Load ``configs/<config_name>`` from ``project_dir``, resolve its
    project-relative data paths to absolute, and write it to a temp YAML file.
    Returns the temp file path."""
    raw = yaml.safe_load((project_dir / "configs" / config_name).read_text())

    loader = raw["loader"]
    loader["source"] = str((project_dir / loader["source"]).resolve())
    loader["kwargs"]["image_dir"] = str((project_dir / loader["kwargs"]["image_dir"]).resolve())

    for pool in raw.get("demo_pools", {}).values():
        if pool.get("source"):
            pool["source"] = str((project_dir / pool["source"]).resolve())
        if pool.get("image_dir"):
            pool["image_dir"] = str((project_dir / pool["image_dir"]).resolve())

    fd, path = tempfile.mkstemp(prefix="af_cfg_", suffix=f"_{config_name}")
    with open(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)
    return path


def make_runner(project: str) -> Callable[[str], Pipeline]:
    """Return a ``run_pipeline(config_name)`` bound to ``examples/<project>``."""
    project_dir = EXAMPLES_DIR / project
    prompt_dir = project_dir / "prompts"

    def run_pipeline(config_name: str) -> Pipeline:
        net = Pipeline(resolved_config_path(project_dir, config_name), prompt_dir=str(prompt_dir))
        net.execute_all()
        return net

    return run_pipeline


def project_prompt_dir(project: str) -> Path:
    return EXAMPLES_DIR / project / "prompts"


def output_path(config_name: str) -> Path:
    return OUTPUT_DIR / config_name
