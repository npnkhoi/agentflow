"""Integration test for the in-process vLLM backend (``cls: vllm_local``).

Runs ``examples/captioning/configs/vllm_local_caption.yaml``, which loads
Qwen2.5-VL-3B-Instruct inside the test process -- no ``vllm serve`` needed.
Skipped unless vLLM is importable in this Python env and a CUDA GPU is visible.

Environment variables:
    AGENTFLOW_REQUIRE_VLLM=1      fail instead of skipping when vLLM/GPU is missing
    VLLM_LOCAL_ENGINE_ARGS='{..}' JSON merged into the model's engine_args, e.g.
                                  '{"attention_backend": "TRITON_ATTN", "dtype": "half"}'
                                  on pre-Ampere GPUs

Run from the repo root (e.g. inside a Slurm GPU job; see
experimental/e1119_test_vllm_local.sbatch):
    pytest test/integration/test_vllm_local.py -v
"""

import importlib.util
import json
import os

import pytest

from test.integration.harness import make_runner, output_path

run_pipeline = make_runner("captioning")
REQUIRED = os.environ.get("AGENTFLOW_REQUIRE_VLLM") == "1"


def _unavailable(reason: str):
    if REQUIRED:
        pytest.fail(f"AGENTFLOW_REQUIRE_VLLM=1 but {reason}")
    pytest.skip(reason)


def _apply_engine_overrides(raw: dict) -> None:
    overrides = json.loads(os.environ.get("VLLM_LOCAL_ENGINE_ARGS") or "{}")
    for model in raw["models"].values():
        model.setdefault("engine_args", {}).update(overrides)


def test_caption_with_in_process_vllm():
    if importlib.util.find_spec("vllm") is None:
        _unavailable("vllm is not installed in this Python env")
    import torch
    if not torch.cuda.is_available():
        _unavailable("no CUDA GPU visible")

    run_pipeline("vllm_local_caption.yaml", _apply_engine_overrides)
    out = output_path("test_vllm_local_caption") / "SampleOutput"
    files = list(out.glob("*/output.json"))
    assert len(files) == 2, f"Expected 2 output files, got {len(files)}"
    for f in files:
        text = json.loads(f.read_text())["text"]
        assert isinstance(text, str) and text.strip()

    # Second pipeline in the same process: everything is cached, so nothing is rewritten.
    mtimes = {f: f.stat().st_mtime for f in files}
    run_pipeline("vllm_local_caption.yaml", _apply_engine_overrides)
    assert {f: f.stat().st_mtime for f in files} == mtimes
