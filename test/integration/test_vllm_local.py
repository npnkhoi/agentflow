"""Integration test for the in-process vLLM backend (``cls: vllm_local``).

Runs ``examples/captioning/configs/vllm_local_caption.yaml``, which loads
Qwen2.5-VL-3B-Instruct inside the test process -- no ``vllm serve`` needed.
Skipped unless vLLM is importable in this Python env and a CUDA GPU is visible.

Run from the repo root (e.g. inside a Slurm GPU job):
    pytest test/integration/test_vllm_local.py -v
"""

import importlib.util
import json

import pytest

from test.integration.harness import make_runner, output_path

pytestmark = pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vllm not installed")

run_pipeline = make_runner("captioning")


def test_caption_with_in_process_vllm():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU visible")

    run_pipeline("vllm_local_caption.yaml")
    out = output_path("test_vllm_local_caption") / "SampleOutput"
    files = list(out.glob("*/output.json"))
    assert len(files) == 2, f"Expected 2 output files, got {len(files)}"
    for f in files:
        text = json.loads(f.read_text())["text"]
        assert isinstance(text, str) and text.strip()

    # Second pipeline in the same process: everything is cached, and the
    # engine (if it were needed) would be reused rather than loaded twice.
    mtimes = {f: f.stat().st_mtime for f in files}
    run_pipeline("vllm_local_caption.yaml")
    assert {f: f.stat().st_mtime for f in files} == mtimes
