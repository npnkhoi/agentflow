"""Integration tests for the ``examples/captioning`` project (local vision server).

Exercises single-stage, two-stage, two-input, and demo-augmented pipelines that
caption images. Assumes an OpenAI-compatible vision server at
http://0.0.0.0:8000. Start one with llama.cpp (SmolVLM-500M, 8K context):

    llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --host 0.0.0.0 --port 8000 -c 8192

Run from the repo root:
    pytest test/integration/test_captioning.py -v
"""

import json

from test.integration.harness import make_runner, output_path, project_prompt_dir

run_pipeline = make_runner("captioning")
PROMPT_DIR = project_prompt_dir("captioning")


class TestCaption:
    """Single-stage pipeline: Image → SampleOutput."""

    def test_runs_without_error(self):
        run_pipeline("caption.yaml")

    def test_output_files_created(self):
        run_pipeline("caption.yaml")
        out = output_path("test_caption") / "SampleOutput"
        assert out.exists(), "SampleOutput stage dir not found"
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid_sample_output(self):
        run_pipeline("caption.yaml")
        out = output_path("test_caption") / "SampleOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "text" in data, f"SampleOutput missing 'text' in {json_file}"
            assert isinstance(data["text"], str) and data["text"].strip()

    def test_log_files_created(self):
        run_pipeline("caption.yaml")
        out = output_path("test_caption") / "SampleOutput"
        logs = list(out.glob("*/run.log"))
        assert len(logs) == 2, f"Expected 2 log files, got {len(logs)}"

    def test_second_run_uses_cache(self):
        run_pipeline("caption.yaml")
        out = output_path("test_caption") / "SampleOutput"
        mtimes_first = {f: f.stat().st_mtime for f in out.glob("*/output.json")}

        run_pipeline("caption.yaml")
        mtimes_second = {f: f.stat().st_mtime for f in out.glob("*/output.json")}

        assert mtimes_first == mtimes_second, (
            "Output files were re-written on second run (cache not used)"
        )


class TestTwoStage:
    """Two-stage pipeline: Image → SampleOutput → RefinedOutput."""

    def test_runs_without_error(self):
        run_pipeline("two_stage.yaml")

    def test_both_stage_dirs_created(self):
        run_pipeline("two_stage.yaml")
        out = output_path("test_two_stage")
        stage_dirs = [d.name for d in out.iterdir() if d.is_dir()]
        assert "SampleOutput" in stage_dirs
        assert "RefinedOutput" in stage_dirs

    def test_second_stage_reads_first_stage_output(self):
        """Verify stage 2 output exists in its own dir — it can only exist if stage 1 succeeded."""
        run_pipeline("two_stage.yaml")
        out = output_path("test_two_stage") / "RefinedOutput"
        items = list(out.glob("*/output.json"))
        assert len(items) == 2


class TestWithDemos:
    """Single-stage pipeline with 1-shot random demo."""

    def test_runs_without_error(self):
        run_pipeline("with_demos.yaml")

    def test_output_created(self):
        run_pipeline("with_demos.yaml")
        out = output_path("test_with_demos") / "SampleOutput"
        items = list(out.glob("*/output.json"))
        assert len(items) == 2

    def test_demos_json_created(self):
        run_pipeline("with_demos.yaml")
        out = output_path("test_with_demos") / "SampleOutput"
        demos_files = list(out.glob("**/demos.json"))
        assert len(demos_files) == 2, (
            f"Expected demos.json per item, found {len(demos_files)}"
        )

    def test_demos_json_contains_one_id(self):
        run_pipeline("with_demos.yaml")
        out = output_path("test_with_demos") / "SampleOutput"
        for demos_file in out.glob("**/demos.json"):
            ids = json.loads(demos_file.read_text())
            assert isinstance(ids, list), "demos.json should be a list"
            assert len(ids) == 1, f"Expected 1 demo id (shots=1), got {len(ids)}"
            assert all(isinstance(i, str) for i in ids), "Demo ids should be strings"

    def test_demo_id_is_different_from_item(self):
        """The selected demo must not be the item itself."""
        run_pipeline("with_demos.yaml")
        out = output_path("test_with_demos") / "SampleOutput"
        for item_dir in out.iterdir():
            if not item_dir.is_dir():
                continue
            demos_file = item_dir / "demos.json"
            if not demos_file.exists():
                continue
            item_id = item_dir.name
            demo_ids = json.loads(demos_file.read_text())
            assert item_id not in demo_ids, (
                f"Item '{item_id}' selected itself as a demo"
            )


class TestTwoInput:
    """Two-stage pipeline where stage 2 takes both Image and SampleOutput as inputs → RefinedOutput."""

    def test_runs_without_error(self):
        run_pipeline("two_input.yaml")

    def test_both_stage_dirs_created(self):
        run_pipeline("two_input.yaml")
        out = output_path("test_two_input")
        stage_dirs = [d.name for d in out.iterdir() if d.is_dir()]
        assert "SampleOutput" in stage_dirs
        assert "RefinedOutput" in stage_dirs

    def test_refined_output_files_created(self):
        run_pipeline("two_input.yaml")
        out = output_path("test_two_input") / "RefinedOutput"
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_refined_output_is_valid(self):
        run_pipeline("two_input.yaml")
        out = output_path("test_two_input") / "RefinedOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "text" in data, f"Missing 'text' in {json_file}"
            assert isinstance(data["text"], str) and data["text"].strip()

    def test_prompt_resolves_two_input_names(self):
        """Stage 2 prompt file name encodes both inputs: RefinedOutput__Image_SampleOutput.md."""
        prompt_file = PROMPT_DIR / "RefinedOutput__Image_SampleOutput.md"
        assert prompt_file.exists(), "Two-input prompt file not found"
