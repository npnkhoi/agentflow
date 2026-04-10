"""
Integration tests for Pipeline.
Assumes google/gemma-3-27b-it is running at http://0.0.0.0:8000 (OpenAI-compatible).
Run from the repo root: pytest agentflow/test/test_integration.py -v
"""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from agentflow.typing.config import Config
from agentflow.pipeline import Pipeline

TEST_DIR = Path(__file__).parent
PROMPT_DIR = TEST_DIR / "prompts"
OUTPUT_DIR = Path("output")
CONFIGS_DIR = TEST_DIR / "configs"


def load_config(name: str) -> Config:
    """Load a YAML config from agentflow/test/configs/, resolving relative paths to absolute."""
    raw = yaml.safe_load((CONFIGS_DIR / name).read_text())
    # Resolve relative paths in loader source relative to repo root
    repo_root = TEST_DIR.parent.parent
    loader = raw["loader"]
    loader["source"] = str(repo_root / loader["source"])
    loader["kwargs"]["image_dir"] = str(repo_root / loader["kwargs"]["image_dir"])
    # Resolve demo_pools source/image_dir if present
    for pool in raw.get("demo_pools", {}).values():
        if pool.get("source"):
            pool["source"] = str(repo_root / pool["source"])
        if pool.get("image_dir"):
            pool["image_dir"] = str(repo_root / pool["image_dir"])
    return Config.model_validate(raw)


def run_pipeline(config_name: str) -> Pipeline:
    cfg = load_config(config_name)
    # Redirect output to test output dir
    cfg.name = cfg.name  # keep original name; output goes to output/<name>/
    net = Pipeline(cfg, prompt_dir=str(PROMPT_DIR))
    net.execute_all()
    return net


@pytest.fixture(autouse=True)
def clean_output():
    """Remove test output before each test, restore nothing after."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield
    # Leave output in place for post-test inspection if needed


def output_path(config_name: str) -> Path:
    return OUTPUT_DIR / config_name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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

        assert mtimes_first == mtimes_second, "Output files were re-written on second run (cache not used)"


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

    def test_output_name_collision_raises(self):
        """A stage whose output name matches a model-sourced input must fail at config load."""
        import yaml
        from agentflow.typing.config import Config
        bad_config = {
            "name": "bad",
            "wandb_enabled": False,
            "loader": {
                "source": "agentflow/test/data/test_data.json",
                "kwargs": {"image_dir": "agentflow/test/data/images"},
            },
            "models": {},
            "stages": [
                {"inputs": [["Image", "human"]], "output": "SampleOutput", "processor": "LLMProcessor", "model": "m"},
                {"inputs": [["SampleOutput", "model"]], "output": "SampleOutput", "processor": "LLMProcessor", "model": "m"},
            ],
        }
        import pytest
        with pytest.raises(Exception, match="collides"):
            Config.model_validate(bad_config)


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
        assert len(demos_files) == 2, f"Expected demos.json per item, found {len(demos_files)}"

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
            assert item_id not in demo_ids, f"Item '{item_id}' selected itself as a demo"


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


class TestCountWithDemos:
    """Counting task using CountBench images with a 3-shot demo pool.

    Dataset: 2 test items (count_0, count_1).
    Demo pool: all 5 items (count_0..4); count_2..4 have pre-computed CountOutput.
    Self-exclusion ensures each item does not appear in its own demo list.
    """

    def test_runs_without_error(self):
        run_pipeline("count_with_demos.yaml")

    def test_two_output_files_created(self):
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid_count(self):
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "count" in data, f"CountOutput missing 'count' in {json_file}"
            assert isinstance(data["count"], int), f"'count' must be an int, got {type(data['count'])}"
            assert data["count"] > 0, f"'count' must be positive, got {data['count']}"

    def test_demos_json_lists_three_ids(self):
        """Each test item must have exactly 3 demo IDs (shots=3)."""
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        demos_files = list(out.glob("*/demos.json"))
        assert len(demos_files) == 2, f"Expected demos.json per item, found {len(demos_files)}"
        for demos_file in demos_files:
            ids = json.loads(demos_file.read_text())
            assert len(ids) == 3, f"Expected 3 demo ids (shots=3), got {len(ids)}: {ids}"

    def test_demos_exclude_self(self):
        """Self-exclusion: each item must not appear in its own demo list."""
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        for item_dir in out.iterdir():
            if not item_dir.is_dir():
                continue
            demos_file = item_dir / "demos.json"
            if not demos_file.exists():
                continue
            item_id = item_dir.name
            demo_ids = json.loads(demos_file.read_text())
            assert item_id not in demo_ids, f"Item '{item_id}' selected itself as a demo"

    def test_demos_are_from_pool(self):
        """All selected demo IDs must come from the 5-item pool (count_0..4)."""
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        pool_ids = {"count_0", "count_1", "count_2", "count_3", "count_4"}
        for demos_file in out.glob("*/demos.json"):
            ids = json.loads(demos_file.read_text())
            for demo_id in ids:
                assert demo_id in pool_ids, (
                    f"Demo ID '{demo_id}' is not from the pool {pool_ids}"
                )
