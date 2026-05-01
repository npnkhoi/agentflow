"""
Integration tests for OpenAILLM and GeminiVLM via full pipeline config.
Requires API keys in .env (see .env.example).
Run: pytest test/test_models.py -v
"""
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

load_dotenv()

from agentflow.typing.config import Config
from agentflow.pipeline import Pipeline

TEST_DIR = Path(__file__).parent
PROMPT_DIR = TEST_DIR / "prompts"
OUTPUT_DIR = Path("output")
CONFIGS_DIR = TEST_DIR / "configs"


def load_config(name: str) -> Config:
    raw = yaml.safe_load((CONFIGS_DIR / name).read_text())
    repo_root = TEST_DIR.parent.parent
    loader = raw["loader"]
    loader["source"] = str(repo_root / loader["source"])
    loader["kwargs"]["image_dir"] = str(repo_root / loader["kwargs"]["image_dir"])
    return Config.model_validate(raw)


def run_pipeline(config_name: str) -> Pipeline:
    cfg = load_config(config_name)
    net = Pipeline(cfg, prompt_dir=str(PROMPT_DIR))
    net.execute_all()
    return net


@pytest.fixture(autouse=True)
def clean_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield


class TestOpenAICaption:
    def test_runs_without_error(self):
        run_pipeline("openai_caption.yaml")

    def test_output_files_created(self):
        run_pipeline("openai_caption.yaml")
        out = OUTPUT_DIR / "test_openai_caption" / "SampleOutput"
        assert out.exists()
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid(self):
        run_pipeline("openai_caption.yaml")
        out = OUTPUT_DIR / "test_openai_caption" / "SampleOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "text" in data
            assert isinstance(data["text"], str) and data["text"].strip()


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestGeminiCaption:
    def test_runs_without_error(self):
        run_pipeline("gemini_caption.yaml")

    def test_output_files_created(self):
        run_pipeline("gemini_caption.yaml")
        out = OUTPUT_DIR / "test_gemini_caption" / "SampleOutput"
        assert out.exists()
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid(self):
        run_pipeline("gemini_caption.yaml")
        out = OUTPUT_DIR / "test_gemini_caption" / "SampleOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "text" in data
            assert isinstance(data["text"], str) and data["text"].strip()
