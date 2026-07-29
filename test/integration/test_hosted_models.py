"""Integration tests for hosted model backends (OpenAI/Azure and Gemini).

These caption the ``examples/captioning`` images through real hosted APIs, so
they need API keys in ``.env`` (see ``.env.example``) rather than a local server.

Run from the repo root:
    pytest test/integration/test_hosted_models.py -v
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

from test.integration.harness import make_runner, output_path  # noqa: E402

import pytest  # noqa: E402

run_pipeline = make_runner("captioning")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAICaption:
    def test_runs_without_error(self):
        run_pipeline("openai_caption.yaml")

    def test_output_files_created(self):
        run_pipeline("openai_caption.yaml")
        out = output_path("test_openai_caption") / "SampleOutput"
        assert out.exists()
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid(self):
        run_pipeline("openai_caption.yaml")
        out = output_path("test_openai_caption") / "SampleOutput"
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
        out = output_path("test_gemini_caption") / "SampleOutput"
        assert out.exists()
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid(self):
        run_pipeline("gemini_caption.yaml")
        out = output_path("test_gemini_caption") / "SampleOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "text" in data
            assert isinstance(data["text"], str) and data["text"].strip()


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestGeminiDirect:
    def test_gemini_direct_generation(self):
        from agentflow.models import GeminiVLM

        token = os.getenv("GEMINI_API_KEY")
        model = GeminiVLM(base_url="", token=token, model_id="gemini-2.5-flash")
        res = model.generate(
            system_prompt="You are a helpful assistant.",
            image_path=None,
            examples=[],
            input_text="Say hello!",
        )
        assert res is not None
        assert isinstance(res, str)
        assert len(res.strip()) > 0
