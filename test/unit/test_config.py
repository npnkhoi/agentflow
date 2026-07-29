"""Unit tests for config validation (no LLM server required)."""

import pytest

from agentflow.typing.config import Config


class TestOutputNameCollision:
    def test_output_name_collision_raises(self):
        """A stage whose output name matches a model-sourced input must fail at config load."""
        bad_config = {
            "name": "bad",
            "wandb_enabled": False,
            "loader": {
                "source": "examples/captioning/data/items.json",
                "kwargs": {"image_dir": "examples/captioning/data/images"},
            },
            "models": {},
            "stages": [
                {
                    "inputs": [["Image", "human"]],
                    "output": "SampleOutput",
                    "processor": "LLMProcessor",
                    "model": "m",
                },
                {
                    "inputs": [["SampleOutput", "model"]],
                    "output": "SampleOutput",
                    "processor": "LLMProcessor",
                    "model": "m",
                },
            ],
        }

        with pytest.raises(Exception, match="collides"):
            Config.model_validate(bad_config)
