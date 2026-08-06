"""Unit tests for ModelConfig env-var expansion and wandb project selection.

Usage:
    pytest test/unit/test_model_config.py
"""

from agentflow.typing.config import Config, ModelConfig

BASE_CONFIG = {
    "name": "pipe",
    "wandb_enabled": False,
    "loader": {
        "source": "examples/captioning/data/items.json",
        "kwargs": {"image_dir": "examples/captioning/data/images"},
    },
    "stages": [
        {
            "inputs": [["Image", "human"]],
            "output": "SampleOutput",
            "processor": "LLMProcessor",
            "model": "m",
        }
    ],
}


class TestEnvExpansion:
    def test_base_url_and_token_expand(self, monkeypatch):
        monkeypatch.setenv("MY_VLM_URL", "http://10.0.0.1:8000/v1")
        monkeypatch.setenv("MY_KEY", "sk-secret")

        cfg = ModelConfig(
            cls="openai",
            base_url="${MY_VLM_URL}",
            token="${MY_KEY}",
            model_id="some-model",
        )

        assert cfg.base_url == "http://10.0.0.1:8000/v1"
        assert cfg.token == "sk-secret"

    def test_unset_variable_expands_to_empty(self, monkeypatch):
        monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
        cfg = ModelConfig(base_url="${DEFINITELY_UNSET_VAR}", token="-", model_id="m")
        assert cfg.base_url == ""

    def test_literal_url_is_untouched(self):
        cfg = ModelConfig(base_url="http://0.0.0.0:8000/v1", token="-", model_id="m")
        assert cfg.base_url == "http://0.0.0.0:8000/v1"


class TestWandbProject:
    def test_defaults_to_none_so_caller_falls_back_to_name(self):
        cfg = Config.model_validate(BASE_CONFIG)
        assert cfg.wandb_project is None
        assert (cfg.wandb_project or cfg.name) == "pipe"

    def test_explicit_project_is_kept(self):
        cfg = Config.model_validate({**BASE_CONFIG, "wandb_project": "argraph"})
        assert (cfg.wandb_project or cfg.name) == "argraph"
