"""Check configurable OpenAI generation options without contacting a model server.

Usage:
    python -m pytest test/unit/test_generation_kwargs.py -q
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from pydantic import BaseModel

from agentflow.models import OpenAILLM
from agentflow.pipeline import Pipeline
from agentflow.typing.config import ModelConfig


class Output(BaseModel):
    text: str


def backend(options=None):
    model = OpenAILLM("http://unused/v1", "-", "gemma", generation_kwargs=options)
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"text":"ok"}'))]))
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model._ready = True
    return model, create


def test_defaults_do_not_set_temperature_or_thinking():
    model, create = backend()
    model.generate("system", None, [], "input")
    assert set(create.call_args.kwargs) == {"messages", "model", "max_completion_tokens"}
    assert create.call_args.kwargs["max_completion_tokens"] == 4096


def test_temperature_zero_and_thinking_false_are_forwarded_with_schema():
    options = {"temperature": 0, "max_completion_tokens": 2500,
               "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    model, create = backend(options)
    assert model.generate("system", None, [], "input", output_type=Output) == '{"text":"ok"}'
    request = create.call_args.kwargs
    assert request["temperature"] == 0
    assert request["max_completion_tokens"] == 2500
    assert request["extra_body"] == options["extra_body"]
    assert request["response_format"]["json_schema"]["schema"] == Output.model_json_schema()
    request["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
    model.generate("system", None, [], "input")
    assert create.call_args.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert options["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("key", ["model", "messages", "response_format"])
def test_pipeline_managed_request_fields_cannot_be_overridden(key):
    with pytest.raises(ValueError, match="pipeline-managed"):
        backend({key: "override"})


def test_generation_options_default_is_not_shared():
    first = ModelConfig(base_url="unused", model_id="one")
    second = ModelConfig(base_url="unused", model_id="two")
    first.generation_kwargs["temperature"] = 0
    assert second.generation_kwargs == {}


@pytest.mark.parametrize("configured", [False, True])
def test_pipeline_passes_options_only_when_configured(tmp_path, monkeypatch, configured):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "items.json").write_text(json.dumps([{"id": "one", "data": {"image": "unused"}}]))
    calls = []

    class LegacyBackend:
        def __init__(self, base_url, token, model_id):
            calls.append({"base_url": base_url, "token": token, "model_id": model_id})

    Pipeline.register_model_backend("legacy_generation_test", LegacyBackend)
    model = {"cls": "openai" if configured else "legacy_generation_test", "base_url": "http://unused/v1",
             "token": "-", "model_id": "gemma"}
    options = {"temperature": 0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if configured:
        model["generation_kwargs"] = options
    cfg = {"name": "test", "loader": {"source": "items.json", "kwargs": {"image_dir": "."}},
           "models": {"model": model}, "stages": []}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    pipeline = Pipeline(path)
    if configured:
        assert pipeline.get_model("model")._generation_kwargs == options
    else:
        assert calls == [{"base_url": "http://unused/v1", "token": "-", "model_id": "gemma"}]
