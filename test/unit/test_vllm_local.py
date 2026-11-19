"""Check the in-process vLLM backend against a fake ``vllm`` module (no GPU or vLLM install needed).

Usage:
    python -m pytest test/unit/test_vllm_local.py -q
"""
import json
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel

import agentflow  # noqa: F401  (registers built-in backends)
from agentflow.models import VLLMLocalLLM
from agentflow.pipeline import Pipeline
from agentflow.typing.config import ModelConfig


class Output(BaseModel):
    text: str


@dataclass
class FakeStructuredOutputsParams:
    json: dict | None = None


class FakeSamplingParams:
    def __init__(self, max_tokens=16, **kwargs):
        self.max_tokens = max_tokens
        self.kwargs = kwargs


class FakeLLM:
    instances: list["FakeLLM"] = []

    def __init__(self, model, **engine_args):
        self.model = model
        self.engine_args = engine_args
        self.calls = []
        self.finish_reason = "stop"
        FakeLLM.instances.append(self)

    def chat(self, messages, sampling_params, use_tqdm=True, chat_template_kwargs=None):
        self.calls.append({"messages": messages, "params": sampling_params,
                           "use_tqdm": use_tqdm, "chat_template_kwargs": chat_template_kwargs})
        return [SimpleNamespace(outputs=[SimpleNamespace(text='{"text":"ok"}', finish_reason=self.finish_reason)])]


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeLLM.instances = []
    monkeypatch.setattr(VLLMLocalLLM, "_engines", {})
    vllm = ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams
    sampling = ModuleType("vllm.sampling_params")
    sampling.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    return vllm


def test_engine_is_lazy_and_loaded_once(fake_vllm):
    model = VLLMLocalLLM("", "", "Qwen/Qwen2.5-VL-3B-Instruct", engine_args={"tensor_parallel_size": 2})
    assert FakeLLM.instances == []
    model.generate("system", None, [], "a")
    model.generate("system", None, [], "b")
    assert len(FakeLLM.instances) == 1
    engine = FakeLLM.instances[0]
    assert engine.model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert engine.engine_args == {"tensor_parallel_size": 2}


def test_engine_shared_across_instances_with_same_engine_args(fake_vllm):
    VLLMLocalLLM("", "", "m", generation_kwargs={"temperature": 0}).generate("sys", None, [], "in")
    VLLMLocalLLM("", "", "m", generation_kwargs={"temperature": 1}).generate("sys", None, [], "in")
    assert len(FakeLLM.instances) == 1
    VLLMLocalLLM("", "", "m", engine_args={"max_model_len": 4096}).generate("sys", None, [], "in")
    VLLMLocalLLM("", "", "other").generate("sys", None, [], "in")
    assert len(FakeLLM.instances) == 3


def test_messages_match_openai_format_with_demo_images(fake_vllm, tmp_path):
    demo, query = tmp_path / "demo.png", tmp_path / "query.png"
    demo.write_bytes(b"demo")
    query.write_bytes(b"query")
    model = VLLMLocalLLM("", "", "m")
    model.generate("sys", query, [(demo, "demo in", "demo out")], "question")
    call = FakeLLM.instances[0].calls[0]
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert call["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert call["messages"][-1]["content"][-1] == {"type": "text", "text": "question"}
    assert call["use_tqdm"] is False


def test_structured_output_uses_json_schema(fake_vllm):
    model = VLLMLocalLLM("", "", "m")
    assert model.generate("sys", None, [], "in", output_type=Output) == '{"text":"ok"}'
    params = FakeLLM.instances[0].calls[0]["params"]
    assert params.kwargs["structured_outputs"].json == Output.model_json_schema()
    model.generate("sys", None, [], "in")
    assert "structured_outputs" not in FakeLLM.instances[0].calls[1]["params"].kwargs


def test_sampling_defaults_and_overrides(fake_vllm):
    model = VLLMLocalLLM("", "", "m")
    model.generate("sys", None, [], "in")
    assert FakeLLM.instances[0].calls[0]["params"].max_tokens == 4096  # not vLLM's default of 16

    options = {"temperature": 0, "max_tokens": 1000, "chat_template_kwargs": {"enable_thinking": False}}
    model = VLLMLocalLLM("", "", "m", generation_kwargs=options)
    model.generate("sys", None, [], "in")
    calls = FakeLLM.instances[0].calls  # same model_id/engine_args -> same shared engine
    assert calls[1]["params"].max_tokens == 1000
    assert calls[1]["params"].kwargs == {"temperature": 0}
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": False}
    model.generate("sys", None, [], "in")  # popping chat_template_kwargs must not mutate stored options
    assert calls[2]["chat_template_kwargs"] == {"enable_thinking": False}


def test_truncation_is_logged(fake_vllm, tmp_path):
    model = VLLMLocalLLM("", "", "m")
    model._lazy_init()
    FakeLLM.instances[0].finish_reason = "length"
    log = tmp_path / "run.log"
    with log.open("w") as f:
        model.generate("sys", None, [], "in", logger=f)
    assert "truncated" in log.read_text()


@pytest.mark.parametrize("key", ["structured_outputs", "guided_decoding"])
def test_pipeline_managed_sampling_fields_cannot_be_overridden(key):
    with pytest.raises(ValueError, match="pipeline-managed"):
        VLLMLocalLLM("", "", "m", generation_kwargs={key: "x"})


def test_engine_args_cannot_set_model():
    with pytest.raises(ValueError, match="model_id"):
        VLLMLocalLLM("", "", "m", engine_args={"model": "other"})


def test_missing_vllm_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ImportError, match="same Python environment"):
        VLLMLocalLLM("", "", "m").generate("sys", None, [], "in")


def test_config_base_url_optional_except_for_openai():
    assert ModelConfig(cls="vllm_local", model_id="m").base_url == ""
    with pytest.raises(ValueError, match="requires base_url"):
        ModelConfig(cls="openai", model_id="m")


def test_pipeline_builds_vllm_local_from_yaml(tmp_path, monkeypatch, fake_vllm):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "items.json").write_text(json.dumps([{"id": "one", "data": {"image": "unused"}}]))
    cfg = {"name": "test", "loader": {"source": "items.json", "kwargs": {"image_dir": "."}},
           "models": {"qwen": {"cls": "vllm_local", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                               "engine_args": {"max_model_len": 8192},
                               "generation_kwargs": {"temperature": 0}}},
           "stages": []}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    model = Pipeline(path).get_model("qwen")
    assert isinstance(model, VLLMLocalLLM)
    assert model._engine_args == {"max_model_len": 8192}
    assert model._generation_kwargs == {"temperature": 0}
    assert FakeLLM.instances == []  # building the pipeline must not load the model
