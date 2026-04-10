# Extension points

Every user-facing extension can be registered programmatically. `agentflow/` is library code — no edits required.

---

## Output types

Register Pydantic models by name before building the pipeline.

```python
from agentflow.pipeline import Pipeline
from mypackage.types import MyOutput

Pipeline.register_type("MyOutput", MyOutput)
```

See [output_types.md](output_types.md) for full details.

---

## Input formats

Register named formatters that control how inputs are serialized into the user message.

```python
from agentflow.input_formater import InputFormater

def my_handler(inputs: dict, input_names_snake: list[str]) -> str:
    return f"Score: {inputs['score']}\nComment: {inputs['comment']}"

InputFormater.register("my_format", my_handler)
```

Then set `input_format: my_format` on any stage. See [input_formater.md](input_formater.md).

---

## Model backends

Register custom LLM/VLM client classes for new providers or deployment types.

```python
from agentflow.pipeline import Pipeline
from agentflow.models import BaseLLM

class MyLLM(BaseLLM):
    def _do_init(self): ...
    def generate(self, system_prompt, image_path, examples, input_text,
                 output_type=None, logger=None): ...

Pipeline.register_model_backend("my_provider", MyLLM)
```

Then use `cls: my_provider` in the config:

```yaml
models:
  my_model:
    cls: my_provider
    base_url: "..."
    token: "..."
    model_id: "..."
```

Built-in backends: `openai`, `azure`, `gemini`.

---

## Processors

Register processor classes by name before building the pipeline.

```python
from agentflow.pipeline import Pipeline
from mypackage.processors import MyProcessor

Pipeline.register_processor("MyProcessor", MyProcessor)
```

Then use the short name in config:

```yaml
stages:
  - processor: MyProcessor
    ...
```

Built-in processors (`LLMProcessor`, `CopyProcessor`) are pre-registered in `agentflow/__init__.py`.

---

## Summary

| Extension point | Registration API |
|---|---|
| Output type | `Pipeline.register_type(name, cls)` |
| Input format | `agentflow.input_formater.InputFormater.register(name, fn)` |
| Model backend | `agentflow.pipeline.Pipeline.register_model_backend(name, cls)` |
| Processor | `Pipeline.register_processor(name, cls)` |

> **Loader:** `DataItemLoader` is the only built-in loader. It accepts any `data` dict structure, so non-standard sources are best handled by a preprocessing step that converts them to the standard JSON format (`[{"id": str, "data": {...}}, ...]`) rather than by swapping the loader.

> **Demo pools:** Demo sources are specified directly via `source` + `image_dir` in the stage config. The pool may overlap with the test set — self-exclusion is always applied.
