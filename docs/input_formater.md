# InputFormater

`InputFormater` converts the `inputs` dict into the text string sent as the user message to the model.

## Default behavior

When no `input_format` is set (the common case), each non-image input is serialized as a labeled block:

```
<input_name>:
<value>
```

Pydantic models are serialized with `model_dump_json(indent=2)`. Image inputs are omitted from the text (they are passed separately as the `image_path` argument to the model).

## Custom formats (registry)

Domain-specific formatting logic can be registered at application startup:

```python
from agentflow.input_formater import InputFormater

def my_handler(inputs: dict, input_names_snake: list[str]) -> str:
    score = inputs["score"]
    comment = inputs["comment"]
    return f"Score: {score}/5\nComment: {comment}"

InputFormater.register("review_summary", my_handler)
```

Then reference it in the stage config:

```yaml
stages:
  - inputs:
      - [Score, human]
      - [Comment, human]
    output: ReviewSummary
    processor: LLMProcessor
    model: my_model
    input_format: review_summary
```

If `input_format` is set but not registered, `InputFormater` raises `NotImplementedError` immediately — fail-fast at pipeline construction rather than at runtime.

## Registration in a domain package

Register all custom formats once, at import time, by calling `InputFormater.register()` in your package's `__init__.py` or a dedicated module:

```python
# mypackage/input_formater.py
from agentflow.input_formater import InputFormater

InputFormater.register("my_format", _my_handler)
InputFormater.register("other_format", _other_handler)
```

```python
# mypackage/__init__.py
import mypackage.input_formater  # triggers registration
```

## Handler signature

```python
def handler(inputs: dict, input_names_snake: list[str]) -> str:
    ...
```

- `inputs` — the full inputs dict for the current item (same dict passed to `Processor.__call__`)
- `input_names_snake` — list of snake_case input type names declared for this stage
- return value — the user message text sent to the model
