# Output types

Every stage declares an `output` type name. This name serves three purposes:

1. **Pydantic model** — the output is validated and serialized as this type.
2. **Cache directory** — outputs are stored under `output/<pipeline>/<TypeName>/`.
3. **Prompt file component** — the type name appears in the prompt filename.

## Built-in types

| Type | Fields | Use |
|---|---|---|
| `SampleOutput` | `text: str` | Generic single-text output |
| `RefinedOutput` | `text: str` | Refined/rewritten text |

## Registering custom output types

Define a Pydantic model and register it under a logical name before constructing `Pipeline`:

```python
from pydantic import BaseModel
from agentflow.pipeline import Pipeline

class ReviewResult(BaseModel):
    score: int
    summary: str

Pipeline.register_type("ReviewResult", ReviewResult)
```

Then reference the name in the config:

```yaml
stages:
  - output: ReviewResult
    ...
```

`register_type` must be called before `Pipeline(cfg)`. The typical place is in your domain package's `__init__.py` so it runs at import time.

## Registering the same schema under multiple names

The `name` argument is the logical stage name — it controls the config key and the cache directory. The same Pydantic class can be registered under multiple names to produce independently cached outputs that share a schema:

```python
from pydantic import BaseModel
from agentflow.pipeline import Pipeline

class TextOutput(BaseModel):
    text: str

Pipeline.register_type("Caption", TextOutput)
Pipeline.register_type("RefinedCaption", TextOutput)
```

```yaml
stages:
  - output: Caption        # stored in output/<pipeline>/Caption/
    ...
  - output: RefinedCaption # stored in output/<pipeline>/RefinedCaption/
    ...
```

Each name gets its own independent cache directory even though the underlying schema is identical.

## Pattern for domain packages

```python
# mypackage/__init__.py
from agentflow.pipeline import Pipeline
from mypackage.types import ReviewResult, FinalReport

Pipeline.register_type("ReviewResult", ReviewResult)
Pipeline.register_type("FinalReport", FinalReport)
```

`agentflow/` itself never needs to be modified.
