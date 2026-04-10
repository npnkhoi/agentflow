# Processors

A processor transforms a dict of inputs into a single Pydantic model output. Every processor inherits from `Processor` and implements `__call__`.

## Base class

```python
class Processor(ABC):
    def __init__(self, pipeline: Pipeline, stage_config: StageConfig): ...

    @abstractmethod
    def __call__(
        self,
        inputs: dict,
        logger=None,
        output_dir: Path | None = None,
    ) -> BaseModel | None: ...
```

`inputs` is a flat dict containing all declared input fields plus `"id"`. Keys are snake_case versions of the declared type names (e.g. `SampleOutput` → `sample_output`).

Returning `None` signals failure for that item; the pipeline will retry or abandon.

---

## Built-in processors

### LLMProcessor

The main workhorse. Calls a language/vision model with an auto-resolved prompt.

**Prompt resolution:** the prompt file is found at
`<prompt_dir>/<Output>__<Input1>_<Input2>.md` (double underscore separates
output from inputs; inputs joined by single underscore).

**Input formatting:** by default, each non-image text input is serialized to a
labeled block. Custom formats can be registered (see [input_formater.md](input_formater.md)).

**Few-shot demos:** if `demo` is configured, `DemoPool` selects demo items and
injects them as `(image, user_prompt, expected_output)` triples before the
actual query. Demo items that lack the expected output key are silently skipped.
The selected demo IDs are saved to `output_dir/demos.json` for traceability.

**Output parsing:** the model response is parsed with `OutputType.model_validate_json()`. If parsing fails, `None` is returned.

**Config fields used:** `model`, `demo`, `input_format`, `prompt_version`.

### CopyProcessor

Passes a single input field through unchanged. Useful for re-labeling or forwarding a previous stage's output to give it a new type name.

```yaml
- inputs:
    - [Caption, model]
  output: FinalCaption
  processor: CopyProcessor
```

---

## Writing a custom processor

```python
from agentflow.processors.base import Processor
from pydantic import BaseModel

class MyOutput(BaseModel):
    result: str

class MyProcessor(Processor):
    def __call__(self, inputs, logger=None, output_dir=None):
        text = inputs.get("sample_output")
        if text is None:
            return None
        return MyOutput(result=text.text.upper())
```

Register it before building the pipeline, then reference it by name in config:

```python
from agentflow.pipeline import Pipeline
Pipeline.register_processor("MyProcessor", MyProcessor)
```

```yaml
stages:
  - processor: MyProcessor
    ...
```
