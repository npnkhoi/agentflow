# AgentFlow

A lightweight framework for building batch LLM/VLM pipelines over structured datasets.

## What it does

Pipeline runs a configurable sequence of processing stages over a dataset. Each stage reads its inputs (either from the original dataset or from a previous stage's outputs), calls a processor, and writes results to disk. All outputs are cached — re-running skips completed items automatically.

## Quick start

```python
import yaml
from agentflow.pipeline import Pipeline
from agentflow.typing.config import Config

with open("pipeline.yaml") as f:
    cfg = Config.model_validate(yaml.safe_load(f))

net = Pipeline(cfg, prompt_dir="prompts/")
net.execute_all()
```

Or use the `Client` wrapper for production runs (parallel execution, wandb logging, include/exclude filtering):

```python
from agentflow.client import Client

client = Client(prompt_dir="prompts/")
with open("pipeline.yaml") as f:
    client.run(f)
```

## Config format

Pipelines are defined in YAML:

```yaml
name: my_pipeline
wandb_enabled: false
loader:
  source: data/items.json
  kwargs:
    image_dir: data/images/

models:
  gpt4o:
    cls: openai
    base_url: "https://api.openai.com/v1"
    token: "sk-..."
    model_id: gpt-4o

stages:
  - inputs:
      - [Image, human]
    output: Caption
    processor: LLMProcessor
    model: gpt4o
  - inputs:
      - [Caption, model]
    output: Summary
    processor: LLMProcessor
    model: gpt4o
```

See [docs/config.md](docs/config.md) for the full config reference.

## Output layout

```
output/<pipeline-name>/
  <StageName>/
    <item_id>/
      output.json   # stage output (Pydantic model serialized to JSON)
      run.log       # model call log for this item
      demos.json    # (if demos enabled) list of selected demo item_ids
```

## Data format

The loader expects a JSON file of the form:

```json
[
  {"id": "item_001", "data": {"image": "001.png", ...}},
  {"id": "item_002", "data": {"image": "002.png", ...}}
]
```

## Architecture

```
Config (YAML)
    │
    ▼
Pipeline
    ├── Loader  ──────── reads raw data items
    ├── Stage[]
    │     ├── Processor  ── transforms inputs → output
    │     └── Cache      ── reads/writes output.json per item
    └── Models dict  ──── lazily-initialized LLM/VLM clients
```

See [docs/](docs/) for detailed documentation on each component.

## Extending to a new domain

All extension is done via registration APIs — `agentflow/` itself never needs to be edited.

```python
from agentflow.pipeline import Pipeline
from agentflow.input_formater import InputFormater

# 1. Register output types
Pipeline.register_type("MyOutput", MyOutputModel)

# 2. Register custom input formats (optional)
InputFormater.register("my_format", my_format_handler)

# 3. Register custom model backends (optional)
Pipeline.register_model_backend("my_provider", MyLLMClass)
```

Then reference these names in YAML as normal. See [docs/extension_points.md](docs/extension_points.md) for the full reference.

## Running tests

Tests require a vLLM-compatible server at `http://0.0.0.0:8000` serving `google/gemma-3-27b-it`:

```bash
pytest test/ -v
```

Unit tests for `InputFormater` run without a server:

```bash
pytest test/test_input_formater.py -v
```
