# AgentFlow

A framework for processing batches of data over agent-based processing graphs. Suitable for running inference and automatic evaluation with static agentic workflows.

## What it does

Agentflow runs a sequence of processing stages over multiple data items.
Each stage reads its inputs (either from the original dataset or from a previous stage's outputs), calls a processor, and writes results to disk.
All outputs are cached (index by (item's ID, stage)) — re-running skips completed item/stage's automatically.

## Quick start

```python
from agentflow.pipeline import Pipeline

# The constructor takes the path to a YAML config file; it is loaded and
# validated internally.
net = Pipeline("pipeline.yaml", prompt_dir="prompts/")
net.execute_all()
```

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

The integration tests (`test/test_integration.py`) need an OpenAI-compatible
server with vision support at `http://0.0.0.0:8000`, served via
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server`). The
model is downloaded automatically on first run. Pick one in a separate
terminal:

**Lightweight — [SmolVLM-500M](https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF)** (~1 GB, fast, minimal RAM):

```bash
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF --host 0.0.0.0 --port 8000 -c 8192
```

The `-c 8192` (8K context) leaves room for the demo-shot prompts used by some
test configs. Once the server is up (`curl http://0.0.0.0:8000/health`), run:

```bash
pytest test/test_integration.py -v
```

Unit tests for `InputFormater` run without any server:

```bash
pytest test/test_input_formater.py -v
```

`test/test_models.py` exercises the hosted OpenAI/Azure and Gemini backends and
needs API keys in `.env` (see `.env.example`), not the local server.
