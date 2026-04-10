# Config reference

Pipelines are configured in YAML (or JSON). The top-level structure:

```yaml
name: <str>               # pipeline name; also the output directory name
wandb_enabled: <bool>     # default: true
n_parallel: <int>         # number of parallel workers; default: 1
loader:   <LoaderConfig>
models:   <dict[str, ModelConfig]>
stages:   <list[StageConfig]>
include:  <list[str]>     # optional: only run these item_ids
exclude:  <list[str]>     # optional: skip these item_ids
include_first: <int>      # optional: only run first N items
```

`include` and `exclude` are mutually exclusive. `include_first` cannot be combined with either.

---

## LoaderConfig

```yaml
loader:
  source: data/items.json   # path to the data file
  kwargs:
    image_dir: data/images/ # passed to DataItemLoader
```

The data file must be a JSON array of `{"id": str, "data": {...}}` objects. The `data.image` field is resolved to a full path using `image_dir`.

---

## ModelConfig

```yaml
models:
  my_model:
    cls: openai          # "openai" | "azure" | "gemini"
    base_url: "http://..."
    token: "sk-..."      # API key; use "-" for local servers
    model_id: "gpt-4o"
```

| `cls`    | Backend               | Notes                                      |
|----------|-----------------------|--------------------------------------------|
| `openai` | OpenAI-compatible API | Works with vLLM, LM Studio, OpenRouter, etc. |
| `azure`  | Azure OpenAI          | Reads `AZURE_OPENAI_API_VERSION` from env  |
| `gemini` | Google Gemini         | `token` is the Gemini API key              |

Models are lazy-initialized — the connection is only opened on the first call.

---

## StageConfig

```yaml
stages:
  - inputs:
      - [TypeName, source]     # one or more input descriptors
    output: TypeName           # name of the Pydantic output type
    processor: LLMProcessor    # processor class name (or dotted path)
    model: my_model            # reference to a key in `models`
    demo:                      # optional few-shot demo config
      source: data/demos.json
      image_dir: data/images/
      select: random           # "random" | "similar"
      shots: 2
    input_format: null         # optional named format (see input_formater.md)
    prompt_version: null       # optional version suffix for prompt file
```

### inputs

Each entry is `[TypeName, source]` where `source` is:
- `human` — read from the raw data loader (the item's original data)
- `model` — read from a previous stage's cache (the named type's output directory)

The `TypeName` must match either a key in the raw data dict (for `human` inputs) or a previously declared stage `output` (for `model` inputs).

**Constraint:** a stage's `output` name must not equal any `model`-sourced input name — they would share the same directory. The config validator enforces this.

### Prompt file resolution

`LLMProcessor` automatically locates the prompt at:

```
<prompt_dir>/<Output>__<Input1>_<Input2>.md
```

If `prompt_version` is set, the filename becomes `<Output>__<Input1>_<Input2>__<version>.md`.

---

## Example: two-stage pipeline

```yaml
name: caption_then_summarize
wandb_enabled: false
loader:
  source: data/items.json
  kwargs:
    image_dir: data/images/
models:
  local:
    cls: openai
    base_url: "http://0.0.0.0:8000/v1"
    token: "-"
    model_id: google/gemma-3-27b-it
stages:
  - inputs:
      - [Image, human]
    output: Caption
    processor: LLMProcessor
    model: local
  - inputs:
      - [Caption, model]
    output: Summary
    processor: LLMProcessor
    model: local
```
