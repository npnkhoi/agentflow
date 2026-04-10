# Pipeline execution — Pipeline & Stage

## Pipeline

`Pipeline` is the top-level pipeline object. It is constructed from a `Config` and a `prompt_dir`.
All backends (loader, model, output type) are resolved from registries at construction time —
see [extension_points.md](extension_points.md) for how to register custom backends.



```python
net = Pipeline(cfg, prompt_dir="prompts/")
net.execute_all()          # run every item in the dataset
net.execute("item_001")    # run a single item
```

During construction it:
1. Instantiates a `DataItemLoader` from `cfg.loader`.
2. Builds the `models` dict — one lazy `BaseLLM` instance per model declared in `cfg.models`.
3. For each stage in `cfg.stages`, instantiates an `Stage`.

### Key properties / methods

| Member | Description |
|---|---|
| `item_ids` | Ordered list of all item IDs from the loader |
| `get_model(name)` | Returns the named `BaseLLM`; raises `KeyError` with helpful message |
| `prompt_dir` | `Path` to the prompt file directory |
| `cache(name)` | Returns the `Cache` for the named stage, or `None` |
| `execute(item_id)` | Run the full pipeline for one item; returns `True` on success |
| `execute_all()` | `execute()` for every item in `item_ids` order |

---

## Stage

Each pipeline stage is represented by an `Stage`, constructed inside `Pipeline.__init__`:

```python
Stage(
    name=stage_config.output,      # e.g. "Caption"
    processor=processor_instance,
    loader=loader,
    inputs=[(TypeName, source), ...],
    output_type=CaptionModel,
    root=Path("output/my_pipeline"),
)
```

`Stage.__init__` builds:
- `input_caches` — one `Cache` per model-sourced input, pointing at a previous stage's directory
- `cache` — the output `Cache` for this stage

---

## Execution flow

For each item, `_execute_stage` follows this sequence:

```
has cache? ──yes──► return True (skip)
     │
    no
     │
     ▼
load human inputs from DataItemLoader
load model inputs from input_caches
build inputs dict {"id": item_id, ...inputs}
     │
     ▼
call processor(inputs, logger, output_dir)
     │
  None? ──► return False (failure)
     │
  output
     │
     ▼
cache.store(item_id, output)
return True
```

On failure, the stage is retried once (with all preceding caches cleared). After `MAX_RESET=1` failures the item is abandoned.

---

## Output directory layout

```
output/<pipeline-name>/
  <StageName>/
    <item_id>/
      output.json   # Pydantic model → JSON
      run.log       # stdout from the model call
      demos.json    # (LLMProcessor only) list of demo item_ids used
```

The `Cache` class manages this layout. `output_filepath(item_id)` returns the canonical path; `has()`, `load()`, `store()`, `delete()` operate on `output.json`.
