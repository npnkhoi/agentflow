# Few-shot demos

`DemoPool` selects in-context examples for `LLMProcessor`. Demo pools are declared at the top level of the config and referenced by name in each stage.

```yaml
demo_pools:
  train:
    source: data/train.json
    image_dir: data/images/

stages:
  - inputs:
      - [Image, human]
    output: Caption
    processor: LLMProcessor
    model: my_model
    demo:
      pool: train
      select: random
      shots: 2
```

## Config fields

### `demo_pools` (top-level)

| Field | Type | Description |
|---|---|---|
| `source` | `str` | Path to the demo pool JSON file |
| `image_dir` | `str` | Path to the directory containing demo images (default: `""`) |

### `demo` (per stage)

| Field | Type | Description |
|---|---|---|
| `pool` | `str` | Key into `demo_pools` |
| `select` | `"random"` \| `"similar"` | Selection strategy |
| `shots` | `int` | Number of demos to select (default: 4) |

## Selection strategies

### `random`

Picks `shots` items at random, excluding the current item. Fast; no additional setup required.

The pool may overlap with the test set — self-exclusion is always applied so an item is never selected as its own demo.

### `similar`

Picks the `shots` items whose images are most similar to the current item, as measured by CLIP cosine similarity. Requires `transformers`, `torch`, and `Pillow`. Embeddings are computed once at `DemoPool` construction and cached in memory.

## Demo data format

The demo pool JSON follows the same format as the main data file:

```json
[
  {"id": "demo_001", "data": {"image": "001.png", "caption": {...}}},
  ...
]
```

For a demo to contribute an example to the model prompt, the demo item must contain the expected output under the snake_case key of the stage's output type. For example, if the stage output is `Caption`, the demo item must have a `caption` key. Items missing this key are silently skipped.

## Traceability

For each processed item, `LLMProcessor` writes the selected demo IDs to:

```
output/<pipeline>/<Stage>/<item_id>/demos.json
```

This file contains a JSON array of demo item IDs, in selection order:

```json
["demo_042", "demo_017"]
```
