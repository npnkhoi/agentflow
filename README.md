# AgentFlow

Agentflow is a framework for managing batch inferences and evaluation for agentic workflows.

Given some data for a task and some models available, it is 
necessary to experiment with multiple agent harnesses (agentic topology, 
prompts, in-context learning set up, and generation hyperameters) for evaluation 
purposes.

Features:

- Idempotency: never rerun inference on a finished stage or processed data item
- Convenient prompt management in actual text files, enabling hill-climbing with prompts.
- In-context learning support (configurable `num_shots`, `selection_strategy`, and `demo_pool`)
- Variety of supported model providers
- Media-first design, starting with images

How it works:

- It runs a sequence of processing stages over multiple data items.
- Each stage reads its inputs (either from the dataset or from a 
previous stage's outputs), calls a processor, and writes results to disk.
- All outputs are cached (index by (data item's ID and stage)) — re-running skips 
completed item/stage's automatically.

## Example: Elementary Math Problems

Let's consider the task of answering an elementary math question about a figure:
```
Input: Image, QuestionText
Output: A real number
```

To reduce mathematical hallucinations, it is helpful to outsource the 
calculations into an actual calculator (which is just another agent). So you 
want to prototype the following workflow:

- Step 1: `Image`, `QuestionText` -> `Calculations`
- Step 2: `Calculations` -> `CalculatedNumbers`
- Step 3: `Image`, `QuestionText`, `Calculations`, `CalculatedNumbers` -> `Answer` (float)

Additionally, because we happen to have the ground truth answer to each 
question, we want the workflow to give us a binary verdict as well. Whether two 
numbers agree is arithmetic rather than judgement, so this stage is a 
deterministic processor, not another agent:

- Step 4: `GroundTruthAnswer`, `Answer` -> `Verdict`

So the whole agentic workflow, comprising inference (first 3 steps) and evaluation (the final step) is illustrated in the following graph:

![](docs/assets/e0728_workflow_graph.svg)

This workflow is implemented end-to-end in
[examples/elementary_math/](examples/elementary_math/), over 8 elementary items
from MathVista. Run it from the repo root with
`python -m examples.elementary_math.run`.

To build a workflow like this, do the following steps:

1. Prepare the dataset as a JSON file. Every field in `data` is available to
stages as a human-annotated input, addressed by its CamelCase name
(`ground_truth_answer` → `GroundTruthAnswer`):

```json
// data/items.json
[
  {"id": "tabmwp_144", "data": {"image": "tabmwp_144.png", "question_text": "...", "ground_truth_answer": 13.8}},
  {"id": "iconqa_21",  "data": {"image": "iconqa_21.png",  "question_text": "...", "ground_truth_answer": 2.0}}
]
```

2. Put the images in `data/images/`.

```bash
#  inside `data/images/`
tabmwp_144.png
iconqa_21.png
```

3. Write the prompts in `prompts/`, one per LLM stage, named
`<Output>__<Input1>_<Input2>.md`. Deterministic stages need no prompt:

```bash
Calculations__Image_QuestionText.md
Answer__Image_QuestionText_Calculations_CalculatedNumbers.md
```

4. Write the config in `pipeline.yaml`. Each stage names its inputs (and whether
each comes from the dataset or from an earlier stage), its output type, and the
processor that produces it:

```yaml
name: elementary_math
loader:
  source: data/items.json
  kwargs:
    image_dir: data/images
models:
  qwen:
    cls: openai
    base_url: "http://0.0.0.0:8010/v1"
    token: "-"
    model_id: Qwen/Qwen2.5-VL-3B-Instruct
stages:
  - inputs: [[Image, human], [QuestionText, human]]
    output: Calculations
    processor: LLMProcessor
    model: qwen
  - inputs: [[Calculations, model]]
    output: CalculatedNumbers
    processor: CalculatorProcessor      # not an LLM — exact arithmetic
  - inputs: [[Image, human], [QuestionText, human], [Calculations, model], [CalculatedNumbers, model]]
    output: Answer
    processor: LLMProcessor
    model: qwen
  - inputs: [[GroundTruthAnswer, human], [Answer, model]]
    output: Verdict
    processor: VerdictProcessor         # not an LLM — exact comparison
    kwargs:
      tolerance: 0.01
```

5. Write the domain's pieces: a Pydantic model per stage output, and a
`Processor` subclass for each stage that is not an LLM call. A processor is one
method — inputs in (keyed by snake_case name), an output model out:

```python
# extensions.py
from pydantic import BaseModel
from agentflow.processors.base import Processor

class Calculations(BaseModel):
    reasoning: str
    expressions: list[str]          # e.g. ["3 * 4.6"]

class CalculatedNumber(BaseModel):
    expression: str
    value: float | None = None      # exactly one of value / error is set
    error: str | None = None

class CalculatedNumbers(BaseModel):
    results: list[CalculatedNumber]

class Answer(BaseModel):
    value: float

class Verdict(BaseModel):
    correct: bool
    reason: str


class CalculatorProcessor(Processor):
    """Calculations -> CalculatedNumbers. `evaluate` parses with `ast`, allowing
    only numbers and + - * / // % ** — model text is parsed, never executed."""

    def __call__(self, inputs, logger=None, output_dir=None):
        calculations = inputs["calculations"]
        if calculations is None:
            return None                              # upstream failed; fail this stage too

        results = []
        for expression in calculations.expressions:
            try:
                results.append(CalculatedNumber(expression=expression, value=evaluate(expression)))
            except Exception as e:
                results.append(CalculatedNumber(expression=expression, error=f"{type(e).__name__}: {e}"))
        return CalculatedNumbers(results=results)   # a bad expression is recorded, not fatal


class VerdictProcessor(Processor):
    """GroundTruthAnswer, Answer -> Verdict, by numeric comparison."""

    def __call__(self, inputs, logger=None, output_dir=None):
        answer, ground_truth = inputs["answer"], inputs["ground_truth_answer"]
        if answer is None or ground_truth is None:
            return None                              # upstream failed; fail this stage too
        tolerance = (self._stage_config.kwargs or {}).get("tolerance", 0.01)
        difference = abs(answer.value - float(ground_truth))
        return Verdict(correct=difference <= tolerance, reason=f"|difference| = {difference:g}")
```

Returning `None` fails the stage for that item: the pipeline abandons the item
and writes nothing, so a later run retries it instead of caching a bogus result.
See
[examples/elementary_math/extensions.py](examples/elementary_math/extensions.py)
for the full versions.

6. Teach agentflow this domain. Everything the config referred to by name — the
output types, and the two processors that are not LLM calls — is supplied
through registration APIs, so `agentflow/` itself never needs to be edited:

```python
# run.py
from agentflow.pipeline import Pipeline

# 1. Output types: a Pydantic model per stage output. The name is what the
#    config's `output` field and the cache directory use.
Pipeline.register_type("Calculations", Calculations)
Pipeline.register_type("CalculatedNumbers", CalculatedNumbers)
Pipeline.register_type("Answer", Answer)
Pipeline.register_type("Verdict", Verdict)

# 2. Processors: any stage that is not an LLM call. Here, exact arithmetic and
#    exact grading.
Pipeline.register_processor("CalculatorProcessor", CalculatorProcessor)
Pipeline.register_processor("VerdictProcessor", VerdictProcessor)

net = Pipeline("pipeline.yaml", prompt_dir="prompts/")
net.execute_all()
```

7. Run it:

```bash
python -m examples.elementary_math.run
```

Two further extension points exist, which this example does not need:

```python
from agentflow.input_formater import InputFormater

# How a stage's inputs are rendered into the user prompt (default: labelled JSON)
InputFormater.register("my_format", my_format_handler)

# A model provider beyond the built-in `openai` and `gemini` backends
Pipeline.register_model_backend("my_provider", MyLLMClass)
```

See [docs/extension_points.md](docs/extension_points.md) for the full reference.

## Output layout

```
output/<pipeline-name>/
  <StageName>/
    <item_id>/
      output.json   # stage output (Pydantic model serialized to JSON)
      run.log       # model call log for this item
      demos.json    # (if demos enabled) list of selected demo item_ids
```

## Documentation

Detailed docs for each component live in [docs/](docs/):

| | |
|---|---|
| [pipeline.md](docs/pipeline.md) | stages, execution, caching |
| [config.md](docs/config.md) | the YAML schema |
| [processors.md](docs/processors.md) | built-in and custom processors |
| [output_types.md](docs/output_types.md) | stage outputs as Pydantic models |
| [models.md](docs/models.md) | model backends and providers |
| [demos.md](docs/demos.md) | in-context learning and demo pools |
| [input_formater.md](docs/input_formater.md) | rendering inputs into prompts |
| [extension_points.md](docs/extension_points.md) | the registration APIs |
| [test.md](docs/test.md) | running the test suites |
