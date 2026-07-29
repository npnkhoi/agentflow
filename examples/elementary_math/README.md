# Elementary Math Problems

The worked example from the top-level [README](../../README.md): answer an
elementary-school math question about a figure, outsourcing the arithmetic to a
real calculator so the model never has to do mental math, then grade the answer
against ground truth.

```
Step 1: Image, QuestionText                                  -> Calculations       (LLM)
Step 2: Calculations                                         -> CalculatedNumbers  (calculator)
Step 3: Image, QuestionText, Calculations, CalculatedNumbers -> Answer             (LLM)
Step 4: GroundTruthAnswer, Answer                            -> Verdict            (exact comparison)
```

Steps 1–3 are inference; step 4 is evaluation. Only steps 1 and 3 call a model.
Step 2 is the point of the whole design — a `CalculatorProcessor`, so the
arithmetic is exact — and step 4 is a `VerdictProcessor`, because whether two
numbers agree is arithmetic rather than judgement and the score should not move
when the judge is having an off day.

## Dataset

8 items drawn from [MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
`testmini`, filtered to elementary-school grade, free-form question type, and a
numeric answer — 2 each from CLEVR-Math, IconQA, TabMWP, and IQTest. Rebuild or
resize the sample with:

```bash
python experimental/e0729_build_math_dataset.py --per-source 2
```

Each item supplies three human-annotated fields:

```json
{
  "id": "tabmwp_144",
  "data": {
    "image": "tabmwp_144.png",
    "question_text": "Natalie buys 4.6 kilograms of turmeric. What is the total cost? (Unit: $)",
    "ground_truth_answer": 13.8,
    "source": "TabMWP"
  }
}
```

The image input must be named `Image` in the config: the loader stores the
picture under the item field `image`, and `LLMProcessor` looks for exactly that
key when deciding what to attach to the request. The question text rides along
as a separate human input, `QuestionText`.

## Running

Needs an OpenAI-compatible vision server matching `configs/math.yaml`:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8010 \
  --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85
```

Then, from the repo root:

```bash
python -m examples.elementary_math.run
```

Output lands in `output/elementary_math/<StageName>/<item_id>/`, and the script
prints a per-item verdict plus overall accuracy:

```
    clevr_math_4: FAIL  answer=3.0  answer 3 vs ground truth 4: |difference| = 1, outside tolerance 0.01
       iconqa_21: PASS  answer=2.0  answer 2 vs ground truth 2: |difference| = 0, within tolerance 0.01
      tabmwp_144: PASS  answer=13.8  answer 13.8 vs ground truth 13.8: |difference| = 0, within tolerance 0.01

accuracy: 3/8 = 37.5%
```

To browse the results stage by stage — the figure, the expressions, what the
calculator made of them, the final verdict — use the Streamlit viewer:

```bash
streamlit run agentflow/viewer.py -- examples/elementary_math/configs output
```

To reach it from another machine, see
[Sharing it over Cloudflare](../../README.md#sharing-it-over-cloudflare).

Qwen2.5-VL-3B gets roughly 3/8 here. The failures are visual — miscounting
objects in a CLEVR scene, misreading an IQ-test grid — not arithmetic, which is
what outsourcing step 2 to a calculator is meant to buy you.

## Extension points used

[`extensions.py`](extensions.py) registers everything through the public APIs,
so `agentflow/` itself is untouched:

| What | How |
|---|---|
| `Calculations`, `CalculatedNumbers`, `Answer`, `Verdict` | `Pipeline.register_type(...)` |
| `CalculatorProcessor`, `VerdictProcessor` | `Pipeline.register_processor(...)` |

`VerdictProcessor` reads its `tolerance` from the stage's `kwargs` in the
config (default `0.01`), so how strict grading is lives with the pipeline
definition rather than in code.

`CalculatorProcessor` evaluates each expression with Python's `ast` restricted
to numbers and `+ - * / // % **` — model-authored text is parsed, never
executed. An expression that fails to parse is stored with its error rather
than failing the stage, so the answering step can see which computations worked:

```json
{"expression": "3 | 6", "value": null, "error": "ValueError: unsupported operator: BitOr"}
```

Because a 3B model drifts into formats like `4.6, 3, *`, the step-1 prompt pins
the expression format down with worked examples. That lifted parseable
expressions from 3/8 items to 7/8.
