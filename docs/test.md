# Running tests

Tests are split into two suites:

- `test/unit/` — pure unit tests (`InputFormater`, config validation, and the
  elementary-math example's calculator and grading processors). No server or API
  keys required.
- `test/integration/` — end-to-end pipeline runs over the example projects in
  [../examples/](../examples/). Each test module targets one example project
  (`test_captioning.py` → `examples/captioning`, `test_counting.py` →
  `examples/counting`, `test_elementary_math.py` → `examples/elementary_math`,
  `test_hosted_models.py` → hosted OpenAI/Gemini backends).

Run them as modules from the repo root, so `agentflow` and `examples` import
normally:

```bash
python -m pytest test/ -v
```

## Local server

The integration tests (except the hosted-model ones) need an OpenAI-compatible
server with vision support at `http://0.0.0.0:8010`, served via
[vLLM](https://github.com/vllm-project/vllm). The example configs use
[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
(~7 GB, downloaded automatically on first run). In a separate terminal:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --host 0.0.0.0 --port 8010 \
  --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85
```

`--max-model-len 8192` (8K context) leaves room for the demo-shot prompts used
by some test configs. Two flags matter on older GPUs: `--dtype half`, because
pre-Ampere cards lack bfloat16, and — if startup dies in a FlashInfer
`Ninja build failed` — `--attention-backend TRITON_ATTN`, because FlashInfer
cannot JIT its prefill kernels below compute capability 8.0. (The environment
variable `VLLM_ATTENTION_BACKEND` no longer does anything in vLLM 0.19; it warns
and is ignored.)

Once the server is up (`curl http://0.0.0.0:8010/health`), run everything:

```bash
python -m pytest test/ -v
```

Or just the local-server integration tests:

```bash
python -m pytest test/integration/test_captioning.py test/integration/test_counting.py \
  test/integration/test_elementary_math.py -v
```

Unit tests run without any server:

```bash
python -m pytest test/unit/ -v
```

## Hosted models

`test/integration/test_hosted_models.py` exercises the hosted OpenAI/Azure and
Gemini backends and needs API keys in `.env` (see `.env.example`), not the local
server. Without keys, those tests skip.

## What integration tests assert

They check pipeline mechanics — which stages ran, that outputs validate against
their Pydantic types, that caching skips completed work — not model quality. A
weak model that answers every question wrong still passes.

One caveat when adding tests: `Pipeline` catches per-item exceptions and prints
`failed stage N ... giving up` rather than raising, so a test that only runs a
pipeline cannot fail. Assert on the outputs, and make sure assertions that loop
over a glob also assert the glob is non-empty — otherwise they pass vacuously
when every item failed.
