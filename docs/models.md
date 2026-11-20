# Models

Models are wrappers around LLM/VLM API clients. They share a common interface and are lazily initialized — the underlying client is created on the first `generate()` call.

## Interface

```python
class BaseLLM:
    def __init__(self, base_url: str, token: str, model_id: str): ...

    def generate(
        self,
        system_prompt: str,
        image_path: Path | None,
        examples: list[tuple[Path, str, str]],  # (image, user_prompt, expected_output)
        input_text: str,
        output_type: type[BaseModel] | None = None,
        logger=None,
    ) -> str | None: ...
```

`generate()` returns the raw response text (not yet parsed), or `None` on unrecoverable failure.

When `output_type` is provided, models that support structured output (JSON schema) will constrain the response format accordingly.

---

## Implementations

### OpenAILLM

Uses the OpenAI Python SDK with any OpenAI-compatible endpoint. Works with:
- OpenAI API (`https://api.openai.com/v1`)
- Azure OpenAI (`https://<resource>.openai.azure.com/openai/v1/`)
- Local vLLM servers (`http://0.0.0.0:8010/v1`) — used by the test suite with
  Qwen2.5-VL-3B-Instruct; see [test.md](test.md)
- Local llama.cpp `llama-server` (`http://0.0.0.0:8000/v1`)
- OpenRouter (`https://openrouter.ai/api/v1`)
- LM Studio and similar

```yaml
models:
  my_model:
    cls: openai
    base_url: "https://api.openai.com/v1"
    token: "${OPENAI_API_KEY}"
    model_id: gpt-4o
```

For local servers with no auth, use `token: "-"` and set `base_url` to the local endpoint.

Structured output is requested via `response_format` with a JSON schema.

Set per-model OpenAI SDK request options with `generation_kwargs`:

```yaml
models:
  gemma:
    cls: openai
    base_url: "http://localhost:8000/v1"
    token: "-"
    model_id: google/gemma-4-31B-it
    generation_kwargs:
      temperature: 0
      max_completion_tokens: 2500
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
```

`extra_body` supports provider-specific parameters such as vLLM's thinking
control; the serving model must support them. Options are passed unchanged to
`chat.completions.create`. An omitted temperature or thinking option follows the
server's default. The default completion-token cap remains 4096 unless overridden.
`model`, `messages`, and `response_format` are managed by AgentFlow and cannot be
overridden here.

This option is implemented by `OpenAILLM`. Custom backends can opt in by accepting
a `generation_kwargs` constructor argument. When no options are configured,
existing backends receive the original three constructor arguments unchanged.

### VLLMLocalLLM (`cls: vllm_local`)

Loads the model **in-process** with vLLM's offline engine (`vllm.LLM`), so no
separate `vllm serve` process, port, or `base_url` is needed. This is the
simplest option on Slurm: one process per job, GPUs come from the job's
allocation (`CUDA_VISIBLE_DEVICES`), and everything is torn down when the job ends.

```yaml
models:
  qwen:
    cls: vllm_local
    model_id: Qwen/Qwen2.5-VL-3B-Instruct
    engine_args:              # -> vllm.LLM(model=model_id, **engine_args)
      tensor_parallel_size: 1
      max_model_len: 8192
      gpu_memory_utilization: 0.85
      dtype: half             # e.g. pre-Ampere GPUs
    generation_kwargs:        # -> vllm.SamplingParams(**generation_kwargs)
      temperature: 0
      max_tokens: 2500
      chat_template_kwargs:   # -> LLM.chat(chat_template_kwargs=...)
        enable_thinking: false
```

- **Install:** vLLM must be importable from the Python env that runs agentflow.
  The easiest route is installing agentflow into your existing vLLM env
  (`pip install -e /path/to/agentflow`).
- **Lazy loading:** the engine loads on the first `generate()` call and stays
  resident for the rest of the process. A run where every item is cached never
  loads the model.
- **Engine sharing:** within one process, model entries with the same
  `model_id` and `engine_args` share one loaded engine, even across `Pipeline`
  objects. So you can declare the same model twice with different
  `generation_kwargs`, or run several configs in one script/notebook, without
  loading the weights twice.
- **Messages:** built by the same `build_chat_messages()` helper as `OpenAILLM`
  (system prompt, few-shot demos with images, query), passed to `LLM.chat()`.
- **Structured output:** when the stage has an output type, its JSON schema is
  enforced with `StructuredOutputsParams(json=...)` (falling back to
  `GuidedDecodingParams` on vLLM < 0.10.2).
- **`generation_kwargs` use vLLM `SamplingParams` names**, not OpenAI SDK names:
  `max_tokens` (default 4096, since vLLM's own default is 16) rather than
  `max_completion_tokens`; `chat_template_kwargs` at the top level rather than
  under `extra_body`. `structured_outputs` and `guided_decoding` are managed by
  AgentFlow and cannot be set here. Output that hits `max_tokens` is logged as
  truncated in `run.log`.
- **Engine-level settings** (parallelism, context length, memory, dtype,
  quantization, `trust_remote_code`, `download_dir`, ...) go in `engine_args`.
  `model` is reserved; use `model_id`.
- **Engine subprocess:** AgentFlow sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`
  (unless you already set it), because vLLM forking its engine after CUDA is
  initialized in the main process fails with "Cannot re-initialize CUDA in
  forked subprocess". If you drive a pipeline from your own script, put the
  entry point under `if __name__ == "__main__":` so spawned processes can
  import it safely.
- Requests are sent one item at a time, as with the other backends.

Example Slurm job:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
conda activate vllm      # env with both vLLM and agentflow installed
python my_pipeline.py    # config uses cls: vllm_local
```

### GeminiVLM

Uses `google-genai` for Google Gemini models. Includes:
- Automatic retry with exponential backoff on transient errors
- Rate-limit handling (waits 60 s on 429)
- Blocked-content detection

```yaml
models:
  gemini:
    cls: gemini
    base_url: ""        # not used; kept for interface consistency
    token: "${GEMINI_API_KEY}"
    model_id: gemini-2.0-flash
```

---

## Model definition in config

Models are declared once in the `models` section and referenced by name in each stage:

```yaml
models:
  fast: {cls: openai, base_url: "...", token: "${OPENAI_API_KEY}", model_id: gpt-4o-mini}
  strong: {cls: openai, base_url: "...", token: "${OPENAI_API_KEY}", model_id: gpt-4o}

stages:
  - output: Draft
    model: fast
    ...
  - output: Refined
    model: strong
    ...
```

---

## Message format

For `OpenAILLM` and `VLLMLocalLLM`, each call builds a message list:

```
[system]  system_prompt
[user]    example_image + example_prompt    ─┐
[asst]    example_output                     ├ repeated per demo
...                                          ─┘
[user]    query_image + input_text
```

Images are base64-encoded inline as `data:<mime>;base64,...`.
