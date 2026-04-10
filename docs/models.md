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
- Local vLLM servers (`http://0.0.0.0:8000/v1`)
- OpenRouter (`https://openrouter.ai/api/v1`)
- LM Studio and similar

```yaml
models:
  my_model:
    cls: openai
    base_url: "http://0.0.0.0:8000/v1"
    token: "-"          # use "-" for local servers with no auth
    model_id: google/gemma-3-27b-it
```

Structured output is requested via `response_format` with a JSON schema.

### AzureVLM

Extends `OpenAILLM` for Azure OpenAI. Reads `AZURE_OPENAI_API_VERSION` from the environment (default: `2025-01-01-preview`).

```yaml
models:
  azure:
    cls: azure
    base_url: "https://my-resource.openai.azure.com/"
    token: "<azure-api-key>"
    model_id: gpt-4o
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
    token: "<GEMINI_API_KEY>"
    model_id: gemini-2.0-flash
```

---

## Model definition in config

Models are declared once in the `models` section and referenced by name in each stage:

```yaml
models:
  fast: {cls: openai, base_url: "...", token: "...", model_id: gpt-4o-mini}
  strong: {cls: openai, base_url: "...", token: "...", model_id: gpt-4o}

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

For `OpenAILLM`, each call builds a message list:

```
[system]  system_prompt
[user]    example_image + example_prompt    ─┐
[asst]    example_output                     ├ repeated per demo
...                                          ─┘
[user]    query_image + input_text
```

Images are base64-encoded inline as `data:<mime>;base64,...`.
