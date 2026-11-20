import json
import os
import time
import httpx
from pathlib import Path
from openai import OpenAI
import pydantic_core
from pydantic import BaseModel
from dotenv import load_dotenv
import base64
from copy import deepcopy

load_dotenv()


def image_path_to_base64(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        base64_encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded}"


def build_chat_messages(
    system_prompt: str,
    image_path: Path | None,
    examples: list[tuple[Path, str, str]],
    input_text: str,
) -> list[dict]:
    """Build an OpenAI-style chat message list: system, few-shot demos, then the query.

    Shared by ``OpenAILLM`` and ``VLLMLocalLLM``; images are inlined as base64 data URIs.
    """
    messages: list[dict] = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    for example_image_path, example_prompt, example_output in examples:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path_to_base64(example_image_path)}},
                {"type": "text", "text": example_prompt},
            ]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example_output}]
        })
    user_input = []
    if image_path:
        user_input.append({"type": "image_url", "image_url": {"url": image_path_to_base64(image_path)}})
    user_input.append({"type": "text", "text": input_text})
    messages.append({"role": "user", "content": user_input})
    return messages


class BaseLLM:
    def __init__(self, base_url: str, token: str, model_id: str):
        self._base_url = base_url
        self._token = token
        self._model_id = model_id
        self._ready: bool = False

    def _lazy_init(self) -> None:
        if not self._ready:
            self._do_init()
            self._ready = True

    def _do_init(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        system_prompt: str,
        image_path: Path | None,
        examples: list[tuple[Path, str, str]],
        input_text: str,
        output_type: type[BaseModel] | None = None,
        logger=None,
    ) -> str | None:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self, base_url: str, token: str, model_id: str, generation_kwargs: dict | None = None):
        super().__init__(base_url, token, model_id)
        self._generation_kwargs = deepcopy(generation_kwargs or {})
        reserved = {"model", "messages", "response_format"} & self._generation_kwargs.keys()
        if reserved:
            raise ValueError(f"generation_kwargs cannot override pipeline-managed fields: {sorted(reserved)}")

    def _do_init(self) -> None:
        self._client = OpenAI(base_url=self._base_url, api_key=self._token)

    def generate(
        self,
        system_prompt: str,
        image_path: Path | None,
        examples: list[tuple[Path, str, str]],
        input_text: str,
        output_type: type[BaseModel] | None = None,
        logger=None,
    ) -> str:
        self._lazy_init()
        messages = build_chat_messages(system_prompt, image_path, examples, input_text)

        create_kwargs: dict = {"max_completion_tokens": 4096, **deepcopy(self._generation_kwargs)}
        create_kwargs.update(messages=messages, model=self._model_id)
        if output_type is not None:
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "schema": output_type.model_json_schema(),
                },
            }
        chat_completion = self._client.chat.completions.create(**create_kwargs)
        content = str(chat_completion.choices[0].message.content)
        print("OUTPUT:", content, file=logger, flush=True)
        return content


class VLLMLocalLLM(BaseLLM):
    """Run a model in-process with vLLM's offline engine (``vllm.LLM``); no server needed.

    The engine loads on the first ``generate()`` call and lives until the process
    exits, so fully cached runs never touch the GPU. ``engine_args`` go to
    ``vllm.LLM(...)`` (e.g. ``tensor_parallel_size``, ``max_model_len``, ``dtype``);
    ``generation_kwargs`` go to ``vllm.SamplingParams(...)`` (e.g. ``temperature``,
    ``max_tokens``), except ``chat_template_kwargs``, which goes to ``LLM.chat()``.
    Requires vLLM installed in the same Python environment as agentflow.
    """

    _RESERVED_SAMPLING = {"structured_outputs", "guided_decoding"}
    # One engine per (model_id, engine_args) per process: pipelines built repeatedly in one
    # process (notebooks, sweeps), or several config entries for the same model with different
    # generation_kwargs, reuse the loaded weights instead of loading a second copy onto the GPU.
    _engines: dict[str, object] = {}

    def __init__(
        self,
        base_url: str,
        token: str,
        model_id: str,
        generation_kwargs: dict | None = None,
        engine_args: dict | None = None,
    ):
        super().__init__(base_url, token, model_id)
        self._generation_kwargs = deepcopy(generation_kwargs or {})
        self._engine_args = deepcopy(engine_args or {})
        reserved = self._RESERVED_SAMPLING & self._generation_kwargs.keys()
        if reserved:
            raise ValueError(f"generation_kwargs cannot override pipeline-managed fields: {sorted(reserved)}")
        if "model" in self._engine_args:
            raise ValueError("engine_args cannot set 'model'; use model_id instead")

    def _do_init(self) -> None:
        # vLLM starts its engine core in a subprocess. Forking breaks once CUDA is initialized
        # in this process (e.g. by vLLM's own multimodal warmup, or by caller code touching
        # torch.cuda), and vLLM does not always detect that; spawn is safe either way.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "cls: vllm_local needs vLLM in the same Python environment as agentflow "
                "(e.g. `pip install -e .` inside your vLLM env)"
            ) from e
        key = json.dumps([self._model_id, self._engine_args], sort_keys=True, default=str)
        if key not in VLLMLocalLLM._engines:
            VLLMLocalLLM._engines[key] = LLM(model=self._model_id, **self._engine_args)
        self._llm = VLLMLocalLLM._engines[key]

    @staticmethod
    def _sampling_params(options: dict, output_type: type[BaseModel] | None):
        from vllm import SamplingParams
        options = {"max_tokens": 4096, **options}
        if output_type is not None:
            schema = output_type.model_json_schema()
            try:
                from vllm.sampling_params import StructuredOutputsParams
                options["structured_outputs"] = StructuredOutputsParams(json=schema)
            except ImportError:  # vLLM < 0.10.2 named this guided decoding
                from vllm.sampling_params import GuidedDecodingParams
                options["guided_decoding"] = GuidedDecodingParams(json=schema)
        return SamplingParams(**options)

    def generate(
        self,
        system_prompt: str,
        image_path: Path | None,
        examples: list[tuple[Path, str, str]],
        input_text: str,
        output_type: type[BaseModel] | None = None,
        logger=None,
    ) -> str | None:
        self._lazy_init()
        messages = build_chat_messages(system_prompt, image_path, examples, input_text)
        options = deepcopy(self._generation_kwargs)
        chat_template_kwargs = options.pop("chat_template_kwargs", None)
        params = self._sampling_params(options, output_type)
        outputs = self._llm.chat(messages, params, use_tqdm=False, chat_template_kwargs=chat_template_kwargs)
        completion = outputs[0].outputs[0]
        content = completion.text
        if completion.finish_reason == "length":
            print(f"WARNING: hit max_tokens ({params.max_tokens}); output is truncated", file=logger, flush=True)
        print("OUTPUT:", content, file=logger, flush=True)
        return content


class GeminiVLM(BaseLLM):
    def _do_init(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=self._token)

    @staticmethod
    def _get_bytes_from_path(path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _handle_retry(self, attempt: int, max_retry: int, base_delay: int, error_msg: str, logger=None) -> bool:
        print(f"{error_msg} (attempt {attempt + 1}/{max_retry})", file=logger, flush=True)
        if attempt < max_retry - 1:
            delay = base_delay * (2 ** attempt)
            print(f"Retrying in {delay:.1f} seconds...", file=logger, flush=True)
            time.sleep(delay)
            return True
        return False

    def generate(
        self,
        system_prompt: str,
        image_path: Path | None,
        examples: list[tuple[Path, str, str]],
        input_text: str,
        output_type: type[BaseModel] | None = None,
        logger=None,
    ) -> str | None:
        from google.genai import types
        from google.genai.errors import ClientError
        self._lazy_init()
        messages = []
        for example_image_path, example_input_text, example_output_text in examples:
            messages.append(types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=self._get_bytes_from_path(example_image_path), mime_type="image/png"),
                    types.Part.from_text(text=example_input_text),
                ]
            ))
            messages.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=example_output_text)]
            ))
        user_input = []
        if image_path is not None:
            user_input.append(types.Part.from_bytes(data=self._get_bytes_from_path(image_path), mime_type="image/png"))
        user_input.append(types.Part.from_text(text=input_text))
        messages.append(types.Content(role="user", parts=user_input))

        schema = output_type.model_json_schema() if output_type is not None else None
        MAX_RETRY = 3
        base_delay = 2

        for attempt in range(MAX_RETRY):
            print(f"MESSAGES: {messages}", file=logger, flush=True)
            try:
                response = self._client.models.generate_content(
                    model=self._model_id,
                    contents=messages,
                    config={
                        "response_json_schema": schema,
                        "response_mime_type": "application/json",
                        "system_instruction": system_prompt,
                        "temperature": 0.0,
                    }
                )
            except ClientError as e:
                if e.code == 429:
                    delay = 60
                    msg = f"Rate limited (429). Waiting {delay}s... (attempt {attempt + 1}/{MAX_RETRY})"
                    print(msg, file=logger, flush=True)
                    print(msg)
                    time.sleep(delay)
                    continue
                raise
            except Exception as e:
                if isinstance(e, (httpx.RemoteProtocolError, httpx.TimeoutException)):
                    if not self._handle_retry(attempt, MAX_RETRY, base_delay, f"API error: {e}", logger):
                        return None
                    continue
                raise

            try:
                raw = str(response.text)
                prompt_feedback = getattr(response, "prompt_feedback", None)
                block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback is not None else None
                if block_reason is not None and str(block_reason) == "BlockedReason.PROHIBITED_CONTENT":
                    if not self._handle_retry(attempt, MAX_RETRY, base_delay, f"Prompt blocked: {prompt_feedback}", logger):
                        return None
                    continue
                print(f"RESPONSE TEXT: {raw}", file=logger, flush=True)
                print(f"PROMPT FEEDBACK: {response.prompt_feedback}", file=logger, flush=True)
                print(f"USAGE METADATA: {response.usage_metadata}", file=logger, flush=True)
                if output_type is not None:
                    output_type.model_validate_json(raw)
                return raw
            except pydantic_core._pydantic_core.ValidationError:
                if not self._handle_retry(attempt, MAX_RETRY, base_delay, f"Invalid response: {response.text}", logger):
                    return None
                continue

        return None
