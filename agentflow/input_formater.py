"""
Domain-agnostic input formatter for LLMProcessor.

Custom formats can be registered at application startup:
    from agentflow.input_formater import InputFormater
    InputFormater.register("my_format", my_handler_fn)

A handler has signature: (inputs: dict, input_names_snake: list[str]) -> str
"""
from typing import Callable
import warnings
from pydantic import BaseModel


class InputFormater:
    _registry: dict[str, Callable] = {}

    def __init__(self, input_format: str | None, input_names_snake: list[str]):
        self.input_format = input_format
        self._input_names_snake = input_names_snake

    @classmethod
    def register(cls, format_name: str, handler: Callable) -> None:
        """Register a named input format handler.

        handler(inputs: dict, input_names_snake: list[str]) -> str
        """
        cls._registry[format_name] = handler

    def format(self, inputs: dict) -> str:
        if not self.input_format:
            return self._default_format(inputs)
        if self.input_format in self._registry:
            return self._registry[self.input_format](inputs, self._input_names_snake)
        raise NotImplementedError(
            f"Input format '{self.input_format}' is not registered. "
            f"Call InputFormater.register('{self.input_format}', handler) before use."
        )

    def _default_format(self, inputs: dict) -> str:
        user_prompt = ""
        for input_name in self._input_names_snake:
            if input_name == "image":
                continue
            if user_prompt:
                user_prompt += "\n\n"
            value = inputs[input_name]
            if isinstance(value, BaseModel):
                value = value.model_dump_json(indent=2)
            user_prompt += f"{input_name}:\n{value}"
        return user_prompt
