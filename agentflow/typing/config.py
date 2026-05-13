from pydantic import BaseModel, model_validator, field_validator
from pathlib import Path
from typing import Any, Union
from enum import Enum
import os
import re
from agentflow.const import AnnotationSource, DemoSelect

class ModelConfig(BaseModel):
    cls: str = "openai"  # "openai" | "gemini"
    base_url: str
    token: str = ""
    model_id: str

    @field_validator("token", mode="before")
    @classmethod
    def _expand_env_vars(cls, v: str) -> str:
        return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), v)


class LoaderConfig(BaseModel):
    source: Path
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None


class DemoPoolConfig(BaseModel):
    source: str
    image_dir: str = ""


class DemoConfig(BaseModel):
    pool: str                        # key into Config.demo_pools
    select: DemoSelect = DemoSelect.RANDOM
    shots: int = 4


class StageConfig(BaseModel):
    """
    a stage is defined by the variables below.
    the output of the stage will be saved in a dir named as the output field.
    """

    # name: str
    # args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    inputs: list[
        Union[
            tuple[str, AnnotationSource],
            tuple[str, AnnotationSource, str],
        ]
    ]
    output: str
    processor: str
    model: str | None = None  # reference name into Config.models; required if processor == LLMProcessor
    demo: DemoConfig | None = None
    prompt_version: str | None = None
    input_format: str | None = None

    @model_validator(mode="after")
    def _no_output_same_as_model_input(self) -> "StageConfig":
        """Output dir must not collide with a model-sourced input dir."""
        for entry in self.inputs:
            type_name, source = entry[0], entry[1]
            if source == AnnotationSource.MODEL and type_name == self.output:
                raise ValueError(
                    f"Stage output '{self.output}' collides with model-sourced input "
                    f"'{type_name}': both map to the same output directory. "
                    f"Use a different output name."
                )
        return self


class Config(BaseModel):
    name: str
    loader: LoaderConfig
    models: dict[str, ModelConfig] = {}
    demo_pools: dict[str, DemoPoolConfig] = {}
    stages: list[StageConfig]
    exclude: list[str] | None = None
    include: list[str] | None = None
    include_first: int | None = None
    wandb_enabled: bool = True
    use_new_logging: bool = False
    # use_cache: bool = True
    n_parallel: int = 1

    @model_validator(mode='after')
    def _check_demo_pool_refs(self) -> 'Config':
        for i, stage in enumerate(self.stages):
            if stage.demo is not None and stage.demo.pool not in self.demo_pools:
                raise ValueError(
                    f"Stage {i} (output='{stage.output}') references demo pool "
                    f"'{stage.demo.pool}', but it is not declared in demo_pools. "
                    f"Available: {list(self.demo_pools)}"
                )
        return self

    @model_validator(mode='before')
    @classmethod
    def _normalize_id_lists(cls, data: Any) -> Any:
        """Accept both old [item_id, idx] pairs and new plain item_id strings."""
        for field in ('include', 'exclude'):
            val = data.get(field)
            if val is not None:
                data[field] = [
                    item if isinstance(item, str) else item[0]
                    for item in val
                ]
        return data

    @model_validator(mode='after')
    def _check_include_first_exclusivity(self) -> 'Config':
        if self.include_first is not None and (self.include is not None or self.exclude is not None):
            raise ValueError("include_first cannot be used together with include or exclude")
        return self
