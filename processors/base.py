from abc import ABC, abstractmethod

from pydantic import BaseModel

from agentflow.typing.config import StageConfig
from agentflow.pipeline import Pipeline
from agentflow.util import camel_to_snake

class Processor(ABC):
    def __init__(self, pipeline: Pipeline, stage_config: StageConfig):
        self._pipeline = pipeline
        self._stage_config = stage_config
        self._output_type: type[BaseModel] = Pipeline.get_type(stage_config.output)
        self._input_names_camel = [t[0] for t in stage_config.inputs]
        self._input_names_snake = [camel_to_snake(t[0]) for t in stage_config.inputs]

    @abstractmethod
    def __call__(self, inputs: dict, logger=None, output_dir: "Path | None" = None) -> BaseModel | None:
        pass
