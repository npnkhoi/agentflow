from typing import Any
from agentflow.processors.base import Processor

class CopyProcessor(Processor):
    def __call__(self, inputs: dict, logger=None, output_dir=None) -> Any:
        """
        Simply copy the input field (there's only one)
        """
        assert len(self._input_names_snake) == 1
        return inputs[self._input_names_snake[0]]
