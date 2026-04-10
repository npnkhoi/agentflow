import json
import warnings
from pathlib import Path

from pydantic import BaseModel, ValidationError
from agentflow.demo import DemoPool
from agentflow.pipeline import Pipeline
from agentflow.typing.config import StageConfig
from agentflow.processors.base import Processor
from agentflow.input_formater import InputFormater
from agentflow.util import camel_to_snake


class LLMProcessor(Processor):
    def __init__(self, pipeline: Pipeline, stage_config: StageConfig):
        super().__init__(pipeline, stage_config)
        self._model = pipeline.get_model(stage_config.model)
        if stage_config.demo:
            pool_config = pipeline.get_demo_pool(stage_config.demo.pool)
            self._demo_pool = DemoPool(stage_config.demo, pool_config)
        else:
            self._demo_pool = None

        # prompt resolution logic: {output}__{input1}_{input2}__{prompt_version}.md
        prompt_id = f"{stage_config.output}__{'_'.join(self._input_names_camel)}"
        if stage_config.prompt_version:
            prompt_id += f"__{stage_config.prompt_version}"
        filepath = pipeline.prompt_dir / f"{prompt_id}.md"
        with filepath.open() as f:
            self._system_prompt = f.read()
        self.input_formater = InputFormater(stage_config.input_format, self._input_names_snake)

    def _parse_inputs(self, inputs: dict, output_name: str = None):
        image_path = inputs.get("image")
        user_prompt = self.input_formater.format(inputs)
        return (image_path, user_prompt)

    def __call__(self, inputs: dict, logger=None, output_dir: Path | None = None) -> BaseModel | None:
        for input_name in self._input_names_snake:
            if input_name == "image":
                continue
            if inputs.get(input_name) is None:
                warnings.warn(f"Input `{input_name}` is None; skipping LLMProcessor.")
                return None

        image_path, user_prompt = self._parse_inputs(inputs)

        if self._demo_pool:
            demos = self._demo_pool.demos(inputs)
            # persist demo IDs for traceability
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                demo_ids = [d.get("id") for d in demos]
                (output_dir / "demos.json").write_text(
                    json.dumps(demo_ids, indent=2), encoding="utf-8"
                )
            llm_examples = []
            output_name = camel_to_snake(self._stage_config.output)
            for demo in demos:
                demo_output = demo.get(output_name)
                if demo_output is None:
                    continue
                if isinstance(demo_output, BaseModel):
                    demo_output = demo_output.model_dump_json(indent=2)
                elif isinstance(demo_output, dict):
                    demo_output = json.dumps(demo_output, indent=2)
                demo_image_path, demo_user_prompt = self._parse_inputs(demo, output_name)
                llm_examples.append((demo_image_path, demo_user_prompt, demo_output))
        else:
            llm_examples = []

        response_text = self._model.generate(
            system_prompt=self._system_prompt,
            image_path=image_path,
            examples=llm_examples,
            input_text=user_prompt,
            output_type=self._output_type,
            logger=logger,
        )
        if response_text is None:
            return None

        try:
            return self._output_type.model_validate_json(response_text)
        except ValidationError:
            return None
