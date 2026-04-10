import time
from pathlib import Path
from collections.abc import Callable
from typing import cast
from agentflow.typing.config import Config
from pydantic import BaseModel as _BaseModel
from pydantic import BaseModel
from agentflow.loaders import Loader, Cache, DataItemLoader
from agentflow.models import BaseLLM
from agentflow.const import AnnotationSource
from agentflow.util import ShutdownFlag, camel_to_snake


class Stage:
    def __init__(
        self,
        name: str,
        processor: Callable[[BaseModel], BaseModel | None],
        loader: Loader,
        inputs: list[tuple[str, AnnotationSource]],
        output_type: type[BaseModel],
        root: Path,
    ):
        self.name = name
        self.processor = processor
        self.loader = loader
        self.inputs = inputs

        self.input_caches: dict[str, Cache] = {}
        for input_type, input_source in inputs:
            if input_source == AnnotationSource.MODEL:
                self.input_caches[input_type] = Cache(path=root / input_type, datatype=Pipeline.get_type(input_type))

        self.cache = Cache(path=root / name, datatype=output_type)


class Pipeline:
    _type_registry: dict[str, type[_BaseModel]] = {}

    @classmethod
    def register_type(cls, name: str, output_cls: type[_BaseModel]) -> None:
        """Register an output type under a logical name.

            Pipeline.register_type("MyOutput", MyOutput)

        The name is what appears in the config's ``output`` field and as the
        cache directory name. The same Pydantic class can be registered under
        multiple names to produce independently cached outputs that share a
        schema:

            Pipeline.register_type("Caption", TextOutput)
            Pipeline.register_type("RefinedCaption", TextOutput)

        Must be called before constructing Pipeline.
        """
        cls._type_registry[name] = output_cls

    @classmethod
    def get_type(cls, name: str) -> type[_BaseModel]:
        if name not in cls._type_registry:
            raise KeyError(
                f"Output type '{name}' is not registered. "
                f"Call Pipeline.register_type('{name}', YourClass) before building the pipeline."
            )
        return cls._type_registry[name]

    _processor_registry: dict[str, type] = {}

    @classmethod
    def register_processor(cls, name: str, processor_cls: type) -> None:
        """Register a processor class under a short name.

            from agentflow.pipeline import Pipeline
            Pipeline.register_processor("MyProcessor", MyProcessor)

        Then use ``processor: MyProcessor`` in the config.
        """
        cls._processor_registry[name] = processor_cls

    _model_backends: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register_model_backend(cls, name: str, backend: type[BaseLLM]) -> None:
        """Register a custom model backend class under a new cls name.

            from agentflow.pipeline import Pipeline
            Pipeline.register_model_backend("my_backend", MyLLM)

        Then use ``cls: my_backend`` in the config's models section.
        """
        cls._model_backends[name] = backend

    def __init__(self, cfg: Config, prompt_dir: str = "prompts"):
        if cfg.loader.args is None:
            cfg.loader.args = []
        if cfg.loader.kwargs is None:
            cfg.loader.kwargs = {}
        loader = DataItemLoader(cfg.loader.source, *cfg.loader.args, **cfg.loader.kwargs)
        root = Path("output") / cfg.name

        self._stages: list[Stage] = []
        self._root = root.resolve()
        self._loader = loader
        self._prompt_dir = Path(prompt_dir)
        self._shutdown_flag = ShutdownFlag()
        self._demo_pools = cfg.demo_pools
        self._models: dict[str, BaseLLM] = {
            name: self._model_backends[mcfg.cls](
                base_url=mcfg.base_url,
                token=mcfg.token,
                model_id=mcfg.model_id,
            )
            for name, mcfg in cfg.models.items()
        }

        for stage_config in cfg.stages:
            output_type = Pipeline.get_type(stage_config.output)
            processor_cls = self._processor_registry.get(stage_config.processor)
            if processor_cls is None:
                raise KeyError(
                    f"Processor '{stage_config.processor}' is not registered. "
                    f"Call Pipeline.register_processor('{stage_config.processor}', YourClass) before building the pipeline."
                )
            self._stages.append(Stage(
                name=stage_config.output,
                processor=processor_cls(self, stage_config),
                loader=loader,
                inputs=cast(list[tuple], stage_config.inputs),
                output_type=output_type,
                root=self._root,
            ))

    @property
    def prompt_dir(self) -> Path:
        return self._prompt_dir

    def get_demo_pool(self, name: str):
        from agentflow.typing.config import DemoPoolConfig
        if name not in self._demo_pools:
            raise KeyError(f"Demo pool '{name}' not declared in config. Available: {list(self._demo_pools)}")
        return self._demo_pools[name]

    def get_model(self, name: str) -> BaseLLM:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not defined in config. Available: {list(self._models)}")
        return self._models[name]

    @property
    def item_ids(self) -> list[str]:
        return self._loader.item_ids

    def cache(self, name: str) -> Cache | None:
        for stage in self._stages:
            if stage.name == name:
                return stage.cache
        return None

    def execute_all(self):
        for item_id in self.item_ids:
            self.execute(item_id)

    def execute(self, item_id: str) -> bool:
        """Execute pipeline on a single item. Returns True on success."""
        if item_id not in self.item_ids:
            raise ValueError(f"item_id '{item_id}' not found")

        i = 0
        cnt_reset = 0
        MAX_RESET = 1
        while i < len(self._stages):
            if self._execute_stage(item_id, i):
                i += 1
            else:
                cnt_reset += 1
                if cnt_reset >= MAX_RESET:
                    print(f"[{item_id}] failed stage {i} after {MAX_RESET} resets. giving up.")
                    return False
                else:
                    print(f"[{item_id}] failed stage {i}. resetting (attempt {cnt_reset}).")
                    for stage in self._stages[:i]:
                        stage.cache.delete(item_id)
                    i = 0
        return True

    def _execute_stage(self, item_id: str, stage_id: int) -> bool:
        storer = self._stages[stage_id].cache
        processor = self._stages[stage_id].processor

        if storer.has(item_id):
            return True

        item_data = self._loader.load(item_id)
        input_data = {}
        for input_type, input_source in self._stages[stage_id].inputs:
            input_name = camel_to_snake(input_type)
            if input_source == AnnotationSource.HUMAN:
                input_data[input_name] = item_data[input_name]
            else:
                input_cache = self._stages[stage_id].input_caches[input_type]
                input_data[input_name] = input_cache.load(item_id)
        inputs = {"id": item_id, **input_data}

        output_dir = storer.output_filepath(item_id).parent  # <stage>/<item_id>/
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as logger:
            start_time = time.perf_counter()
            try:
                output_data = processor(inputs, logger, output_dir=output_dir)
            except (KeyboardInterrupt, InterruptedError):
                raise
            except Exception as e:
                print(f"Stage {stage_id} raised {type(e).__name__}: {e}", file=logger, flush=True)
                print(f"Stage {stage_id} raised {type(e).__name__}: {e}")
                output_data = None
            end_time = time.perf_counter()
            print(f"TIME: {end_time - start_time:.6f}s", file=logger, flush=True)

        if output_data is None:
            return False

        with self._shutdown_flag.lock:
            storer.store(item_id, output_data)
        return True


