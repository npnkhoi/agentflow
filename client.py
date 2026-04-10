
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from typing import TextIO

from tqdm import tqdm

from agentflow.typing.config import Config
from agentflow.pipeline import Pipeline

try:
    import wandb
except Exception:
    wandb = None
    
class Client:
    def __init__(self, extra_processors: list = [], extra_types: list = [], prompt_dir: str = 'prompts/', demo_pools: list[tuple[str, str]] = []):
        self.extra_processors = extra_processors
        self.extra_types = extra_types
        self.prompt_dir = prompt_dir
        self.demo_pools = demo_pools


    def run(self, config_file: TextIO, no_cache: bool = False):
        import yaml
        with config_file:
            cfg = Config.model_validate(yaml.safe_load(config_file))
        # check config
        # 1. filename and config name consistency
        name_from_filename = Path(config_file.name).stem
        if name_from_filename != cfg.name:
            print(f"Config names are inconsistent: {cfg.name}, filename: {name_from_filename}", flush=True)
            verdict = input("Go ahead? (y/n)")
            if verdict != "y":
                return

        if cfg.wandb_enabled:
            if wandb is None:
                print("wandb not available; continuing with wandb disabled")
                cfg.wandb_enabled = False
            else:
                try:
                    wandb.init(project=cfg.name, config=cfg.model_dump(), name=cfg.name)
                except Exception as e:
                    print(
                        f"wandb initialization failed ({e}); continuing with wandb disabled"
                    )
                    cfg.wandb_enabled = False

        # get pipeline
        pipeline = Pipeline(cfg, prompt_dir=self.prompt_dir)

        # determine item_ids to run
        if (cfg.include is not None) and (cfg.exclude is not None):
            raise ValueError("Cannot specify both include and exclude")
        elif cfg.include is not None:
            item_ids = list(cfg.include)
        elif cfg.exclude is not None:
            exclude_set = set(cfg.exclude)
            item_ids = [iid for iid in pipeline.item_ids if iid not in exclude_set]
        elif cfg.include_first is not None:
            item_ids = pipeline.item_ids[: cfg.include_first]
        else:
            item_ids = pipeline.item_ids

        # stats (thread-safe)
        stats = {"success": 0, "fail": 0, "total": 0}
        stats_lock = threading.Lock()

        def run_one(item_id: str) -> None:
            result = pipeline.execute(item_id)
            with stats_lock:
                stats["total"] += 1
                if result:
                    stats["success"] += 1
                else:
                    stats["fail"] += 1
                s = dict(stats)
            if cfg.wandb_enabled and wandb is not None:
                try:
                    wandb.log(s)
                except Exception:
                    pass

        executor = ThreadPoolExecutor(max_workers=cfg.n_parallel)
        futures = {executor.submit(run_one, iid): iid for iid in item_ids}
        try:
            with tqdm(as_completed(futures), total=len(futures), desc="Run") as pbar:
                for future in pbar:
                    future.result()  # re-raise any unexpected exceptions
                    with stats_lock:
                        s = dict(stats)
                    pbar.set_postfix(success=s["success"], fail=s["fail"])
        except (KeyboardInterrupt, InterruptedError):
            print("\nInterrupted — cancelling pending tasks.", flush=True)
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        print(
            f"\nDone. success={stats['success']}, fail={stats['fail']}, total={stats['total']}",
            flush=True,
        )

        if cfg.wandb_enabled and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass