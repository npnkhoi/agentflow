from pathlib import Path
from typing import Callable
from agentflow.typing.config import DemoConfig, DemoPoolConfig
from agentflow.const import DemoSelect
from agentflow.loaders import DataItemLoader
import random


class DemoPool:
    """
    Stores and selects demonstrations for few-shot prompting.

    The pool may overlap with the test set; self-exclusion is always applied so an
    item is never selected as its own demo.

    Built-in selection strategies are `random`, `similar`, and `diverse`. Strategies
    that need domain knowledge (e.g. "the items with the longest annotation") are
    supplied by the application:

        DemoPool.register_strategy("longest", my_handler)

    where ``my_handler(pool: DemoPool, inputs: dict) -> list[dict]``.
    """

    _strategies: dict[str, Callable[["DemoPool", dict], list[dict]]] = {}

    @classmethod
    def register_strategy(cls, name: str, handler: Callable[["DemoPool", dict], list[dict]]) -> None:
        """Register a demo-selection strategy under the name used in `demo.select`."""
        cls._strategies[name] = handler

    def __init__(self, config: DemoConfig, pool_config: DemoPoolConfig):
        self._config = config
        self._loader = DataItemLoader(path=Path(pool_config.source), image_dir=pool_config.image_dir)
        self._item_ids = self._loader.item_ids
        if len(self._item_ids) < config.shots:
            raise ValueError(
                f"Pool size {len(self._item_ids)} is not enough for shots={config.shots}."
            )

        if self._config.select in (DemoSelect.SIMILAR, DemoSelect.DIVERSE):
            self._calc_embeddings()
        # `diverse` ignores the query item, so the selection is made once here
        # rather than per item.
        self._fixed_demos: list[dict] | None = (
            self._calc_diverse() if self._config.select == DemoSelect.DIVERSE else None
        )

    @property
    def config(self) -> DemoConfig:
        return self._config

    @property
    def loader(self) -> DataItemLoader:
        return self._loader

    @property
    def item_ids(self) -> list[str]:
        return list(self._item_ids)

    def items_from_ids(self, ids: list[str]) -> list[dict]:
        """Public accessor used by registered strategies."""
        return self._items_from_ids(ids)

    def _items_from_ids(self, ids: list[str]) -> list[dict]:
        return [self._loader.load(item_id) for item_id in ids]

    def _random(self, id: str) -> list[dict]:
        candidates = [iid for iid in self._item_ids if iid != id]
        if len(candidates) < self._config.shots:
            raise ValueError(
                f"Not enough candidates after self-exclusion: "
                f"{len(candidates)} available, {self._config.shots} needed."
            )
        ids = random.sample(candidates, self._config.shots)
        return self._items_from_ids(ids)

    def _calc_embeddings(self):
        from PIL import Image as PILImage
        from transformers import CLIPProcessor, CLIPModel
        import torch
        from tqdm import tqdm

        self._embeddings = {}
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map="auto")
        for item_id in tqdm(self._loader.item_ids, "Calculating image embeddings"):
            item = self._loader.load(item_id)
            with PILImage.open(item["image"]) as img:
                inputs = self._clip_processor(images=img, return_tensors="pt", padding=True)
                with torch.no_grad():
                    self._embeddings[item_id] = self._clip_model.get_image_features(**inputs)

    def _similar(self, id: str, image_path: Path | str | None) -> list[dict]:
        from PIL import Image as PILImage
        import torch
        import torch.nn.functional as F

        with PILImage.open(image_path) as img:
            inputs = self._clip_processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                target = self._clip_model.get_image_features(**inputs)

        dists = {
            mid: F.cosine_similarity(target, emb, dim=-1).item()
            for mid, emb in self._embeddings.items()
            if mid != id
        }
        ids = [k for k, _ in sorted(dists.items(), key=lambda x: x[1], reverse=True)][: self._config.shots]
        return self._items_from_ids(ids)

    def _calc_diverse(self) -> list[dict]:
        """Pick a spread-out subset of the pool by greedy furthest-first traversal
        over CLIP image embeddings.

        Each round takes the item whose greatest similarity to anything already
        picked is smallest — i.e. the item least like the current selection. The
        query item plays no part, so the same demos serve every item.
        """
        import torch.nn.functional as F
        from tqdm import tqdm

        remaining = list(self._item_ids)
        # Similarity of each remaining item to the closest already-selected item.
        # -inf until something is selected, which makes the first pick arbitrary.
        closest_sim: dict[str, float] = {item_id: -float("inf") for item_id in remaining}
        selected: list[str] = []

        for _ in tqdm(range(self._config.shots), desc="Selecting diverse demos"):
            pick = min(closest_sim, key=lambda k: closest_sim[k])
            remaining.remove(pick)
            closest_sim.pop(pick)
            selected.append(pick)
            for item_id in remaining:
                sim = F.cosine_similarity(
                    self._embeddings[item_id], self._embeddings[pick], dim=-1
                ).item()
                closest_sim[item_id] = max(closest_sim[item_id], sim)

        return self._items_from_ids(selected)

    def demos(self, inputs: dict) -> list[dict]:
        select = self._config.select
        if select == DemoSelect.RANDOM:
            return self._random(inputs["id"])
        elif select == DemoSelect.SIMILAR:
            return self._similar(inputs["id"], inputs["image"])
        elif select == DemoSelect.DIVERSE:
            assert self._fixed_demos is not None
            return self._fixed_demos
        elif str(getattr(select, "value", select)) in self._strategies:
            return self._strategies[str(getattr(select, "value", select))](self, inputs)
        else:
            raise NotImplementedError(
                f"Demo select '{select}' is not implemented and not registered. "
                f"Call DemoPool.register_strategy('{select}', handler) before use."
            )
