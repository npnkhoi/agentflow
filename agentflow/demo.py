from pathlib import Path
from agentflow.typing.config import DemoConfig, DemoPoolConfig
from agentflow.const import DemoSelect
from agentflow.loaders import DataItemLoader
import random


class DemoPool:
    """
    Stores and selects demonstrations for few-shot prompting.

    The pool may overlap with the test set; self-exclusion is always applied so an
    item is never selected as its own demo.
    """

    def __init__(self, config: DemoConfig, pool_config: DemoPoolConfig):
        self._config = config
        self._loader = DataItemLoader(path=Path(pool_config.source), image_dir=pool_config.image_dir)
        self._item_ids = self._loader.item_ids
        if len(self._item_ids) < config.shots:
            raise ValueError(
                f"Pool size {len(self._item_ids)} is not enough for shots={config.shots}."
            )

        if self._config.select == DemoSelect.SIMILAR:
            self._calc_embeddings()

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

    def demos(self, inputs: dict) -> list[dict]:
        if self._config.select == DemoSelect.RANDOM:
            return self._random(inputs["id"])
        elif self._config.select == DemoSelect.SIMILAR:
            return self._similar(inputs["id"], inputs["image"])
        else:
            raise NotImplementedError(f"Demo select '{self._config.select}' not implemented")
