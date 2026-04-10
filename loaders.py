from typing import Protocol
from pydantic import BaseModel
from agentflow.typing.output import DataItem
import json
from pathlib import Path


class Loader(Protocol):
    def load(self, item_id: str) -> dict | None:
        """Load a data item by its id."""
        pass

    @property
    def item_ids(self) -> list[str]:
        """Return available item ids in order."""
        pass


class DataItemLoader(Loader):
    """
    Expects the source file to have the following schema:
    [{"id": str, "data": dict}, ...]
    """
    def __init__(self, path: Path, image_dir: str):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        image_dir = Path(image_dir)
        items = []
        for item in data:
            item["data"]["image"] = image_dir / item["data"]["image"].split("/")[-1]
            item["data"]["id"] = item["id"]
            items.append(DataItem.model_validate(item))

        self._items: dict[str, dict] = {item.id: item.data for item in items}
        self._order: list[str] = [item.id for item in items]

    def load(self, item_id: str) -> dict | None:
        return self._items.get(item_id)

    @property
    def item_ids(self) -> list[str]:
        return self._order


class Cache:
    def __init__(self, path: Path, datatype: type[BaseModel]):
        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._datatype = datatype

    def load(self, item_id: str) -> BaseModel | None:
        path = self.output_filepath(item_id)
        if not path.is_file():
            return None
        return self._datatype.model_validate_json(path.read_text(encoding="utf-8"))

    @property
    def item_ids(self) -> list[str]:
        return [p.parent.name for p in self._path.glob("*/output.json")]

    def has(self, item_id: str) -> bool:
        return self.output_filepath(item_id).is_file()

    def store(self, item_id: str, data: BaseModel):
        path = self.output_filepath(item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")

    def delete(self, item_id: str):
        path = self.output_filepath(item_id)
        if path.is_file():
            path.unlink()

    def output_filepath(self, item_id: str) -> Path:
        return self._path / item_id / "output.json"