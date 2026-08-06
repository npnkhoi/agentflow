"""Unit tests for DemoPool selection (no LLM server or CLIP weights required).

Only the strategies that need no model weights are covered here: `random` and
registered custom strategies. `similar` and `diverse` both build CLIP embeddings,
so they belong in an integration test.

Usage:
    pytest test/unit/test_demo_pool.py
"""

from pathlib import Path

import pytest

from agentflow.demo import DemoPool
from agentflow.typing.config import DemoConfig, DemoPoolConfig

POOL = DemoPoolConfig(
    source="examples/captioning/data/items.json",
    image_dir="examples/captioning/data/images",
)


def make_pool(select: str, shots: int = 1) -> DemoPool:
    return DemoPool(DemoConfig(pool="main", select=select, shots=shots), POOL)


class TestRandom:
    def test_excludes_the_query_item(self):
        pool = make_pool("random")
        for item_id in pool.item_ids:
            demos = pool.demos({"id": item_id})
            assert len(demos) == 1
            assert demos[0]["id"] != item_id


class TestCustomStrategy:
    def test_registered_strategy_is_dispatched(self):
        calls = []

        def first_item(pool: DemoPool, inputs: dict) -> list[dict]:
            calls.append(inputs["id"])
            return pool.items_from_ids(pool.item_ids[: pool.config.shots])

        DemoPool.register_strategy("first_item", first_item)
        pool = make_pool("first_item")

        demos = pool.demos({"id": "whatever"})

        assert calls == ["whatever"]
        assert [d["id"] for d in demos] == pool.item_ids[:1]

    def test_unregistered_strategy_raises(self):
        pool = make_pool("longest")  # declared in DemoSelect, but nobody registered it
        with pytest.raises(NotImplementedError, match="register_strategy"):
            pool.demos({"id": "x"})

    def test_pool_exposes_accessors_for_strategies(self):
        pool = make_pool("random", shots=2)
        assert pool.config.shots == 2
        assert len(pool.item_ids) >= 2
        assert Path(pool.loader.load(pool.item_ids[0])["image"]).name.endswith(".png")
        assert [d["id"] for d in pool.items_from_ids(pool.item_ids[:1])] == pool.item_ids[:1]
