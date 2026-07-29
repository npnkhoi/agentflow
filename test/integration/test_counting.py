"""Integration tests for the ``examples/counting`` project (local vision server).

Counting task over CountBench images with a demo pool. Assumes an
OpenAI-compatible vision server at http://0.0.0.0:8000 (see
``test_captioning`` for how to start one).

Dataset: 2 test items (count_0, count_1).
Demo pool: all 5 items (count_0..4); count_2..4 carry a pre-computed CountOutput.
Self-exclusion ensures each item does not appear in its own demo list.

Run from the repo root:
    pytest test/integration/test_counting.py -v
"""

import json

from test.integration.harness import make_runner, output_path

run_pipeline = make_runner("counting")


class TestCountWithDemos:
    def test_runs_without_error(self):
        run_pipeline("count_with_demos.yaml")

    def test_two_output_files_created(self):
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        items = list(out.glob("*/output.json"))
        assert len(items) == 2, f"Expected 2 output files, got {len(items)}"

    def test_output_is_valid_count(self):
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        for json_file in out.glob("*/output.json"):
            data = json.loads(json_file.read_text())
            assert "count" in data, f"CountOutput missing 'count' in {json_file}"
            assert isinstance(data["count"], int), (
                f"'count' must be an int, got {type(data['count'])}"
            )
            assert data["count"] > 0, f"'count' must be positive, got {data['count']}"

    def test_demos_json_lists_two_ids(self):
        """Each test item must have exactly 2 demo IDs (shots=2)."""
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        demos_files = list(out.glob("*/demos.json"))
        assert len(demos_files) == 2, (
            f"Expected demos.json per item, found {len(demos_files)}"
        )
        for demos_file in demos_files:
            ids = json.loads(demos_file.read_text())
            assert len(ids) == 2, (
                f"Expected 2 demo ids (shots=2), got {len(ids)}: {ids}"
            )

    def test_demos_exclude_self(self):
        """Self-exclusion: each item must not appear in its own demo list."""
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        for item_dir in out.iterdir():
            if not item_dir.is_dir():
                continue
            demos_file = item_dir / "demos.json"
            if not demos_file.exists():
                continue
            item_id = item_dir.name
            demo_ids = json.loads(demos_file.read_text())
            assert item_id not in demo_ids, (
                f"Item '{item_id}' selected itself as a demo"
            )

    def test_demos_are_from_pool(self):
        """All selected demo IDs must come from the 5-item pool (count_0..4)."""
        run_pipeline("count_with_demos.yaml")
        out = output_path("test_count_with_demos") / "CountOutput"
        pool_ids = {"count_0", "count_1", "count_2", "count_3", "count_4"}
        for demos_file in out.glob("*/demos.json"):
            ids = json.loads(demos_file.read_text())
            for demo_id in ids:
                assert demo_id in pool_ids, (
                    f"Demo ID '{demo_id}' is not from the pool {pool_ids}"
                )
