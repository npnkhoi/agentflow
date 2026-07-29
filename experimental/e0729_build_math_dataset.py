"""Build the `examples/elementary_math` dataset from MathVista testmini.

Selects elementary-school items with a free-form numeric answer (sources:
CLEVR-Math, IconQA, TabMWP, IQTest), saves each item's image, and writes an
items.json carrying the question text and the ground-truth answer:

    examples/elementary_math/data/items.json
    examples/elementary_math/data/images/<id>.png

Each item's `data` provides the three human-annotated inputs the pipeline
consumes: `image` (the diagram/table), `question_text`, `ground_truth_answer`.

Usage:
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_build_math_dataset.py
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_build_math_dataset.py --per-source 3
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "examples" / "elementary_math"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=2, help="items to take per MathVista source dataset")
    ap.add_argument("--max-width", type=int, default=768, help="downscale wide images to keep prompts small")
    args = ap.parse_args()

    ds = load_dataset("AI4Math/MathVista", split="testmini")

    picked_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        meta = row["metadata"]
        if meta["grade"] != "elementary school":
            continue
        if row["question_type"] != "free_form" or row["answer_type"] not in ("integer", "float"):
            continue
        source = meta["source"]
        if len(picked_by_source[source]) >= args.per_source:
            continue
        picked_by_source[source].append(row)

    image_dir = PROJECT_DIR / "data" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for source in sorted(picked_by_source):
        for row in picked_by_source[source]:
            item_id = f"{source.lower().replace('-', '_')}_{row['pid']}"
            image = row["decoded_image"].convert("RGB")
            if image.width > args.max_width:
                ratio = args.max_width / image.width
                image = image.resize((args.max_width, round(image.height * ratio)))
            image.save(image_dir / f"{item_id}.png")
            items.append({
                "id": item_id,
                "data": {
                    "image": f"{item_id}.png",
                    "question_text": row["question"],
                    "ground_truth_answer": float(row["answer"]),
                    "source": source,
                },
            })

    items_path = PROJECT_DIR / "data" / "items.json"
    items_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {items_path} ({len(items)} items)")
    for source in sorted(picked_by_source):
        print(f"  {source}: {len(picked_by_source[source])}")


if __name__ == "__main__":
    main()
