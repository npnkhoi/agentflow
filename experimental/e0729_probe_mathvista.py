"""Probe MathVista testmini for elementary-school arithmetic items whose *image*
carries the whole question — the shape the README's elementary-math example needs
(Step 1 is Image -> QuestionText, so the question must be legible in the photo).

Prints the distribution of grade / task / answer_type / source, then dumps a few
candidate rows so we can eyeball whether the image is a rendered word problem or
just a diagram.

Usage:
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_probe_mathvista.py
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_probe_mathvista.py --dump-images 6
"""

import argparse
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("/tmp/claude-1005/-home-khoi-Code-agentflow/6555a7d5-edfd-46f0-b279-81bab620e793/scratchpad/mathvista_probe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-images", type=int, default=0, help="save N candidate images for inspection")
    args = ap.parse_args()

    ds = load_dataset("AI4Math/MathVista", split="testmini")
    print(f"testmini rows: {len(ds)}")
    print(f"columns: {ds.column_names}\n")

    meta = [r for r in ds["metadata"]]
    print("grade:", Counter(m["grade"] for m in meta).most_common())
    print("task:", Counter(m["task"] for m in meta).most_common())
    print("context:", Counter(m["context"] for m in meta).most_common(12))
    print("source:", Counter(m["source"] for m in meta).most_common(15))
    print("answer_type:", Counter(ds["answer_type"]).most_common())
    print("question_type:", Counter(ds["question_type"]).most_common())

    # Candidates: elementary school, free-form (not multiple choice), numeric answer.
    cand = [
        i
        for i in range(len(ds))
        if meta[i]["grade"] == "elementary school"
        and ds[i]["question_type"] == "free_form"
        and ds[i]["answer_type"] in ("integer", "float")
    ]
    print(f"\nelementary + free_form + numeric: {len(cand)}")
    print("  their sources:", Counter(meta[i]["source"] for i in cand).most_common())
    print("  their tasks:", Counter(meta[i]["task"] for i in cand).most_common())

    for i in cand[:8]:
        row = ds[i]
        print(f"\n--- pid={row['pid']} source={meta[i]['source']} task={meta[i]['task']}")
        print(f"    question: {row['question']!r}")
        print(f"    answer:   {row['answer']!r}  unit={row['unit']!r}")
        print(f"    image:    {row['image']}  size={row['decoded_image'].size}")

    if args.dump_images:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for i in cand[: args.dump_images]:
            row = ds[i]
            path = OUT_DIR / f"{row['pid']}.png"
            row["decoded_image"].convert("RGB").save(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
