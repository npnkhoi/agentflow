"""Render the AgentFlow example pipeline (elementary-math workflow) as a graphviz figure.

Illustrates the 5-step workflow from the README's "Elementary Math Problems"
example: 4 inference steps take a photo of a math question through captioning,
calculation, and answering (with the calculator stage outsourcing arithmetic),
and a final evaluation step compares the answer against ground truth to produce
a binary verdict.

Usage:
    # graphviz (dot binary) + python-graphviz must be installed in the conda env
    conda run -n ds python experimental/e0728_workflow_graph.py
    # -> writes docs/assets/e0728_workflow_graph.svg
"""

from pathlib import Path

import graphviz

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "assets"
OUT_NAME = Path(__file__).with_suffix("").name  # e0728_workflow_graph

# (name, label, kind) — kind drives styling: "input" data, "artifact" produced
# by an inference stage, "output" final inference answer, "eval" evaluation output.
NODES = [
    ("Image", "Image\n(photo of question)", "input"),
    ("QuestionText", "QuestionText", "artifact"),
    ("Calculations", "Calculations", "artifact"),
    ("CalculatedNumbers", "CalculatedNumbers", "artifact"),
    ("Answer", "Answer\n(float)", "output"),
    ("GroundTruthAnswer", "GroundTruthAnswer", "input"),
    ("Verdict", "Verdict", "eval"),
]

# (src, dst, step-label)
EDGES = [
    ("Image", "QuestionText", "Step 1"),
    ("QuestionText", "Calculations", "Step 2"),
    ("Calculations", "CalculatedNumbers", "Step 3"),
    ("QuestionText", "Answer", "Step 4"),
    ("Calculations", "Answer", "Step 4"),
    ("CalculatedNumbers", "Answer", "Step 4"),
    ("Answer", "Verdict", "Step 5"),
    ("GroundTruthAnswer", "Verdict", "Step 5"),
]

STYLE = {
    "input": dict(shape="box", style="filled,rounded", fillcolor="#e3f2fd", color="#1565c0"),
    "artifact": dict(shape="box", style="filled,rounded", fillcolor="#f1f8e9", color="#558b2f"),
    "output": dict(shape="box", style="filled,rounded", fillcolor="#fff3e0", color="#e65100"),
    "eval": dict(shape="box", style="filled,rounded", fillcolor="#f3e5f5", color="#6a1b9a"),
}


def build() -> graphviz.Digraph:
    g = graphviz.Digraph("agentflow_math_workflow", format="svg")
    g.attr(rankdir="LR", fontname="Helvetica", labelloc="t",
            label="AgentFlow — Elementary Math Workflow (inference: steps 1-4, evaluation: step 5)")
    g.attr("node", fontname="Helvetica", penwidth="1.5")
    g.attr("edge", fontname="Helvetica", fontsize="10", color="#616161")

    for name, label, kind in NODES:
        g.node(name, label, **STYLE[kind])
    for src, dst, step in EDGES:
        g.edge(src, dst, label=step)
    return g


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = build()
    svg_path = g.render(filename=OUT_NAME, directory=str(OUT_DIR),
                        format="svg", cleanup=True)
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
