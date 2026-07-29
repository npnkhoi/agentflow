"""Render the AgentFlow example pipeline (elementary-math workflow) as a graphviz figure.

Illustrates the 4-step workflow from the README's "Elementary Math Problems"
example, as implemented in examples/elementary_math/: 3 inference steps take a
question figure plus its text through calculation planning, exact evaluation by
a calculator, and answering; a final evaluation step compares the answer against
ground truth to produce a binary verdict. Steps 2 and 4 are deterministic
processors — only steps 1 and 3 call a model.

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
    ("Image", "Image\n(question figure)", "input"),
    ("QuestionText", "QuestionText", "input"),
    ("Calculations", "Calculations", "artifact"),
    ("CalculatedNumbers", "CalculatedNumbers\n(calculator)", "artifact"),
    ("Answer", "Answer\n(float)", "output"),
    ("GroundTruthAnswer", "GroundTruthAnswer", "input"),
    ("Verdict", "Verdict\n(exact comparison)", "eval"),
]

# Stages produced by a deterministic processor rather than a model call. Drawn
# with a dashed border, since "does this step call an LLM?" is the distinction
# the workflow is built around.
DETERMINISTIC = {"CalculatedNumbers", "Verdict"}

# (src, dst, step-label)
EDGES = [
    ("Image", "Calculations", "Step 1"),
    ("QuestionText", "Calculations", "Step 1"),
    ("Calculations", "CalculatedNumbers", "Step 2"),
    ("Image", "Answer", "Step 3"),
    ("QuestionText", "Answer", "Step 3"),
    ("Calculations", "Answer", "Step 3"),
    ("CalculatedNumbers", "Answer", "Step 3"),
    ("Answer", "Verdict", "Step 4"),
    ("GroundTruthAnswer", "Verdict", "Step 4"),
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
            label="AgentFlow — Elementary Math Workflow (inference: steps 1-3, evaluation: step 4)\n"
                  "solid = LLM stage, dashed = deterministic processor")
    g.attr("node", fontname="Helvetica", penwidth="1.5")
    g.attr("edge", fontname="Helvetica", fontsize="10", color="#616161")

    for name, label, kind in NODES:
        style = dict(STYLE[kind])
        if name in DETERMINISTIC:
            style["style"] += ",dashed"
        g.node(name, label, **style)
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
