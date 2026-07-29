"""Headlessly exercise agentflow/viewer.py against an example project, to see
whether the Streamlit viewer actually renders a pipeline's stages.

Uses Streamlit's AppTest harness, so no browser is involved: the script runs,
and we inspect the resulting elements (warnings, errors, images, json blocks)
for every stage of the config.

Usage:
    # from the repo root
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_check_viewer.py
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_check_viewer.py \
        --project examples/captioning --config two_stage.yaml --cwd examples/captioning
"""

import argparse
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER = REPO_ROOT / "agentflow" / "viewer.py"

# Emulate `streamlit run`, which puts the *script's* directory on sys.path and
# not the repo root. Do NOT add REPO_ROOT here: doing so hides import failures
# that the real command hits.
sys.path.insert(0, str(VIEWER.parent))


def run(configs_dir: str, output_dir: str, stage_index: int, config_name: str | None = None) -> AppTest:
    """Run the viewer with the given CLI args, selecting one config and stage."""
    sys.argv = ["viewer.py", configs_dir, output_dir]
    at = AppTest.from_file(str(VIEWER), default_timeout=60)
    at.run()
    if config_name and at.selectbox:
        # the only selectbox is the config picker, shown when the dir holds several
        at.selectbox[0].set_value(config_name).run()
    if stage_index:
        # number_inputs are [Example, Stage]
        if len(at.number_input) < 2:
            return None
        stage_picker = at.number_input[1]
        if stage_index > stage_picker.max:
            return None  # config has fewer stages than requested
        stage_picker.set_value(stage_index).run()
    return at


def report(at: AppTest, label: str) -> None:
    print(f"\n--- {label}")
    if at.exception:
        for e in at.exception:
            print(f"  EXCEPTION: {e.value}")
    for e in at.error:
        print(f"  st.error:   {e.value}")
    for w in at.warning:
        print(f"  st.warning: {str(w.value)[:120]}")
    print(f"  json blocks: {len(at.json)}   captions: {[c.value for c in at.caption][:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="examples/elementary_math")
    ap.add_argument("--cwd", default=None, help="directory to run from (default: the project dir)")
    ap.add_argument("--output-dir", default=None, help="output root as the viewer should see it")
    ap.add_argument("--stages", type=int, default=4, help="how many stages to step through")
    ap.add_argument("--config", default=None, help="config file name to select, when the dir holds several")
    args = ap.parse_args()

    project = (REPO_ROOT / args.project).resolve()
    cwd = Path(args.cwd).resolve() if args.cwd else project
    output_dir = args.output_dir or os.path.relpath(REPO_ROOT / "output", cwd)

    os.chdir(cwd)
    print(f"cwd={cwd}")
    print(f"configs={os.path.relpath(project / 'configs', cwd)}  output={output_dir}")

    for stage_index in range(args.stages):
        at = run(os.path.relpath(project / "configs", cwd), output_dir, stage_index, args.config)
        if at is None:
            break
        report(at, f"stage index {stage_index}")


if __name__ == "__main__":
    main()
