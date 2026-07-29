"""Streamlit app for browsing agentflow pipeline outputs.

Usage:
    streamlit run agentflow/viewer.py -- <configs_dir> <output_dir>
"""
import argparse
import json
import sys
from pathlib import Path

# `streamlit run agentflow/viewer.py` puts *this file's* directory on sys.path
# rather than the repo root. That breaks the app two ways: `import agentflow`
# finds nothing, and `agentflow/typing/` shadows the standard library's `typing`
# for every later import (streamlit's own included). Swap that entry for the
# repo root before importing anything else, so the documented command works from
# a plain checkout.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from agentflow.typing.config import Config  # noqa: E402
from agentflow.util import camel_to_snake  # noqa: E402


def _parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Browse agentflow pipeline outputs.")
    parser.add_argument("configs_dir", type=Path, help="Directory containing YAML config files")
    parser.add_argument("output_dir", type=Path, help="Pipeline output root directory")
    args = parser.parse_args()
    return args.configs_dir, args.output_dir


def _select_config(configs_dir: Path) -> tuple[Config, Path]:
    yaml_files = sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))
    if not yaml_files:
        st.error(f"No YAML config files found in {configs_dir}")
        st.stop()

    if len(yaml_files) == 1:
        config_path = yaml_files[0]
    else:
        names = [f.name for f in yaml_files]
        selected = st.selectbox("Config", names, key="config_select")
        config_path = configs_dir / selected

    cfg = Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    return cfg, config_path


def main():
    st.set_page_config(layout="wide", page_title="AgentFlow Viewer")
    configs_dir, output_root = _parse_args()

    cfg, config_path = _select_config(configs_dir)

    # Example projects write data paths relative to the project dir (the parent
    # of configs/), so resolve them against it rather than against the cwd.
    project_dir = config_path.resolve().parent.parent

    def _resolve(path: "str | Path") -> Path:
        path = Path(path)
        return path if path.is_absolute() else project_dir / path

    loader_source = _resolve(cfg.loader.source)
    image_dir = _resolve((cfg.loader.kwargs or {})["image_dir"])

    stages = [
        {
            "name": stage.output,
            "inputs": [{"type": inp[0], "source": inp[1].value} for inp in stage.inputs],
        }
        for stage in cfg.stages
    ]

    st.title(f"Pipeline: {cfg.name}")

    if not loader_source.exists():
        st.error(f"Loader source not found: {loader_source}")
        st.stop()

    items = json.loads(loader_source.read_text(encoding="utf-8"))
    item_ids = [item["id"] for item in items]
    item_data = {item["id"]: item["data"] for item in items}

    # --- Pickers: example on the left, stage on the right ---
    stage_names = [s["name"] for s in stages]
    col_item, col_stage = st.columns(2)

    with col_item:
        idx = st.number_input(
            "Example",
            min_value=0,
            max_value=len(item_ids) - 1,
            value=0,
            step=1,
            key="item_idx",
        )
        item_id = item_ids[idx]
        st.caption(f"`{item_id}` · {idx + 1} of {len(item_ids)}")

    with col_stage:
        stage_idx = st.number_input(
            "Stage",
            min_value=0,
            max_value=len(stage_names) - 1,
            value=0,
            step=1,
            key="stage_idx",
        )
        stage = stages[stage_idx]
        stage_name = stage["name"]
        st.caption(f"`{stage_name}` · {stage_idx + 1} of {len(stage_names)}")

    stage_output_dir = output_root / cfg.name / stage_name
    n_done = sum(
        1 for iid in item_ids if (stage_output_dir / iid / "output.json").exists()
    )
    st.caption(f"{len(item_ids)} items · {n_done} with output at this stage")
    st.divider()

    # --- Two-column display ---
    left, right = st.columns(2)

    with left:
        st.subheader("Inputs")
        for inp in stage["inputs"]:
            input_type = inp["type"]
            input_source = inp["source"]
            st.markdown(f"**{input_type}** *(from {input_source})*")

            if input_source == "human":
                field = camel_to_snake(input_type)
                val = item_data.get(item_id, {}).get(field)
                if val is None:
                    st.warning("Not found in loader data.")
                elif input_type == "Image":
                    img_path = image_dir / Path(str(val)).name
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.text(str(val))
                elif isinstance(val, (dict, list)):
                    st.json(val, expanded=True)
                else:
                    st.text(str(val))
            else:
                cache_file = output_root / cfg.name / input_type / item_id / "output.json"
                if cache_file.exists():
                    st.json(json.loads(cache_file.read_text(encoding="utf-8")), expanded=True)
                else:
                    st.warning(f"Stage output not found: {cache_file}")

    with right:
        st.subheader(f"Output: {stage_name}")
        output_file = stage_output_dir / item_id / "output.json"
        if output_file.exists():
            st.json(json.loads(output_file.read_text(encoding="utf-8")), expanded=True)
        else:
            st.warning("Not executed, failed, or cached from a previous run.")
            st.warning(output_file)
            log_file = stage_output_dir / item_id / "run.log"
            if log_file.exists():
                log_text = log_file.read_text(encoding="utf-8").strip()
                if log_text:
                    st.subheader("Log")
                    st.code(log_text, language="text", wrap_lines=True)


if __name__ == "__main__":
    main()
