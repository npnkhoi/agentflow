"""Streamlit app for browsing agentflow pipeline outputs.

Usage:
    streamlit run agentflow/viewer.py -- <configs_dir> <output_dir>
"""
import argparse
import json
from pathlib import Path

import streamlit as st
import yaml

from agentflow.typing.config import Config
from agentflow.util import camel_to_snake


def _parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Browse agentflow pipeline outputs.")
    parser.add_argument("configs_dir", type=Path, help="Directory containing YAML config files")
    parser.add_argument("output_dir", type=Path, help="Pipeline output root directory")
    args = parser.parse_args()
    return args.configs_dir, args.output_dir


def _select_config(configs_dir: Path) -> Config:
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

    return Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))


def main():
    st.set_page_config(layout="wide", page_title="AgentFlow Viewer")
    configs_dir, output_root = _parse_args()

    cfg = _select_config(configs_dir)

    loader_source = Path(cfg.loader.source)
    image_dir = Path((cfg.loader.kwargs or {})["image_dir"])

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

    # --- Stage selector ---
    stage_names = [s["name"] for s in stages]
    stage_name = st.selectbox("Stage", stage_names)
    stage = next(s for s in stages if s["name"] == stage_name)
    stage_output_dir = output_root / cfg.name / stage_name

    n_done = sum(
        1 for iid in item_ids if (stage_output_dir / iid / "output.json").exists()
    )
    st.caption(f"{len(item_ids)} items · {n_done} with output")

    # --- Item navigator ---
    col_prev, col_idx, col_next = st.columns([1, 8, 1])
    if "item_idx" not in st.session_state:
        st.session_state.item_idx = 0

    with col_prev:
        st.write("")
        if st.button("◀", use_container_width=True):
            st.session_state.item_idx = max(0, st.session_state.item_idx - 1)

    with col_next:
        st.write("")
        if st.button("▶", use_container_width=True):
            st.session_state.item_idx = min(len(item_ids) - 1, st.session_state.item_idx + 1)

    with col_idx:
        idx = st.slider("Item", 0, len(item_ids) - 1, st.session_state.item_idx, label_visibility="collapsed")
        st.session_state.item_idx = idx

    item_id = item_ids[idx]
    st.caption(f"ID: `{item_id}`")
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
                cache_file = output_root / input_type / item_id / "output.json"
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
