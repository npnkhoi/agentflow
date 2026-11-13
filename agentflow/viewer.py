"""Streamlit app for browsing agentflow pipeline outputs.

Usage:
    agentflow view [configs_dir_or_file] [output_dir]
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
from agentflow.viewer_images import resolve_image, show_image, show_value  # noqa: E402


def _parse_args() -> tuple[Path, Path, Path | None]:
    parser = argparse.ArgumentParser(description="Browse agentflow pipeline outputs.")
    parser.add_argument("configs_dir", type=Path, help="YAML config file or directory containing config files")
    parser.add_argument("output_dir", type=Path, help="Pipeline output root directory")
    parser.add_argument("--project-dir", type=Path, help="Base directory for relative loader and image paths")
    args = parser.parse_args()
    return args.configs_dir, args.output_dir, args.project_dir


def _select_config(configs_dir: Path) -> tuple[Config, Path]:
    yaml_files = ([configs_dir] if configs_dir.is_file() else
                  sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml")))
    if not yaml_files:
        st.error(f"No YAML config files found in {configs_dir}")
        st.stop()

    if len(yaml_files) == 1:
        config_path = yaml_files[0]
    else:
        names = [f.name for f in yaml_files]
        selected = st.selectbox("Config", names, key="config_select")
        config_path = yaml_files[names.index(selected)]

    cfg = Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    return cfg, config_path


def main():
    st.set_page_config(layout="wide", page_title="AgentFlow Viewer")
    configs_dir, output_root, explicit_project_dir = _parse_args()

    title = st.empty()
    col_item, col_stage = st.columns(2)
    with col_item:
        cfg, config_path = _select_config(configs_dir)

    # Example projects write data paths relative to the project dir (the parent
    # of configs/), so resolve them against it rather than against the cwd.
    project_dir = config_path.resolve().parent.parent
    if explicit_project_dir is not None:
        project_dir = explicit_project_dir.resolve()
    elif not cfg.loader.source.is_absolute() and not (project_dir / cfg.loader.source).exists():
        # Nested/symlinked configs often use paths relative to the launch directory.
        if (Path.cwd() / cfg.loader.source).exists():
            project_dir = Path.cwd()

    def _resolve(path: "str | Path") -> Path:
        path = Path(path)
        return path if path.is_absolute() else project_dir / path

    loader_source = _resolve(cfg.loader.source)
    image_dir = _resolve((cfg.loader.kwargs or {}).get("image_dir", "."))
    image_roots = [image_dir, project_dir, Path.cwd()]

    stages = [
        {
            "name": stage.output,
            "inputs": [{"type": inp[0], "source": inp[1].value} for inp in stage.inputs],
        }
        for stage in cfg.stages
    ]

    title.title(f"Pipeline: {cfg.name}")

    if not loader_source.exists():
        st.error(f"Loader source not found: {loader_source}")
        st.stop()

    items = json.loads(loader_source.read_text(encoding="utf-8"))
    item_ids = [item["id"] for item in items]
    item_data = {item["id"]: item["data"] for item in items}

    # Keep config and example controls together; expose every stage on the right.
    stage_names = [s["name"] for s in stages]

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
        stage_idx = st.radio(
            "Stage",
            options=range(len(stage_names)),
            format_func=lambda index: f"{index + 1}. {stage_names[index]}",
            key="stage_idx",
        )
        stage = stages[stage_idx]
        stage_name = stage["name"]

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
                    img_path = resolve_image(str(val), image_roots, explicit=True)
                    if img_path is not None:
                        show_image(img_path, input_type)
                    else:
                        st.text(str(val))
                else:
                    show_value(val, image_roots, input_type)
            else:
                cache_file = output_root / cfg.name / input_type / item_id / "output.json"
                if cache_file.exists():
                    show_value(json.loads(cache_file.read_text(encoding="utf-8")),
                               [cache_file.parent, *image_roots], input_type)
                else:
                    st.warning(f"Stage output not found: {cache_file}")

    with right:
        st.subheader(f"Output: {stage_name}")
        output_file = stage_output_dir / item_id / "output.json"
        if output_file.exists():
            show_value(json.loads(output_file.read_text(encoding="utf-8")),
                       [output_file.parent, *image_roots], stage_name)
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
