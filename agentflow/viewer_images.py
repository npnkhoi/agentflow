"""Discover and preview local image paths embedded in viewer JSON data.

Usage:
    agentflow view configs output
    python -m pytest test/unit/test_viewer_images.py -q
"""
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


def resolve_image(value: str, roots: list[Path], *, explicit: bool = False) -> Path | None:
    """Resolve a local image path without fetching URLs or discarding nested directories.

    Explicit Image inputs also support extensionless files and the loader's
    historical basename lookup. Generic JSON paths require an image suffix.
    """
    if not value or "://" in value:
        return None
    path = Path(value)
    if not explicit and path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    if explicit and not path.is_absolute():
        candidates.extend(root / path.name for root in roots)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except (OSError, ValueError):
            # Arbitrary JSON strings may be invalid filenames or exceed path limits.
            continue
    return None


def find_images(value, roots: list[Path], field: str = "") -> list[tuple[str, Path]]:
    """Find distinct existing local images in dictionaries, lists, or scalar strings."""
    found = []
    seen = set()

    def visit(node, label):
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, f"{label}.{key}" if label else str(key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{label}[{index}]")
        elif isinstance(node, str):
            path = resolve_image(node, roots)
            if path is not None and path not in seen:
                seen.add(path)
                found.append((label or "Image", path))

    visit(value, field)
    return found


def show_image(path: Path, caption: str) -> None:
    """Keep the viewer usable when a referenced image is corrupt or disappears."""
    try:
        st.image(str(path), caption=caption, use_container_width=True)
    except (OSError, ValueError, StreamlitAPIException) as exc:
        st.warning(f"Cannot display {caption}: {exc}")


def show_value(value, roots: list[Path], field: str = "") -> None:
    """Show image previews with field-path captions, followed by the original data."""
    images = find_images(value, roots, field)
    if images:
        columns = st.columns(min(3, len(images)))
        for index, (caption, path) in enumerate(images):
            with columns[index % len(columns)]:
                show_image(path, caption)
    if isinstance(value, (dict, list)):
        st.json(value, expanded=True)
    else:
        st.text(str(value))
