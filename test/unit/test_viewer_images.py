"""Test nested image discovery and rendering for viewer inputs and stage caches.

Usage:
    python -m pytest test/unit/test_viewer_images.py -v
"""
import json
import sys
from pathlib import Path

import yaml
from PIL import Image
from streamlit.testing.v1 import AppTest

from agentflow import cli
from agentflow.viewer_images import find_images, resolve_image


def make_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "red").save(path, format="PNG")
    return path.resolve()


def test_nested_discovery_deduplicates_and_preserves_field_labels(tmp_path):
    base = make_image(tmp_path / "images/base.png")
    example = make_image(tmp_path / "examples/one.JPG")
    payload = {"image": "images/base.png", "examples": ["examples/one.JPG", {"image": str(base)}],
               "missing": "missing.jpg", "directory": "images", "url": "https://example.com/image.jpg",
               "metadata": {"text": "A description", "count": 3, "nothing": None}}
    assert find_images(payload, [tmp_path], "Template") == [
        ("Template.image", base), ("Template.examples[0]", example)]


def test_path_resolution_preserves_directories_and_legacy_image_basename(tmp_path):
    nested = make_image(tmp_path / "images/nested/base.png")
    flat = make_image(tmp_path / "images/base.png")
    assert resolve_image("nested/base.png", [tmp_path / "images"]) == nested
    assert resolve_image("old/unavailable/base.png", [tmp_path / "images"]) is None
    assert resolve_image("old/unavailable/base.png", [tmp_path / "images"], explicit=True) == flat
    extensionless = make_image(tmp_path / "images/no_extension")
    assert resolve_image("no_extension", [tmp_path / "images"]) is None
    assert resolve_image("no_extension", [tmp_path / "images"], explicit=True) == extensionless


def test_absolute_paths_and_cache_relative_artifacts(tmp_path):
    project = tmp_path / "project"
    cache = tmp_path / "cache"
    source = make_image(project / "source.png")
    generated = make_image(cache / "preview.png")
    assert find_images({"source": str(source), "result": "preview.png"}, [cache, project]) == [
        ("source", source), ("result", generated)]
    assert resolve_image("\x00.jpg", [tmp_path]) is None
    assert resolve_image("a" * 5000 + ".png", [tmp_path]) is None


def viewer_app(tmp_path, monkeypatch, bad_image=False, explicit_image=False):
    (tmp_path / "configs").mkdir()
    base = make_image(tmp_path / "images/base.png")
    first = make_image(tmp_path / "examples/first.png")
    second = make_image(tmp_path / "examples/second.png")
    data = {"template": {"image": "images/base.png", "examples": [str(first), {"image": str(second)}]},
            "image": "images/base.png"}
    (tmp_path / "items.json").write_text(json.dumps([{"id": "one", "data": data}]))
    config = {"name": "sample", "loader": {"source": "items.json", "kwargs": {"image_dir": "images"}},
              "stages": [{"inputs": [["Image" if explicit_image else "Template", "human"]],
                          "output": "SelectedExamples", "processor": "CopyProcessor"},
                         {"inputs": [["SelectedExamples", "model"]], "output": "Prediction", "processor": "CopyProcessor"}]}
    (tmp_path / "configs/sample.yaml").write_text(yaml.safe_dump(config))
    selected = tmp_path / "output/sample/SelectedExamples/one"
    make_image(selected / "preview.png")
    payload = {"examples": [{"image": "preview.png", "meaning": "original output text"}]}
    if bad_image:
        (selected / "broken.png").write_text("not an image")
        payload["examples"].append({"image": "broken.png"})
    (selected / "output.json").write_text(json.dumps(payload))
    prediction = tmp_path / "output/sample/Prediction/one"
    prediction.mkdir(parents=True)
    (prediction / "output.json").write_text(json.dumps({"image": str(base), "text": "final output text"}))
    monkeypatch.chdir(tmp_path)
    viewer = Path(cli.__file__).with_name("viewer.py")
    monkeypatch.setattr(sys, "argv", [str(viewer), "configs", "output"])
    return AppTest.from_file(str(viewer)).run()


def image_captions(app):
    return [image.caption for element in app.get("image") for image in element.proto.imgs]


def test_viewer_shows_nested_human_inputs_and_outputs(tmp_path, monkeypatch):
    app = viewer_app(tmp_path, monkeypatch)
    assert not app.exception
    assert not app.error
    assert image_captions(app) == ["Template.image", "Template.examples[0]", "Template.examples[1].image",
                                   "SelectedExamples.examples[0].image"]
    assert len(app.json) == 2
    assert "original output text" in app.json[1].value


def test_viewer_shows_model_input_images_and_output_images(tmp_path, monkeypatch):
    app = viewer_app(tmp_path, monkeypatch)
    app.number_input[1].set_value(1).run()
    assert not app.exception
    assert not app.error
    assert image_captions(app) == ["SelectedExamples.examples[0].image", "Prediction.image"]
    assert "original output text" in app.json[0].value
    assert "final output text" in app.json[1].value


def test_viewer_keeps_explicit_image_inputs_working(tmp_path, monkeypatch):
    app = viewer_app(tmp_path, monkeypatch, explicit_image=True)
    assert not app.exception
    assert image_captions(app) == ["Image", "SelectedExamples.examples[0].image"]


def test_corrupt_image_does_not_hide_json_or_other_previews(tmp_path, monkeypatch):
    app = viewer_app(tmp_path, monkeypatch, bad_image=True)
    assert not app.exception
    assert not app.error
    assert any("Cannot display" in warning.value for warning in app.warning)
    assert "original output text" in app.json[1].value
    assert "SelectedExamples.examples[0].image" in image_captions(app)
