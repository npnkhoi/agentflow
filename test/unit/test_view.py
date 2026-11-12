"""Check viewer CLI launching and config path resolution offline.

Usage:
    python -m pytest test/unit/test_view.py -v
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from agentflow import cli


def test_view_defaults_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    run = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["agentflow", "view"])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 7
    command = run.call_args.args[0]
    assert command == [sys.executable, "-m", "streamlit", "run",
                       str(Path(cli.__file__).with_name("viewer.py")),
                       "--server.port", "8501", "--server.address", "127.0.0.1",
                       "--", "configs", "output"]


def test_view_custom_options(tmp_path, monkeypatch):
    config = tmp_path / "config with spaces.yaml"
    config.touch()
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["agentflow", "view", str(config), "results with spaces",
                                      "--port", "9000", "--host", "0.0.0.0", "--headless",
                                      "--project-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 0
    command = run.call_args.args[0]
    assert command[command.index("--server.port") + 1] == "9000"
    assert command[command.index("--server.address") + 1] == "0.0.0.0"
    assert command[command.index("--server.headless") + 1] == "true"
    assert command[command.index("--") + 1:] == [str(config), "results with spaces", "--project-dir", str(tmp_path)]


def test_view_interrupt_exits_without_traceback(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", Mock(side_effect=KeyboardInterrupt))
    assert cli.launch_viewer(Path("configs"), Path("output")) == 130


@pytest.mark.parametrize("arguments", [["missing"], ["configs", "--port", "0"], ["configs", "--port", "65536"]])
def test_invalid_view_arguments_do_not_launch(tmp_path, monkeypatch, capsys, arguments):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    run = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["agentflow", "view", *arguments])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 2
    assert "Error:" in capsys.readouterr().err
    run.assert_not_called()


@pytest.mark.parametrize("layout", ["conventional", "symlink", "explicit"])
def test_viewer_renders_cached_output(tmp_path, monkeypatch, layout):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "data").mkdir()
    (project / "data/items.json").write_text(json.dumps([{"id": "one", "data": {"argument": "example"}}]))
    config = project / "configs/sample.yaml"
    config.write_text(yaml.safe_dump({"name": "sample", "loader": {"source": "data/items.json", "kwargs": {"image_dir": "."}},
                                      "stages": [{"inputs": [["Argument", "human"]], "output": "Prediction", "processor": "CopyProcessor"}]}))
    output = tmp_path / "output/sample/Prediction/one"
    output.mkdir(parents=True)
    (output / "output.json").write_text(json.dumps({"value": "cached result"}))
    arguments = [str(config.parent), str(tmp_path / "output")]
    monkeypatch.chdir(tmp_path)
    if layout == "symlink":
        nested = project / "src/domain/af/configs"
        nested.mkdir(parents=True)
        target = nested.parent / "config.yaml"
        config.rename(target)
        (nested / "config.yaml").symlink_to("../config.yaml")
        arguments[0] = str(nested)
        monkeypatch.chdir(project)
    elif layout == "explicit":
        arguments[0] = str(config)
        arguments.extend(["--project-dir", str(project)])
    viewer = Path(cli.__file__).with_name("viewer.py")
    monkeypatch.setattr(sys, "argv", [str(viewer), *arguments])
    app = AppTest.from_file(str(viewer)).run()
    assert not app.exception
    assert not app.error
    assert any("cached result" in element.value for element in app.json)
