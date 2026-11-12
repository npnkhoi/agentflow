import argparse
import re
import subprocess
import sys
from pathlib import Path


def launch_viewer(
    configs_path: Path,
    output_dir: Path,
    *,
    port: int = 8501,
    host: str = "127.0.0.1",
    headless: bool = False,
    project_dir: Path | None = None,
) -> int:
    """Launch the bundled viewer with the current interpreter and return its exit code."""
    command = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).with_name("viewer.py")),
        "--server.port", str(port), "--server.address", host,
    ]
    if headless:
        command.extend(["--server.headless", "true"])
    command.extend(["--", str(configs_path), str(output_dir)])
    if project_dir is not None:
        command.extend(["--project-dir", str(project_dir)])
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


def cmd_view(args: argparse.Namespace) -> int:
    configs_path = Path(args.configs_path)
    if not configs_path.exists():
        print(f"Error: config path '{configs_path}' does not exist.", file=sys.stderr)
        return 2
    if not 1 <= args.port <= 65535:
        print("Error: port must be between 1 and 65535.", file=sys.stderr)
        return 2
    return launch_viewer(configs_path, Path(args.output_dir), port=args.port,
                         host=args.host, headless=args.headless, project_dir=args.project_dir)


def _find_config(configs_dir: Path, name: str) -> Path:
    for ext in ("yaml", "yml"):
        p = configs_dir / f"{name}.{ext}"
        if p.exists():
            return p
    print(f"Error: no config file found for '{name}' in {configs_dir}", file=sys.stderr)
    sys.exit(1)


def cmd_rename(args: argparse.Namespace) -> None:
    old_name: str = args.old_name
    new_name: str = args.new_name
    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)

    config_path = _find_config(configs_dir, old_name)

    text = config_path.read_text(encoding="utf-8")
    updated = re.sub(r"^(name:\s*)" + re.escape(old_name) + r"\s*$", rf"\g<1>{new_name}", text, flags=re.MULTILINE)
    if updated == text:
        print(f"Warning: 'name: {old_name}' not found in {config_path}; file content unchanged.")

    new_config_path = config_path.with_name(f"{new_name}{config_path.suffix}")
    if new_config_path.exists():
        print(f"Error: config '{new_config_path}' already exists.", file=sys.stderr)
        sys.exit(1)

    old_output = output_dir / old_name
    new_output = output_dir / new_name
    if old_output.exists() and new_output.exists():
        print(f"Error: output directory '{new_output}' already exists.", file=sys.stderr)
        sys.exit(1)

    new_config_path.write_text(updated, encoding="utf-8")
    config_path.unlink()
    print(f"Config:  {config_path} → {new_config_path}")

    if old_output.exists():
        old_output.rename(new_output)
        print(f"Output:  {old_output} → {new_output}")
    else:
        print(f"Output:  {old_output} not found, skipping.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentflow", description="AgentFlow CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    view = sub.add_parser("view", help="Browse pipeline outputs in the web viewer")
    view.add_argument("configs_path", nargs="?", default="configs", metavar="CONFIGS",
                      help="YAML config file or directory (default: configs)")
    view.add_argument("output_dir", nargs="?", default="output", metavar="OUTPUT",
                      help="Pipeline output root directory (default: output)")
    view.add_argument("--port", type=int, default=8501, help="Web server port (default: 8501)")
    view.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    view.add_argument("--headless", action="store_true", help="Do not open a browser automatically")
    view.add_argument("--project-dir", type=Path, help="Base directory for relative loader and image paths")

    rename = sub.add_parser("rename", help="Rename a pipeline")
    rename.add_argument("old_name", help="Current pipeline name")
    rename.add_argument("new_name", help="New pipeline name")
    rename.add_argument("--configs-dir", default="configs", metavar="DIR", help="Directory containing config files (default: configs)")
    rename.add_argument("--output-dir", default="output", metavar="DIR", help="Pipeline output root directory (default: output)")

    args = parser.parse_args()
    if args.command == "rename":
        cmd_rename(args)
    elif args.command == "view":
        raise SystemExit(cmd_view(args))


if __name__ == "__main__":
    main()
