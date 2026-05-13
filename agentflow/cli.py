import argparse
import re
import sys
from pathlib import Path


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

    rename = sub.add_parser("rename", help="Rename a pipeline")
    rename.add_argument("old_name", help="Current pipeline name")
    rename.add_argument("new_name", help="New pipeline name")
    rename.add_argument("--configs-dir", default="configs", metavar="DIR", help="Directory containing config files (default: configs)")
    rename.add_argument("--output-dir", default="output", metavar="DIR", help="Pipeline output root directory (default: output)")

    args = parser.parse_args()
    if args.command == "rename":
        cmd_rename(args)


if __name__ == "__main__":
    main()
