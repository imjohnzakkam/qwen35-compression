#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
TOOLCHAINS = ROOT / "configs" / "toolchains.yaml"


def load_toolchains() -> dict[str, Any]:
    return yaml.safe_load(TOOLCHAINS.read_text(encoding="utf-8"))


def bootstrap_commands(scope: str = "all") -> list[list[str]]:
    if scope not in {"text", "all"}:
        raise ValueError("scope must be text or all")
    toolchains = load_toolchains()
    commands: list[list[str]] = []
    environments = ("gpu_text",) if scope == "text" else ("gpu_text", "gpu_vision")
    for name in environments:
        environment = ROOT / f".venv-{name.replace('_', '-')}"
        lock = ROOT / toolchains[name]["requirements"]
        commands.extend(
            (
                ["uv", "venv", "--clear", str(environment), "--python", toolchains[name]["python"]],
                [
                    "uv",
                    "pip",
                    "sync",
                    str(lock),
                    "--python",
                    str(environment / "bin" / "python"),
                    "--torch-backend",
                    "cu130",
                    "--index-strategy",
                    "unsafe-best-match",
                ],
            )
        )

    if scope == "text":
        return commands

    vision_python = ROOT / ".venv-gpu-vision" / "bin" / "python"
    commands.extend(
        toolkit_commands(toolchains["vlmevalkit"], ROOT / toolchains["vlmevalkit"]["directory"])
    )
    commands.append(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(vision_python),
            "--no-deps",
            "--editable",
            str(ROOT / toolchains["vlmevalkit"]["directory"]),
        ]
    )
    return commands


def is_git_checkout(path: Path) -> bool:
    if not path.is_dir():
        return False
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def toolkit_commands(toolkit: dict[str, Any], toolkit_dir: Path) -> list[list[str]]:
    """Ensure the toolkit directory is a real clone pinned to the configured revision.

    A directory uploaded from another machine may carry the files but not the
    Git metadata, so an existing path is only trusted when Git recognises it.
    """
    commands: list[list[str]] = []
    if not is_git_checkout(toolkit_dir):
        if toolkit_dir.exists():
            commands.append(["rm", "-rf", str(toolkit_dir)])
        commands.append(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                toolkit["repository"],
                str(toolkit_dir),
            ]
        )
    commands.append(["git", "-C", str(toolkit_dir), "checkout", toolkit["revision"]])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create isolated, pinned GPU evaluator environments"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scope", choices=("text", "all"), default="all")
    args = parser.parse_args()
    commands = bootstrap_commands(args.scope)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    if platform.system() != "Linux":
        raise RuntimeError("GPU evaluator installation is Linux-only; use --dry-run on macOS")
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
