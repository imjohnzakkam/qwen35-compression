#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def download_command(instance_id: int, remote_path: str, local_path: Path) -> list[str]:
    return ["jl", "download", str(instance_id), remote_path, str(local_path), "-r"]


def verify_download(destination: Path, allow_pilot: bool = False) -> Path:
    manifests = list(destination.rglob("run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected one run_manifest.json, found {len(manifests)}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError(f"remote run did not pass: {manifest.get('status')}")
    if not allow_pilot and manifest.get("research_result") is not True:
        raise ValueError("refusing to accept a limited pilot as the BF16 baseline")
    inventory = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    receipt = destination / "download_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "remote_manifest": manifest,
                "files": inventory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a completed JarvisLabs run into local logs")
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--remote-path", default="/home/qwen35-compression/results/feature1/bf16")
    parser.add_argument("--local-root", type=Path, default=Path("logs/jarvis"))
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = (args.local_root / f"feature1-bf16-{timestamp}").resolve()
    command = download_command(args.instance_id, args.remote_path, destination)
    if args.dry_run:
        print(json.dumps({"command": command, "destination": str(destination)}, indent=2))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    receipt = verify_download(destination, allow_pilot=args.allow_pilot)
    print(f"downloaded={destination}")
    print(f"receipt={receipt}")


if __name__ == "__main__":
    main()
