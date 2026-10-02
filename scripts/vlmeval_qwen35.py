#!/usr/bin/env python3
"""Score a pinned Qwen3.5 checkpoint with VLMEvalKit through a local vLLM server.

VLMEvalKit's in-process Qwen3-VL path sends one prompt at a time to vLLM, which ran at
roughly five seconds per sample on an L4. Serving the checkpoint and driving VLMEvalKit's
OpenAI-compatible API path with many workers lets vLLM batch requests instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_KEY = "sk-local"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def vlmeval_run_prefix(toolkit_dir: Path, limit: int | None) -> list[str]:
    # Same as qwen35_compression.feature1.vlmeval_run_prefix; this script runs in the vision
    # environment, which does not install the project package.
    prefix = [str(Path(__file__).resolve().parent / "vlmeval_run.py"), "--toolkit-dir"]
    prefix.append(str(toolkit_dir))
    if limit is not None:
        prefix.extend(("--limit", str(limit)))
    return [*prefix, "--"]


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def server_command(args: argparse.Namespace) -> list[str]:
    if args.server == "local":
        # Local validation only: scripts/local_vlm_server.py stands in for vLLM on the Mac.
        return [
            str(args.server_python or sys.executable),
            str(Path(__file__).resolve().parent / "local_vlm_server.py"),
            "--model",
            str(args.model_path),
            "--served-model-name",
            args.alias,
            "--port",
            str(args.port),
            "--device",
            args.device,
            "--request-log",
            str(args.output_dir / "local_server_requests.jsonl"),
        ]
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model_path),
        "--served-model-name",
        args.alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--limit-mm-per-prompt",
        json.dumps({"image": args.max_images}),
    ]


def vlmeval_command(
    args: argparse.Namespace,
    toolkit_dir: Path,
    data: list[str],
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        *vlmeval_run_prefix(toolkit_dir, args.limit),
        "--model",
        args.alias,
        "--data",
        *data,
        "--work-dir",
        str(args.output_dir),
        "--mode",
        mode,
        "--reuse",
        "--base-url",
        base_url(args.port),
        "--model-class",
        "LMDeployAPI",
        "--key",
        API_KEY,
        "--api-nproc",
        str(args.api_nproc),
        "--temperature",
        "0",
        "--max-tokens",
        str(args.max_new_tokens),
        "--timeout",
        str(args.timeout),
        "--retry",
        "3",
    ]
    if args.disable_thinking:
        command.extend(("--extra-body", json.dumps(NO_THINKING)))
    if mode == "eval":
        command.extend(("--judge", "exact_matching"))
    return command


def build_plan(args: argparse.Namespace, toolkit_dir: Path) -> dict[str, object]:
    # Datasets whose free-form answers need the fixed answer extractor are only inferred here;
    # scripts/score_vision.py scores them after download so every variant shares one extractor.
    extracted = [name for name in args.data if name in args.extracted_data]
    scored_by_rules = [name for name in args.data if name not in args.extracted_data]
    return {
        "toolkit_dir": str(toolkit_dir),
        "model_path": str(args.model_path),
        "enable_thinking": not args.disable_thinking,
        "extracted_datasets": extracted,
        "server": server_command(args),
        "infer": vlmeval_command(args, toolkit_dir, list(args.data), "infer"),
        "eval": (
            vlmeval_command(args, toolkit_dir, scored_by_rules, "eval") if scored_by_rules else None
        ),
    }


def wait_for_server(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"model server did not become healthy within {timeout:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pinned Qwen3.5 checkpoint with VLMEvalKit via vLLM serving"
    )
    parser.add_argument("--toolkit-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument(
        "--extracted-data",
        nargs="*",
        default=[],
        help="Datasets to infer but leave for scripts/score_vision.py to score",
    )
    parser.add_argument("--alias", default="Qwen3.5-4B-pinned")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-images", type=int, default=8, help="Images per prompt (MMMU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-nproc", type=int, default=32, help="Concurrent requests")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-request timeout")
    parser.add_argument("--server-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render prompts with enable_thinking=False so answers are not reasoning traces",
    )
    parser.add_argument(
        "--server",
        choices=("vllm", "local"),
        default="vllm",
        help="local: transformers stand-in for local validation runs only",
    )
    parser.add_argument("--server-python", type=Path, help="Python for --server local")
    parser.add_argument("--device", default="mps", help="Device for --server local")
    parser.add_argument("--limit", type=int, help="First N questions per dataset (validation)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    toolkit_dir = args.toolkit_dir.resolve()
    if not (toolkit_dir / "run.py").is_file():
        raise FileNotFoundError(f"VLMEvalKit run.py not found: {toolkit_dir}")
    plan = build_plan(args, toolkit_dir)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_log = (args.output_dir / f"{args.server}_server.log").open("ab")
    server = subprocess.Popen(plan["server"], stdout=server_log, stderr=subprocess.STDOUT)  # type: ignore[arg-type]
    try:
        wait_for_server(server, args.port, args.server_timeout)
        for stage in ("infer", "eval"):
            command = plan[stage]
            if command is None:
                continue
            print(f"= vlmeval {stage}", flush=True)
            subprocess.run(command, cwd=toolkit_dir, check=True)  # type: ignore[arg-type]
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()


if __name__ == "__main__":
    main()
