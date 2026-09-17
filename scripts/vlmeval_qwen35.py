#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path


def forwarded_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--model",
        args.alias,
        "--data",
        *args.data,
        "--work-dir",
        str(args.output_dir),
        "--mode",
        "all",
        "--verbose",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a pinned Qwen3.5 checkpoint with VLMEvalKit"
    )
    parser.add_argument("--toolkit-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--alias", default="Qwen3.5-4B-pinned")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render prompts with enable_thinking=False so answers are not reasoning traces",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    toolkit_dir = args.toolkit_dir.resolve()
    if not (toolkit_dir / "run.py").is_file():
        raise FileNotFoundError(f"VLMEvalKit run.py not found: {toolkit_dir}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "toolkit_dir": str(toolkit_dir),
                    "model_path": str(args.model_path),
                    "enable_thinking": not args.disable_thinking,
                    "forwarded_arguments": forwarded_arguments(args),
                },
                indent=2,
            )
        )
        return

    sys.path.insert(0, str(toolkit_dir))
    import vllm
    from vlmeval.config import supported_VLM
    from vlmeval.vlm import Qwen3VLChat

    original_llm = vllm.LLM

    def limited_context_llm(*positional: object, **keywords: object) -> object:
        keywords.setdefault("max_model_len", args.max_model_len)
        return original_llm(*positional, **keywords)

    vllm.LLM = limited_context_llm

    build_chat = partial(
        Qwen3VLChat,
        model_path=str(args.model_path),
        use_custom_prompt=False,
        use_vllm=True,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
    )

    def build_model(**overrides: object) -> object:
        model = build_chat(**overrides)
        if args.disable_thinking:
            # Qwen3VLChat calls processor.apply_chat_template without template kwargs,
            # so the checkpoint's default thinking mode would otherwise apply.
            original = model.processor.apply_chat_template

            def without_thinking(*positional: object, **keywords: object) -> object:
                keywords.setdefault("enable_thinking", False)
                return original(*positional, **keywords)

            model.processor.apply_chat_template = without_thinking
        return model

    supported_VLM[args.alias] = build_model

    sys.path.insert(0, str(toolkit_dir))
    from run import main as vlmeval_main

    sys.argv = [str(toolkit_dir / "run.py"), *forwarded_arguments(args)]
    vlmeval_main()


if __name__ == "__main__":
    main()
