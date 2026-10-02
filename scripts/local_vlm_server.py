#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for local validation runs (Apple GPU or CPU).

Stands in for vLLM's server so VLMEvalKit's API path (LMDeployAPI) runs unchanged on a Mac.
Requests are served one at a time. It is not a research backend: scores from it are only
used to check the pipeline end to end before a GPU run.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def decode_image(url: str, max_pixels: int | None = None) -> Image.Image:
    if url.startswith("data:"):
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    else:
        image = Image.open(url.removeprefix("file://"))
    image = image.convert("RGB")
    # The vision encoder's attention is quadratic in patches without flash attention: a 2257x1764
    # DocVQA page asked the Apple GPU for a 15.6 GB buffer. vLLM on CUDA needs no such limit.
    if max_pixels and image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BICUBIC)
    return image


def to_processor_messages(
    messages: list[dict], max_pixels: int | None = None
) -> tuple[list[dict], list[Image.Image]]:
    converted, images = [], []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part[
                    "image_url"
                ]
                images.append(decode_image(url, max_pixels))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"unsupported content part: {part['type']}")
        converted.append({"role": message["role"], "content": parts})
    return converted, images


class Backend:
    def __init__(
        self,
        model_path: Path,
        device: str,
        alias: str,
        log_path: Path,
        max_pixels: int | None = None,
    ) -> None:
        self.alias = alias
        self.max_pixels = max_pixels
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        self.lock = threading.Lock()
        self.log = log_path.open("a", encoding="utf-8")

    def complete(self, payload: dict) -> dict:
        messages, images = to_processor_messages(payload["messages"], self.max_pixels)
        template_kwargs = payload.get("chat_template_kwargs") or {}
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self.device)
        max_tokens = int(payload.get("max_tokens") or 256)
        temperature = float(payload.get("temperature") or 0.0)
        sampling: dict = {"do_sample": temperature > 0}
        if temperature > 0:
            sampling["temperature"] = temperature
        start = time.monotonic()
        with self.lock, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, **sampling)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True)
        finish = "length" if len(new_tokens) >= max_tokens else "stop"
        record = {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(len(new_tokens)),
            "finish_reason": finish,
            "images": len(images),
            "seconds": round(time.monotonic() - start, 2),
            "thinking_in_prompt": "<think>" in text.split("<|im_start|>assistant")[-1]
            and "</think>" not in text.split("<|im_start|>assistant")[-1],
        }
        with self.lock:
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()
        return {
            "id": f"local-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.alias,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": finish,
                }
            ],
            "usage": {
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["completion_tokens"],
                "total_tokens": record["prompt_tokens"] + record["completion_tokens"],
            },
        }


def make_handler(backend: Backend):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": backend.alias}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(200, backend.complete(payload))
            except Exception as error:  # reported to the client, which counts it as a failure
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1024 * 1024,
        help="Downscale larger images (local memory limit; 0 disables)",
    )
    args = parser.parse_args()
    backend = Backend(
        args.model, args.device, args.served_model_name, args.request_log, args.max_pixels or None
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend))
    server.serve_forever()


if __name__ == "__main__":
    main()
