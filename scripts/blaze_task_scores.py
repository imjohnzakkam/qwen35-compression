#!/usr/bin/env python3
"""Task accuracy of the domain answers: HumanEval and MBPP pass@1, MGSM accuracy in 11 languages.

The answers are scripts/blaze_domain_answers.py's (greedy, instruct mode, at most 1,024 tokens),
so every model is scored on the same prompts its drift was measured along. A code answer passes
when its program, followed by the problem's own tests, exits cleanly within the time limit; an
MGSM answer counts when its last number equals the gold answer.

Code answers are executed. Run this only where running model-written code is acceptable: the
rented GPU instance, not a workstation.

    .venv-gpu-text/bin/python scripts/blaze_task_scores.py --answers DIR --name NAME --output FILE
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from blaze_domain_answers import (  # noqa: E402
    HUMANEVAL,
    MBPP,
    MBPP_PROBLEMS,
    MGSM,
    MGSM_LANGUAGES,
    _load,
)

TIMEOUT_SECONDS = 10
WORKERS = 16
_BLOCK = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.S)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_code(answer: str, name: str | None = None) -> str:
    """The code block that defines `name` (else the longest block), or the whole answer."""
    blocks = _BLOCK.findall(answer)
    if not blocks:
        return answer
    if name:
        defining = [block for block in blocks if f"def {name}" in block]
        if defining:
            return defining[0]
    return max(blocks, key=len)


def humaneval_program(row: dict, answer: str) -> str:
    code = extract_code(answer, row["entry_point"])
    if f"def {row['entry_point']}" not in code:
        code = row["prompt"] + code  # a body without its signature
    return f"{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"


def mbpp_program(row: dict, answer: str) -> str:
    called = re.search(r"assert\s+(?:not\s+)?\(?\s*([A-Za-z_]\w*)\s*\(", row["test_list"][0])
    code = extract_code(answer, called.group(1) if called else None)
    setup = row.get("test_setup_code") or ""
    return f"{setup}\n{code}\n\n" + "\n".join(row["test_list"]) + "\n"


def passes(program: str, timeout: int = TIMEOUT_SECONDS) -> bool:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "program.py"
        path.write_text(program, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=folder,
                env=env,
                capture_output=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False
        return done.returncode == 0


def last_number(answer: str) -> float | None:
    """The answer's last number, commas dropped; any script's decimal digits count."""
    found = _NUMBER.findall(answer)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def mgsm_correct(answer: str, gold: float) -> bool:
    value = last_number(answer)
    return value is not None and abs(value - float(gold)) < 1e-6


def _answers(folder: Path, name: str) -> dict[int, str]:
    rows = {}
    with (folder / f"{name}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["doc_id"]] = row["resps"][0][0]
    return rows


def score(folder: Path) -> dict[str, dict[int, float]]:
    """File name -> {doc_id: 1.0 if correct else 0.0}, for every domain file present."""
    out: dict[str, dict[int, float]] = {}
    jobs: list[tuple[str, int, str]] = []
    if (folder / "domain_code_humaneval.jsonl").exists():
        answers = _answers(folder, "domain_code_humaneval")
        rows = list(_load(*HUMANEVAL, split="test"))
        jobs += [
            ("domain_code_humaneval", i, humaneval_program(rows[i], a)) for i, a in answers.items()
        ]
    if (folder / "domain_code_mbpp.jsonl").exists():
        answers = _answers(folder, "domain_code_mbpp")
        rows = list(_load(*MBPP, "full", split="test"))[:MBPP_PROBLEMS]
        jobs += [("domain_code_mbpp", i, mbpp_program(rows[i], a)) for i, a in answers.items()]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda job: passes(job[2]), jobs))
    for (name, index, _), passed in zip(jobs, results, strict=True):
        out.setdefault(name, {})[index] = float(passed)
    for language in MGSM_LANGUAGES:
        name = f"domain_ml_{language}"
        if not (folder / f"{name}.jsonl").exists():
            continue
        rows = list(_load(*MGSM, language, split="test"))
        out[name] = {
            i: float(mgsm_correct(a, rows[i]["answer_number"]))
            for i, a in _answers(folder, name).items()
        }
    return out


def summarize(per_file: dict[str, dict[int, float]]) -> dict[str, dict[str, float]]:
    groups = {
        "humaneval": ["domain_code_humaneval"],
        "mbpp": ["domain_code_mbpp"],
        "mgsm": [f"domain_ml_{x}" for x in MGSM_LANGUAGES],
        "mgsm_non_english": [f"domain_ml_{x}" for x in MGSM_LANGUAGES if x != "en"],
        **{f"mgsm_{x}": [f"domain_ml_{x}"] for x in MGSM_LANGUAGES},
    }
    out = {}
    for group, names in groups.items():
        values = [v for name in names for v in per_file.get(name, {}).values()]
        if values:
            out[group] = {"accuracy": 100 * sum(values) / len(values), "questions": len(values)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True, help="domain_*.jsonl answers")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    per_file = score(args.answers)
    result = {
        "name": args.name,
        "answers": str(args.answers),
        "timeout_seconds": TIMEOUT_SECONDS,
        "summary": summarize(per_file),
        "per_file": {name: {str(k): v for k, v in rows.items()} for name, rows in per_file.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"name": args.name, **result["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
