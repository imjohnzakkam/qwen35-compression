"""Stage 1: in-domain prompts, BF16's own answers, and packed calibration and held-out blocks.

Prompts come from public training sets, never a test set, and any prompt sharing a 13-gram with
MATH-500 or MMLU-Pro's test questions is dropped. BF16 answers them; prompt and answer are packed
into fixed-length blocks, with a mask of answer tokens and a domain per token, so held-out KL can
be measured on answers alone, per domain.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

DOMAINS = ("math", "mcq", "chat")
DOMAIN_CODES = {name: index for index, name in enumerate(DOMAINS)}
NGRAM = 13
LETTERS = "ABCDEFGHIJ"

MATH_TEMPLATE = (
    "Solve the following math problem step by step. Put the final answer in \\boxed{{}}.\n\n"
    "{problem}"
)
MCQ_TEMPLATE = (
    "Answer the following multiple choice question. Think step by step, then finish with "
    '"The answer is (X)" where X is the letter of the correct option.\n\n'
    "Question: {question}\nOptions:\n{options}"
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class Decontaminator:
    """Refuses any text sharing an n-gram with the test texts it was built from."""

    def __init__(self, tests: Iterable[str], n: int = NGRAM) -> None:
        self.n = n
        self.grams: set[tuple[str, ...]] = set()
        for text in tests:
            self.grams |= ngrams(text, n)

    def clean(self, text: str) -> bool:
        return not (ngrams(text, self.n) & self.grams)


def math_prompt(problem: str) -> str:
    return MATH_TEMPLATE.format(problem=problem.strip())


def mcq_prompt(question: str, options: Sequence[str]) -> str:
    listed = "\n".join(f"({LETTERS[i]}) {option.strip()}" for i, option in enumerate(options))
    return MCQ_TEMPLATE.format(question=question.strip(), options=listed)


def sciq_options(row: dict[str, Any], seed: int) -> list[str]:
    """SciQ's answer and three distractors, in a fixed per-question shuffled order."""
    options = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    random.Random(f"{seed}:{row['question']}").shuffle(options)
    return options


@dataclass(frozen=True)
class Prompt:
    id: str
    domain: str
    text: str
    source: str

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "domain": self.domain, "source": self.source, "text": self.text}


def prompt_id(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\n{text}".encode()).hexdigest()[:16]


def is_held_out(identifier: str, share: float) -> bool:
    """A fixed, content-based split: the prompt's id decides, not its position."""
    return int(identifier[:8], 16) / 0xFFFFFFFF < share


@dataclass
class Sample:
    """One prompt and BF16's answer, as token ids under the chat template."""

    domain: str
    prompt_ids: list[int]
    answer_ids: list[int]


@dataclass
class PackedBlocks:
    ids: list[list[int]] = field(default_factory=list)
    # Per token: 1 for an answer token (scored), 0 for prompt tokens.
    answer: list[list[int]] = field(default_factory=list)
    # Per token: the DOMAIN_CODES of the sample it belongs to.
    domain: list[list[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def tokens_by_domain(self) -> dict[str, int]:
        counts = dict.fromkeys(DOMAINS, 0)
        for row in self.domain:
            for code in row:
                counts[DOMAINS[code]] += 1
        return counts


def pack_samples(samples: Sequence[Sample], length: int, count: int) -> PackedBlocks:
    """Join samples in order and cut `count` blocks of `length` tokens; the rest is dropped."""
    ids: list[int] = []
    answer: list[int] = []
    domain: list[int] = []
    need = length * count
    for sample in samples:
        code = DOMAIN_CODES[sample.domain]
        ids += sample.prompt_ids + sample.answer_ids
        answer += [0] * len(sample.prompt_ids) + [1] * len(sample.answer_ids)
        domain += [code] * (len(sample.prompt_ids) + len(sample.answer_ids))
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"{len(ids)} tokens, {need} needed for {count} blocks of {length}")
    blocks = PackedBlocks()
    for start in range(0, need, length):
        blocks.ids.append(ids[start : start + length])
        blocks.answer.append(answer[start : start + length])
        blocks.domain.append(domain[start : start + length])
    return blocks


def interleave(samples: Sequence[Sample], shares: dict[str, float], seed: int) -> list[Sample]:
    """Samples ordered so every stretch of the packed stream holds the domains in proportion to
    `shares` by tokens: each next sample comes from the domain furthest below its share."""
    pools = {name: [s for s in samples if s.domain == name] for name in shares}
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    taken = dict.fromkeys(shares, 0)
    order: list[Sample] = []
    while any(pools.values()):
        total = sum(taken.values()) or 1
        candidates = [name for name in shares if pools[name]]
        name = min(candidates, key=lambda d: taken[d] / total - shares[d])
        sample = pools[name].pop()
        taken[name] += len(sample.prompt_ids) + len(sample.answer_ids)
        order.append(sample)
    return order


def save_blocks(path: Any, blocks: PackedBlocks) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for ids, answer, domain in zip(blocks.ids, blocks.answer, blocks.domain, strict=True):
            handle.write(json.dumps({"ids": ids, "answer": answer, "domain": domain}) + "\n")


def load_blocks(path: Any) -> PackedBlocks:
    blocks = PackedBlocks()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            blocks.ids.append(row["ids"])
            blocks.answer.append(row["answer"])
            blocks.domain.append(row["domain"])
    return blocks


def chat_sample(
    tokenizer: Any, domain: str, prompt: str, answer: str, template_kwargs: dict[str, Any]
) -> Sample:
    """Token ids of a prompt and its answer under the chat template; the answer is what the full
    conversation adds to the prompt with its generation header."""
    messages = [{"role": "user", "content": prompt}]
    head = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **template_kwargs
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        **template_kwargs,
    )
    head, full = _ids(head), _ids(full)
    if full[: len(head)] != head:
        # Templates that rewrite the assistant turn (e.g. an empty think block) still share the
        # user turn; score only what follows the longest common prefix.
        common = 0
        while common < min(len(head), len(full)) and head[common] == full[common]:
            common += 1
        head = full[:common]
    return Sample(domain, head, full[len(head) :])


def _ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    return [int(token) for token in value]
