"""Offline, pinned study-token accounting; not a claim about provider billing.

o200k_base ranks/pattern are from openai/tiktoken (MIT; see tokenizer_data).
Chat accounting tokenizes the complete explicit serialized message envelope.
Provider-internal chat framing/reasoning remains separately reported usage.
"""

import base64
from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path

import tiktoken

from .context_memory import serialized

RANKS_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


@lru_cache(maxsize=1)
def encoding():
    data = (Path(__file__).parent / "tokenizer_data" / "o200k_base.tiktoken").read_bytes()
    if hashlib.sha256(data).hexdigest() != RANKS_SHA256:
        raise ValueError("Study tokenizer ranks checksum mismatch")
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in data.splitlines())}
    pattern = "|".join([
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
    ])
    return tiktoken.Encoding(name="study-o200k-base-v1", pat_str=pattern, mergeable_ranks=ranks,
                             special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018})


def count(text):
    return len(encoding().encode_ordinary(text))


def input_tokens(system, payload):
    return count(serialized([{"role": "system", "content": system},
                             {"role": "user", "content": serialized(payload)}]))


def metadata():
    return dict(name="study-o200k-base-v1", library="tiktoken", version=version("tiktoken"),
                ranks_sha256=RANKS_SHA256, framing="serialized explicit system/user message envelope",
                provider_native_or_billing_equivalence=False)
