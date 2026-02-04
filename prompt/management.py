"""Prompt chunking, truncation, and compression."""

from typing import List
import re

MAX_TOKENS = 77

REPLACEMENTS = [
    (r"\b(extremely|very|ultra|highly)\s+(detailed|realistic)\b", r"\2"),
    (r"\b(8k|4k|uhd|ultra hd|high resolution)\b.*", r"\1"),
    (r"\bshot on a professional dslr camera\b", "DSLR photo"),
    (r"\bcinematic,?\s*movie[- ]?like,?\s*film still\b", "cinematic"),
    (r"\s*,\s*", ", "),
]


def compress_prompt(text: str) -> str:
    """Apply compression rules to reduce token count."""
    assert isinstance(text, str), f"compress_prompt expected str, got {type(text)}"
    out = text.lower()

    for pattern, repl in REPLACEMENTS:
        out = re.sub(pattern, repl, out)

    # Collapse duplicates
    parts = []
    seen = set()
    for p in [x.strip() for x in out.split(",")]:
        if p and p not in seen:
            seen.add(p)
            parts.append(p)

    return ", ".join(parts)


def chunk_prompt(tokenizer, text: str) -> List[str]:
    """Split prompt into CLIP-sized chunks."""
    parts = [p.strip() for p in text.split(",") if p.strip()]

    chunks = []
    current = []

    def token_len(txt):
        return len(
            tokenizer(
                txt,
                truncation=True,
                max_length=MAX_TOKENS,
                add_special_tokens=True,
            )["input_ids"]
        )

    for part in parts:
        trial = ", ".join(current + [part])
        if token_len(trial) <= MAX_TOKENS:
            current.append(part)
        else:
            if current:
                chunks.append(", ".join(current))
            current = []

            # Hard truncate the offending part
            tokens = tokenizer(
                part,
                truncation=True,
                max_length=MAX_TOKENS,
                add_special_tokens=True,
            )
            truncated = tokenizer.decode(
                tokens["input_ids"],
                skip_special_tokens=True,
            )
            current.append(truncated)

    if current:
        chunks.append(", ".join(current))

    if not chunks:
        return [""]

    return chunks


def tokenize_len(tokenizer, text: str) -> int:
    """Get token count for text."""
    return len(
        tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
    )


def truncate_to_tokens(
    tokenizer,
    text: str,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Hard truncate text to fit CLIP token limit."""
    tokens = tokenizer(
        text,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"]

    if len(tokens) <= max_tokens:
        return text

    truncated = tokens[:max_tokens]

    return tokenizer.decode(
        truncated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def smart_truncate_prompt(
    tokenizer,
    trigger_phrase,
    user_prompt,
    max_tokens=77,
):
    """Intelligently truncate prompt with trigger phrase preservation."""
    debug = {}

    trigger = f"{trigger_phrase}, " if trigger_phrase else ""
    trigger_tokens = tokenize_len(tokenizer, trigger)

    compressed = compress_prompt(user_prompt)

    budget = max_tokens - trigger_tokens
    truncated = truncate_to_tokens(
        tokenizer,
        compressed,
        budget,
    )

    final = f"{trigger}{truncated}".strip()

    debug.update({
        "original_tokens": tokenize_len(tokenizer, user_prompt),
        "compressed_tokens": tokenize_len(tokenizer, compressed),
        "final_tokens": tokenize_len(tokenizer, final),
        "compressed": compressed != user_prompt,
    })

    return final, debug