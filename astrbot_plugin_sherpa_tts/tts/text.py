"""Text preparation helpers for offline TTS.

Kokoro is a sentence-level model: very long inputs are tokenised into one
phoneme sequence and can degrade or fail.  AstrBot replies can be many
paragraphs long, so we split on natural boundaries and synthesise chunk by
chunk.
"""

from __future__ import annotations

import re

# Sentence terminators for Chinese and English.  The punctuation is kept with
# the preceding chunk so the prosody of the sentence stays intact.
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;.!?])\s*")
_WHITESPACE = re.compile(r"\s+")
# Markdown / control characters that should never be read aloud.
_MARKDOWN_NOISE = re.compile(r"[*_`#>~|]")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip markdown emphasis markers."""
    text = _MARKDOWN_NOISE.sub("", text or "")
    return _WHITESPACE.sub(" ", text).strip()


def split_text(text: str, max_chars: int = 160) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Splitting prefers sentence boundaries, then punctuation inside a sentence,
    and finally a hard cut.  Empty input yields an empty list.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        # Sentence is too long: pack smaller pieces at comma-like boundaries.
        pieces = re.split(r"(?<=[，,、：:）)])\s*", sentence)
        buffer = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(buffer) + len(piece) <= max_chars:
                buffer += piece
                continue
            if buffer:
                chunks.append(buffer)
            # A single piece may still be too long: hard-wrap it.
            while len(piece) > max_chars:
                chunks.append(piece[:max_chars])
                piece = piece[max_chars:]
            buffer = piece
        if buffer:
            chunks.append(buffer)

    return chunks


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])(?=\s|$)")


def count_sentences(text: str) -> int:
    """Count sentence-like units for the per-request sentence limit.

    ASCII ``.`` only terminates a sentence when followed by whitespace or the
    end of the text, so decimals and version numbers do not inflate the count.
    Text without punctuation counts as one sentence.
    """
    text = normalize_text(text)
    if not text:
        return 0
    return len([part for part in _SENTENCE_BOUNDARY.split(text) if part.strip()])
