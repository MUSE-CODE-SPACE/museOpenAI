"""Byte-level BPE tokenizer, trained from scratch with zero dependencies.

The tokenizer operates on UTF-8 bytes, so it can encode any string with no
unknown tokens. Text is first split into chunks with a GPT-style regex,
then byte-pair merges learned during training are applied greedily within
each chunk. Token ids 0..255 are the raw bytes; merges and special tokens
occupy the ids above.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# GPT-2 style pre-tokenization pattern, adapted to Python's `re` module.
SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)

DEFAULT_SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]


def _pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter:
    counts: Counter = Counter()
    for chunk, freq in chunks.items():
        for pair in zip(chunk, chunk[1:]):
            counts[pair] += freq
    return counts


def _merge_chunk(chunk: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(chunk[i])
            i += 1
    return tuple(out)


class ByteBPETokenizer:
    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        for i, pair in enumerate(merges or []):
            self.merges[tuple(pair)] = 256 + i
        self.special_tokens: dict[str, int] = {}
        base = 256 + len(self.merges)
        for i, tok in enumerate(special_tokens or DEFAULT_SPECIAL_TOKENS):
            self.special_tokens[tok] = base + i
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            self._vocab[idx] = self._vocab[pair[0]] + self._vocab[pair[1]]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> ByteBPETokenizer:
        """Learn BPE merges from ``text`` until the vocab reaches ``vocab_size``.

        The final vocabulary is 256 byte tokens + merges + special tokens.
        """
        special = special_tokens or DEFAULT_SPECIAL_TOKENS
        n_merges = vocab_size - 256 - len(special)
        if n_merges <= 0:
            raise ValueError(f"vocab_size {vocab_size} too small for byte vocab + special tokens")

        # Unique pre-tokenized chunks with frequencies — merging operates on
        # this multiset, which is dramatically smaller than the raw corpus.
        chunk_freqs: Counter = Counter(SPLIT_PATTERN.findall(text))
        chunks: dict[tuple[int, ...], int] = {
            tuple(chunk.encode("utf-8")): freq for chunk, freq in chunk_freqs.items()
        }

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            counts = _pair_counts(chunks)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + step
            merges.append(pair)
            chunks = {
                _merge_chunk(c, pair, new_id) if pair[0] in c or pair[1] in c else c: f
                for c, f in chunks.items()
            }
            if verbose and (step + 1) % 256 == 0:
                print(f"  merge {step + 1}/{n_merges}: {pair} -> {new_id} (freq {freq})")

        return cls(merges=merges, special_tokens=special)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if best is None:
                break
            ids = list(_merge_chunk(tuple(ids), best, self.merges[best]))
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        ids: list[int] = []
        if self._special_pattern and allow_special:
            parts = self._special_pattern.split(text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in SPLIT_PATTERN.findall(part):
                ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special_tokens.items()}
        out: list[bytes] = []
        for i in ids:
            if i in inv_special:
                out.append(inv_special[i].encode("utf-8"))
            elif i in self._vocab:
                out.append(self._vocab[i])
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(out).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        inv = sorted(self.merges.items(), key=lambda kv: kv[1])
        payload = {
            "version": 1,
            "type": "byte_bpe",
            "merges": [list(pair) for pair, _ in inv],
            "special_tokens": list(self.special_tokens.keys()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            merges=[tuple(p) for p in payload["merges"]],
            special_tokens=payload["special_tokens"],
        )
