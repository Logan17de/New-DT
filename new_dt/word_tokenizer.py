from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass(slots=True)
class WordSpaceTokenizer:
    """Deterministic tokenizer that splits only on Unicode whitespace.

    Punctuation stays attached to the surrounding word. Non-empty input lines are
    separated by ``<eos>`` when a document is encoded for language modelling.
    """

    tokens: list[str]
    lowercase: bool = False
    _token_to_id: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
            raise ValueError("special tokens must occupy the first vocabulary IDs")
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("token vocabulary contains duplicates")
        self._token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def train(
        cls,
        text: str,
        *,
        lowercase: bool = False,
        min_frequency: int = 1,
        max_vocab: int | None = None,
    ) -> WordSpaceTokenizer:
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if max_vocab is not None and max_vocab < len(SPECIAL_TOKENS):
            raise ValueError(
                f"max_vocab must be at least {len(SPECIAL_TOKENS)} for special tokens"
            )

        normalized = text.lower() if lowercase else text
        counts = Counter(normalized.split())
        words = [
            word
            for word, frequency in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency >= min_frequency and word not in SPECIAL_TOKENS
        ]
        if max_vocab is not None:
            words = words[: max_vocab - len(SPECIAL_TOKENS)]
        return cls(list(SPECIAL_TOKENS) + words, lowercase=lowercase)

    @classmethod
    def train_from_file(
        cls,
        path: str | Path,
        *,
        lowercase: bool = False,
        min_frequency: int = 1,
        max_vocab: int | None = None,
    ) -> WordSpaceTokenizer:
        return cls.train(
            Path(path).read_text(encoding="utf-8"),
            lowercase=lowercase,
            min_frequency=min_frequency,
            max_vocab=max_vocab,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<eos>"]

    def tokenize(self, text: str) -> list[str]:
        normalized = text.lower() if self.lowercase else text
        return normalized.split()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [self._token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_document(self, text: str) -> list[int]:
        """Encode non-empty lines and place one EOS token after every line."""

        ids: list[int] = []
        for line in text.splitlines():
            words = self.tokenize(line)
            if not words:
                continue
            ids.extend(self._token_to_id.get(word, self.unk_id) for word in words)
            ids.append(self.eos_id)
        if not ids and text.strip():
            ids = self.encode(text, add_eos=True)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        words: list[str] = []
        for value in ids:
            index = int(value)
            token = self.tokens[index] if 0 <= index < self.vocab_size else "<unk>"
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "word_space",
            "version": 1,
            "lowercase": self.lowercase,
            "tokens": self.tokens,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> WordSpaceTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "word_space":
            raise ValueError("not a WordSpaceTokenizer file")
        return cls(
            tokens=list(payload["tokens"]),
            lowercase=bool(payload.get("lowercase", False)),
        )
