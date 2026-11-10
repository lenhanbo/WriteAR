from __future__ import annotations

import json
from pathlib import Path


SPECIALS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>", "<NULL>")


class CharacterVocabulary:
    def __init__(self, symbols: list[str] | tuple[str, ...]):
        unique = list(SPECIALS)
        unique.extend(ch for ch in symbols if ch not in unique)
        self.symbols = unique
        self.stoi = {ch: i for i, ch in enumerate(unique)}

    def __len__(self):
        return len(self.symbols)

    @property
    def pad_id(self): return self.stoi["<PAD>"]
    @property
    def bos_id(self): return self.stoi["<BOS>"]
    @property
    def eos_id(self): return self.stoi["<EOS>"]
    @property
    def null_id(self): return self.stoi["<NULL>"]

    def encode(self, text: str, boundaries: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, self.stoi["<UNK>"]) for ch in text]
        return ([self.bos_id] + ids + [self.eos_id]) if boundaries else ids

    def decode(self, ids) -> str:
        return "".join(self.symbols[int(i)] for i in ids if int(i) >= len(SPECIALS))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.symbols, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        symbols = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(symbols[len(SPECIALS):])

