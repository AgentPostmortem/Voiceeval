"""Locale packs for mishearing and confirmation checks.

English is the default. A missing pack is a failed check, not a silent pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PACK_DIR = Path(__file__).parent / "locale_packs"


@dataclass(frozen=True)
class LocalePack:
    language: str
    confusable_pairs: tuple[tuple[str, str], ...]
    confirmation_phrases: tuple[str, ...]

    @property
    def confirm_re(self) -> re.Pattern[str]:
        joined = "|".join(self.confirmation_phrases)
        return re.compile(rf"\b(?:{joined})\b", re.I)


def load_pack(path: Path) -> LocalePack:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = tuple((str(a), str(b)) for a, b in data["confusable_pairs"])
    phrases = tuple(str(p) for p in data["confirmation_phrases"])
    return LocalePack(language=str(data["language"]), confusable_pairs=pairs, confirmation_phrases=phrases)


def load_builtin_packs() -> dict[str, LocalePack]:
    packs: dict[str, LocalePack] = {}
    for path in sorted(PACK_DIR.glob("*.json")):
        pack = load_pack(path)
        packs[pack.language] = pack
    return packs
