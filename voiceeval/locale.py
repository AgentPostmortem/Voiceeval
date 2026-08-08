"""Locale packs for language-sensitive checks.

Mis-hearing and confirmation vocabularies are not universal. Hardcoding English
means a Spanish call silently reports zero findings — the same failure shape
this project exists to catch. Packs keep English editable and let suites ship
their own domain terms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


@dataclass(frozen=True)
class LocalePack:
    """Confusable terms and confirmation phrases for one language."""

    language: str
    confusable_pairs: tuple[tuple[str, str], ...] = ()
    confirmation_phrases: tuple[str, ...] = ()
    # Compiled helpers (not serialised)
    confirm_re: re.Pattern[str] = field(
        repr=False, hash=False, compare=False, default=re.compile(r"(?!x)x")
    )
    confusable_res: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = field(
        default=(), repr=False, hash=False, compare=False
    )

    @staticmethod
    def from_dict(data: dict) -> LocalePack:
        lang = str(data["language"]).lower()
        pairs = tuple((str(a), str(b)) for a, b in data.get("confusable_pairs", []))
        phrases = tuple(str(p) for p in data.get("confirmation_phrases", []))
        if phrases:
            confirm_re = re.compile(r"\b(?:" + "|".join(phrases) + r")\b", re.I)
        else:
            confirm_re = re.compile(r"(?!x)x")  # matches nothing
        confusable_res = tuple(
            (
                re.compile(rf"\b{re.escape(a)}\b", re.I),
                re.compile(rf"\b{re.escape(b)}\b", re.I),
            )
            for a, b in pairs
        )
        return LocalePack(
            language=lang,
            confusable_pairs=pairs,
            confirmation_phrases=phrases,
            confirm_re=confirm_re,
            confusable_res=confusable_res,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> LocalePack:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_builtin_packs(directory: Path | None = None) -> dict[str, LocalePack]:
    """Load every ``*.json`` pack from the shipped locales directory."""
    root = directory or _LOCALES_DIR
    packs: dict[str, LocalePack] = {}
    if not root.is_dir():
        return packs
    for path in sorted(root.glob("*.json")):
        pack = LocalePack.from_path(path)
        packs[pack.language] = pack
    return packs


def resolve_pack(
    language: str,
    *,
    packs: dict[str, LocalePack] | None = None,
    pack: LocalePack | None = None,
) -> LocalePack | None:
    """Return the pack to use, or None if the language has no pack loaded."""
    if pack is not None:
        return pack
    available = packs if packs is not None else load_builtin_packs()
    return available.get((language or "en").lower())
