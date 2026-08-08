"""The failures a text eval cannot see.

Every check here is invisible if you read the transcript as text. That is the point: read a call
transcript and it looks fine, because transcripts do not have clocks and do not record what the
caller actually said before the STT mangled it.

The five that matter, roughly in order of how much they cost:

1. mis-hearing on a consequential value. "Fifty" and "fifteen" are one phoneme apart, and the
   agent will act on either with equal confidence. This is the expensive one.
2. acting without confirming. A text agent that misreads can be corrected on the next line. A
   voice agent that mishears and immediately acts has already spent the money.
3. latency. Three seconds of silence on a phone call is a failed call, even if the answer that
   eventually arrives is perfect. Nobody notices this in a transcript.
4. talking over the caller. Barge-in that the agent ignores makes it feel broken.
5. dead air. Silence with nobody speaking is where callers hang up.

Language-sensitive checks (misheard, confirmation) read a :class:`LocalePack`. English ships by
default; suites can pass their own. A call whose language has no pack loaded reports that those
checks did not run — it is not counted as a clean pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .locale import LocalePack, load_builtin_packs, resolve_pack
from .turns import Interaction, Turn

# English number tokens used by misheard_number. Domain-specific number words belong in a
# custom locale pack's confusable_pairs; this keeps the classic English STT trap working.
_NUMERIC = re.compile(
    r"\b\d+(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b",
    re.I,
)


@dataclass
class Finding:
    check: str
    severity: str  # high | medium | low
    message: str
    turn_index: int | None = None


def check_misheard(inter: Interaction, pack: LocalePack | None = None) -> list[Finding]:
    """STT heard something different from what was said.

    Only detectable when you have ground truth, which is exactly why scripted test calls are worth
    the effort: in production this failure is silent by construction.

    Requires a locale pack for ``inter.language``. Without one the check reports that it did not
    run (high severity) instead of returning an empty list that looks like a pass.
    """
    if pack is None:
        return [
            Finding(
                "locale_unavailable",
                "high",
                f"No locale pack for language {inter.language!r}; misheard check did not run. "
                "Load a pack or set language to a supported locale (e.g. 'en').",
            )
        ]

    out: list[Finding] = []
    for i, t in enumerate(inter.turns):
        if t.speaker != "user" or not t.truth:
            continue
        if _norm(t.text) == _norm(t.truth):
            continue

        # English digit/word numbers first so default-pack behaviour stays byte-identical.
        heard_nums = set(m.group(0).lower() for m in _NUMERIC.finditer(t.text))
        said_nums = set(m.group(0).lower() for m in _NUMERIC.finditer(t.truth))
        if heard_nums != said_nums:
            out.append(
                Finding(
                    "misheard_number",
                    "high",
                    f"STT heard {sorted(heard_nums) or '[]'} but caller said {sorted(said_nums) or '[]'}. "
                    "A wrong number the agent acts on is the most expensive failure in voice.",
                    i,
                )
            )
            continue

        # Pack confusable pairs catch non-English (and domain) traps the English numeric list misses.
        if _confusable_mismatch(t.text, t.truth, pack):
            out.append(
                Finding(
                    "misheard_number",
                    "high",
                    f"STT confusable mismatch: heard {t.text!r} but caller said {t.truth!r}. "
                    "A wrong number the agent acts on is the most expensive failure in voice.",
                    i,
                )
            )
            continue

        out.append(
            Finding("misheard", "medium", f"STT differs from truth: {t.text!r} vs {t.truth!r}", i)
        )
    return out


def check_acted_without_confirming(
    inter: Interaction, pack: LocalePack | None = None
) -> list[Finding]:
    """A consequential action with no confirmation anywhere before it.

    In text, a misunderstanding costs one turn. In voice, it costs the refund.
    """
    if pack is None:
        return [
            Finding(
                "locale_unavailable",
                "high",
                f"No locale pack for language {inter.language!r}; confirmation check did not run. "
                "Load a pack or set language to a supported locale (e.g. 'en').",
            )
        ]

    out: list[Finding] = []
    for i, t in enumerate(inter.turns):
        consequential = [a for a in t.actions if a.consequential]
        if not consequential:
            continue
        confirmed = any(
            pack.confirm_re.search(p.text) for p in inter.turns[:i] if p.speaker == "agent"
        )
        if not confirmed:
            names = ", ".join(a.name for a in consequential)
            out.append(
                Finding(
                    "no_confirmation",
                    "high",
                    f"Took consequential action ({names}) without ever confirming. "
                    "If the STT misheard, this already happened.",
                    i,
                )
            )
    return out


def check_policy_violation(inter: Interaction) -> list[Finding]:
    """The agent did something the policy forbids.

    Reads limits from interaction.policy rather than hardcoding: what is allowed is a business
    decision. The check is whether the agent respected it.
    """
    out: list[Finding] = []
    max_refund = inter.policy.get("max_refund")
    if max_refund is None:
        return out
    for i, t in enumerate(inter.turns):
        for a in t.actions:
            amount = a.args.get("amount")
            if a.name == "refund" and isinstance(amount, (int, float)) and amount > max_refund:
                out.append(
                    Finding(
                        "policy_violation",
                        "high",
                        f"Refunded {amount} with a policy cap of {max_refund}. "
                        "Policy belongs in code, not in the prompt.",
                        i,
                    )
                )
    return out


def check_latency(inter: Interaction, budget_s: float = 1.5) -> list[Finding]:
    """Silence between the caller finishing and the agent starting.

    Completely invisible in a transcript. A caller does not experience a correct answer that
    arrives three seconds late as a correct answer.
    """
    out: list[Finding] = []
    for user_turn, agent_turn in inter.pairs():
        gap = agent_turn.start_s - user_turn.end_s
        if gap > budget_s:
            idx = inter.turns.index(agent_turn)
            sev = "high" if gap > budget_s * 2 else "medium"
            out.append(
                Finding(
                    "slow_response",
                    sev,
                    f"{gap:.1f}s of silence before the agent replied (budget {budget_s}s).",
                    idx,
                )
            )
    return out


def check_talked_over_user(inter: Interaction) -> list[Finding]:
    """Overlapping speech: the agent kept going while the caller was talking.

    Barge-in handling is most of what makes a voice agent feel human or broken, and it does not
    exist as a concept in text.
    """
    out: list[Finding] = []
    for i in range(len(inter.turns) - 1):
        a, b = inter.turns[i], inter.turns[i + 1]
        if a.speaker == "agent" and b.speaker == "user" and b.start_s < a.end_s:
            overlap = a.end_s - b.start_s
            if overlap > 0.3:  # sub-300ms overlap is normal human turn-taking, not a fault
                out.append(
                    Finding(
                        "talked_over_user",
                        "medium",
                        f"Agent kept speaking {overlap:.1f}s after the caller started. Barge-in not handled.",
                        i,
                    )
                )
    return out


def check_dead_air(inter: Interaction, max_gap_s: float = 3.0) -> list[Finding]:
    """Nobody is speaking. This is where callers hang up."""
    out: list[Finding] = []
    for i in range(len(inter.turns) - 1):
        gap = inter.turns[i + 1].start_s - inter.turns[i].end_s
        if gap > max_gap_s:
            out.append(Finding("dead_air", "medium", f"{gap:.1f}s of silence in the call.", i))
    return out


def check_incomplete(inter: Interaction) -> list[Finding]:
    if not inter.completed:
        return [Finding("incomplete", "medium", "Call ended without reaching its goal.")]
    return []


# Checks that need a locale pack receive it as a second argument from analyse().
_LOCALE_CHECKS: list[Callable[..., list[Finding]]] = [
    check_misheard,
    check_acted_without_confirming,
]
_LANGUAGE_AGNOSTIC: list[Callable[[Interaction], list[Finding]]] = [
    check_policy_violation,
    check_latency,
    check_talked_over_user,
    check_dead_air,
    check_incomplete,
]

# Public list kept for callers that introspect the module (order ≈ cost).
CHECKS = [
    check_misheard,
    check_acted_without_confirming,
    check_policy_violation,
    check_latency,
    check_talked_over_user,
    check_dead_air,
    check_incomplete,
]


def analyse(
    inter: Interaction,
    *,
    packs: dict[str, LocalePack] | None = None,
    locale_pack: LocalePack | None = None,
) -> list[Finding]:
    """Run all checks.

    ``locale_pack`` forces a single pack. Otherwise the pack is resolved from
    ``inter.language`` against ``packs`` (default: shipped builtins).
    """
    available = packs if packs is not None else load_builtin_packs()
    pack = resolve_pack(inter.language, packs=available, pack=locale_pack)

    out: list[Finding] = []
    for check in _LOCALE_CHECKS:
        out.extend(check(inter, pack))
    for check in _LANGUAGE_AGNOSTIC:
        out.extend(check(inter))
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda f: (order.get(f.severity, 9), f.turn_index if f.turn_index is not None else -1))
    return out


def _confusable_mismatch(heard: str, truth: str, pack: LocalePack) -> bool:
    """True when a confusable pair is split across heard vs truth."""
    for a_re, b_re in pack.confusable_res:
        a_in_heard, b_in_heard = bool(a_re.search(heard)), bool(b_re.search(heard))
        a_in_truth, b_in_truth = bool(a_re.search(truth)), bool(b_re.search(truth))
        if (a_in_heard and b_in_truth and not b_in_heard) or (
            b_in_heard and a_in_truth and not a_in_heard
        ):
            return True
    return False


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
