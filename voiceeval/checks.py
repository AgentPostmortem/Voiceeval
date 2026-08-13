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
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .turns import Interaction, Turn


@dataclass
class Finding:
    check: str
    severity: str  # high | medium | low
    message: str
    turn_index: int | None = None


# Numbers that STT reliably confuses. The -teen/-ty pairs are the classic: one unstressed syllable
# apart, and both are plausible amounts, so neither the model nor a human reviewer notices.
_CONFUSABLE = [
    (r"\bfifteen\b", r"\bfifty\b"),
    (r"\bsixteen\b", r"\bsixty\b"),
    (r"\bseventeen\b", r"\bseventy\b"),
    (r"\beighteen\b", r"\beighty\b"),
    (r"\bnineteen\b", r"\bninety\b"),
    (r"\bthirteen\b", r"\bthirty\b"),
    (r"\bfourteen\b", r"\bforty\b"),
    # Card / account readback: unstressed "oh" and "zero" are the same vowel.
    (r"\boh\b", r"\bzero\b"),
    # Same one-phoneme collapse as fifteen/fifty.
    (r"\ba\b", r"\beight\b"),
    # Amounts and quantities: "to"/"too" vs "two".
    (r"\bto\b", r"\btwo\b"),
    (r"\btoo\b", r"\btwo\b"),
    # How people read repeated digits aloud vs the -ty form.
    (r"\bdouble\s+one\b", r"\btwenty[\s-]+one\b"),
    (r"\bdouble\s+two\b", r"\btwenty[\s-]+two\b"),
    (r"\bdouble\s+three\b", r"\bthirty[\s-]+three\b"),
    (r"\bdouble\s+four\b", r"\bforty[\s-]+four\b"),
    (r"\bdouble\s+five\b", r"\bfifty[\s-]+five\b"),
    (r"\bdouble\s+six\b", r"\bsixty[\s-]+six\b"),
    (r"\bdouble\s+seven\b", r"\bseventy[\s-]+seven\b"),
    (r"\bdouble\s+eight\b", r"\beighty[\s-]+eight\b"),
    (r"\bdouble\s+nine\b", r"\bninety[\s-]+nine\b"),
    # Narrowband telephony: these letters collapse on a phone line and show
    # up in reference codes and postcodes.
    (r"\bb\b", r"\bd\b"),
    (r"\bb\b", r"\be\b"),
    (r"\bb\b", r"\bp\b"),
    (r"\bb\b", r"\bt\b"),
    (r"\bb\b", r"\bv\b"),
    (r"\bb\b", r"\bz\b"),
    (r"\bd\b", r"\be\b"),
    (r"\bd\b", r"\bt\b"),
    (r"\be\b", r"\bp\b"),
    (r"\be\b", r"\bt\b"),
    (r"\be\b", r"\bv\b"),
    (r"\bp\b", r"\bt\b"),
    (r"\bp\b", r"\bv\b"),
    (r"\bt\b", r"\bv\b"),
    (r"\bt\b", r"\bz\b"),
    (r"\bv\b", r"\bz\b"),
]

_NUMERIC = re.compile(r"\b\d+(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                      r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
                      r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b", re.I)

_CONFIRM = re.compile(
    r"\b(?:just to confirm|confirm|did you say|is that right|correct\?|to be clear|"
    r"you'?d like|shall I|should I|can I go ahead)\b",
    re.I,
)


def check_misheard(inter: Interaction) -> list[Finding]:
    """STT heard something different from what was said.

    Only detectable when you have ground truth, which is exactly why scripted test calls are worth
    the effort: in production this failure is silent by construction.
    """
    out: list[Finding] = []
    for i, t in enumerate(inter.turns):
        if t.speaker != "user" or not t.truth:
            continue
        if _norm(t.text) == _norm(t.truth):
            continue

        heard_nums = set(m.group(0).lower() for m in _NUMERIC.finditer(t.text))
        said_nums = set(m.group(0).lower() for m in _NUMERIC.finditer(t.truth))
        pair = _confusable_hit(t.text, t.truth)
        if heard_nums != said_nums or pair:
            detail = (
                f"STT heard {sorted(heard_nums) or '[]'} but caller said {sorted(said_nums) or '[]'}."
                if heard_nums != said_nums
                else f"STT swapped a high-cost pair ({pair}): {t.text!r} vs {t.truth!r}."
            )
            out.append(
                Finding(
                    "misheard_number",
                    "high",
                    f"{detail} A wrong number the agent acts on is the most expensive failure in voice.",
                    i,
                )
            )
        else:
            out.append(
                Finding("misheard", "medium", f"STT differs from truth: {t.text!r} vs {t.truth!r}", i)
            )
    return out


def check_acted_without_confirming(inter: Interaction) -> list[Finding]:
    """A consequential action with no confirmation anywhere before it.

    In text, a misunderstanding costs one turn. In voice, it costs the refund.
    """
    out: list[Finding] = []
    for i, t in enumerate(inter.turns):
        consequential = [a for a in t.actions if a.consequential]
        if not consequential:
            continue
        confirmed = any(
            _CONFIRM.search(p.text) for p in inter.turns[:i] if p.speaker == "agent"
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


CHECKS = [
    check_misheard,
    check_acted_without_confirming,
    check_policy_violation,
    check_latency,
    check_talked_over_user,
    check_dead_air,
    check_incomplete,
]


def analyse(inter: Interaction) -> list[Finding]:
    out: list[Finding] = []
    for check in CHECKS:
        out.extend(check(inter))
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda f: (order.get(f.severity, 9), f.turn_index if f.turn_index is not None else -1))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _confusable_hit(text: str, truth: str) -> str | None:
    """Return a label if truth and STT differ by a known acoustic pair."""
    for left, right in _CONFUSABLE:
        left_in_text = re.search(left, text, re.I)
        right_in_truth = re.search(right, truth, re.I)
        right_in_text = re.search(right, text, re.I)
        left_in_truth = re.search(left, truth, re.I)
        if (left_in_text and right_in_truth) or (right_in_text and left_in_truth):
            return f"{left} ↔ {right}"
    return None
