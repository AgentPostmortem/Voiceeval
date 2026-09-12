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
from decimal import Decimal, InvalidOperation
from itertools import pairwise

from .turns import Interaction


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
]

_NUMERIC = re.compile(
    r"\b\d+(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b",
    re.I,
)

_CONFIRM = re.compile(
    r"\b(?:just to confirm|confirm(?:ing)?|did you say|is that right|correct(?=\?)|to be clear|"
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
        else:
            out.append(
                Finding(
                    "misheard", "medium", f"STT differs from truth: {t.text!r} vs {t.truth!r}", i
                )
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
        confirmed = any(_CONFIRM.search(p.text) for p in inter.turns[:i] if p.speaker == "agent")
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
    normalized_max_refund = _finite_decimal(max_refund)
    if normalized_max_refund is None:
        return out
    for i, t in enumerate(inter.turns):
        for a in t.actions:
            amount = a.args.get("amount")
            normalized_amount = _finite_decimal(amount)
            if (
                a.name == "refund"
                and normalized_amount is not None
                and normalized_amount > normalized_max_refund
            ):
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


def _finite_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


_WORD_TO_NUMBER = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
}


def _numeral_values(text: str) -> set[Decimal] | None:
    """Normalize isolated English number words and unsigned decimal digits.

    Skip compound/signed/grouped forms rather than splitting, for example, "twenty five"
    into two unrelated amounts. This is deliberately not a general number-language parser.
    """
    matches = list(_NUMERIC.finditer(text))
    for previous, current in pairwise(matches):
        if re.fullmatch(
            r"[\s,.-]*(?:and\s+)?", text[previous.end() : current.start()], re.IGNORECASE
        ):
            return None
    if any(m.start() and text[m.start() - 1] in "+-" for m in matches):
        return None
    return {Decimal(_WORD_TO_NUMBER.get(m.group().lower(), m.group())) for m in matches}


def _confirmation_action_amounts(inter: Interaction, index: int) -> set[Decimal]:
    """Use only the first consequential action turn in this confirmation exchange."""
    for j in range(index, len(inter.turns)):
        turn = inter.turns[j]
        if turn.speaker == "user":
            # A new request or correction invalidates the association; an acknowledgement
            # can sit between confirmation and action in the usual refund call shape.
            if not re.fullmatch(
                r"(?:yes|yeah|yep|ok|okay|correct|right)[.!\s]*", turn.text, re.IGNORECASE
            ):
                break
        elif j != index and _CONFIRM.search(turn.text):
            break
        actions = [a for a in turn.actions if a.consequential]
        if actions:
            return {
                amount
                for action in actions
                for amount in [_finite_decimal(action.args.get("amount"))]
                if amount is not None
            }
    return set()


def check_confirmed_wrong_value(inter: Interaction) -> list[Finding]:
    """Flag an agent confirmation echoing a current STT value contradicted by evidence.

    Ground truth can expose a wrong confirmation immediately, even before a caller agrees
    or an action occurs. Without truth, a single confirmed value can still contradict the
    next action amount. Neither path infers whether the caller actually accepted the number.
    """
    out: list[Finding] = []
    for i, user_turn in enumerate(inter.turns):
        if user_turn.speaker != "user":
            continue
        heard = _numeral_values(user_turn.text)
        said = _numeral_values(user_turn.truth) if user_turn.truth else set()
        if not heard or said is None:
            continue

        for j in range(i + 1, len(inter.turns)):
            agent_turn = inter.turns[j]
            if agent_turn.speaker == "user":
                break
            if not _CONFIRM.search(agent_turn.text):
                continue
            confirmed = _numeral_values(agent_turn.text)
            if not confirmed:
                continue
            echoed = confirmed & heard
            wrong = echoed - said if user_turn.truth else set()
            message = None
            if wrong:
                message = (
                    f"Confirmation echoed STT value(s) {sorted(map(str, wrong))} absent from "
                    f"caller truth {sorted(map(str, said))}. "
                    "Caller agreement does not make the number right."
                )
            elif echoed and len(confirmed) == len(heard) == 1:
                amounts = _confirmation_action_amounts(inter, j)
                # Multiple numbers cannot be assigned to an amount without field semantics.
                if len(amounts) == 1 and confirmed != amounts:
                    message = (
                        f"Confirmation echoed STT {sorted(map(str, echoed))} but the next "
                        f"consequential action amount was {sorted(map(str, amounts))}."
                    )
            if message:
                out.append(Finding("confirmed_wrong_value", "high", message, j))
                break
    return out


def check_latency(inter: Interaction, budget_s: float = 1.5) -> list[Finding]:
    """Silence between the caller finishing and the agent starting.

    Completely invisible in a transcript. A caller does not experience a correct answer that
    arrives three seconds late as a correct answer.
    """
    out: list[Finding] = []
    for user_turn, agent_turn in inter.pairs():
        gap = agent_turn.start_s - user_turn.end_s
        if gap < 0:
            idx = inter.turns.index(agent_turn)
            out.append(
                Finding(
                    "invalid_timing",
                    "high",
                    f"Agent started {abs(gap):.1f}s before the caller finished; timestamps overlap.",
                    idx,
                )
            )
            continue
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
    check_confirmed_wrong_value,
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
    out.sort(
        key=lambda f: (order.get(f.severity, 9), f.turn_index if f.turn_index is not None else -1)
    )
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
