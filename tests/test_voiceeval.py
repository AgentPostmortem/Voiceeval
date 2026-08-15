"""Tests.

Each test pins one failure that a text eval would score as a perfect call. That is the claim this
library makes, so it is the claim the tests have to defend.
"""

from __future__ import annotations

from pathlib import Path

from voiceeval.checks import analyse
from voiceeval.score import diff, run_suite, score
from voiceeval.turns import Action, Interaction, Turn, from_dict, load

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _codes(inter: Interaction) -> set[str]:
    return {f.check for f in analyse(inter)}


def _t(speaker, text, start, end, truth=None, actions=None) -> Turn:
    return Turn(speaker=speaker, text=text, start_s=start, end_s=end, truth=truth, actions=actions or [])


# --------------------------------------------------------------------------- the headline case


def test_misheard_number_is_caught_and_is_high_severity():
    """The expensive one.

    Read this call as text and it is flawless: the caller asked for fifty, the agent refunded
    fifty. The transcript agrees with itself. Only ground truth reveals the caller said fifteen.
    """
    inter = load(FIXTURES / "misheard_call.json")
    findings = analyse(inter)
    misheard = [f for f in findings if f.check == "misheard_number"]
    assert misheard, "STT heard 'fifty' when the caller said 'fifteen' and it was not caught"
    assert misheard[0].severity == "high"


def test_misheard_call_also_flags_acting_without_confirmation():
    """The mis-hearing is only expensive because nothing confirmed it before the money moved."""
    inter = load(FIXTURES / "misheard_call.json")
    assert "no_confirmation" in _codes(inter)


def test_misheard_call_fails_the_suite():
    inter = load(FIXTURES / "misheard_call.json")
    assert score(inter).passed is False


def test_spanish_locale_pack_catches_quince_cincuenta():
    inter = load(FIXTURES / "misheard_es.json")
    findings = analyse(inter)
    assert any(f.check == "misheard_number" for f in findings)
    assert "locale_unavailable" not in {f.check for f in findings}


def test_missing_locale_pack_is_not_a_silent_pass():
    inter = load(FIXTURES / "good_call.json")
    inter.language = "hi"
    findings = analyse(inter)
    assert findings and findings[0].check == "locale_unavailable"
    assert score(inter).passed is False


def test_custom_locale_pack_overrides_confirmation():
    from voiceeval.locales import LocalePack

    pack = LocalePack(
        language="en",
        confusable_pairs=( (r"\bfifteen\b", r"\bfifty\b"), ),
        confirmation_phrases=("please double-check",),
    )
    inter = Interaction(
        id="custom",
        turns=[
            _t("user", "refund fifteen", 0, 1, truth="refund fifteen"),
            _t("agent", "Just to confirm, fifteen?", 1.2, 2),
            _t("agent", "Done.", 2.2, 3, actions=[Action("refund", {"amount": 15}, True)]),
        ],
    )
    assert any(f.check == "no_confirmation" for f in analyse(inter, locale_pack=pack))
    assert "no_confirmation" not in {f.check for f in analyse(inter)}


def test_a_clean_call_passes_with_no_findings():
    """Guards against the checker crying wolf. The good call confirms before refunding, replies
    promptly, and the STT matches ground truth."""
    inter = load(FIXTURES / "good_call.json")
    assert analyse(inter) == []
    assert score(inter).passed is True


# --------------------------------------------------------------------------- individual checks


def test_confirmation_prevents_the_no_confirmation_finding():
    inter = Interaction(
        id="t",
        turns=[
            _t("user", "refund me twenty dollars", 0, 2, truth="refund me twenty dollars"),
            _t("agent", "Just to confirm, twenty dollars?", 2.2, 4),
            _t("user", "yes", 4.2, 4.6, truth="yes"),
            _t("agent", "Done.", 4.8, 5.4, actions=[Action("refund", {"amount": 20}, True)]),
        ],
        policy={"max_refund": 50},
    )
    assert "no_confirmation" not in _codes(inter)


def test_non_consequential_action_needs_no_confirmation():
    """A lookup is not a refund. Demanding confirmation for reads would make the agent unusable."""
    inter = Interaction(
        id="t",
        turns=[
            _t("user", "where is my order", 0, 2, truth="where is my order"),
            _t("agent", "It shipped Tuesday.", 2.2, 4, actions=[Action("lookup_order", {}, False)]),
        ],
    )
    assert "no_confirmation" not in _codes(inter)


def test_policy_violation_is_caught():
    """Policy lives in the interaction, not in this library: what is allowed is a business rule."""
    inter = Interaction(
        id="t",
        turns=[
            _t("user", "refund two hundred", 0, 2, truth="refund two hundred"),
            _t("agent", "Just to confirm, two hundred?", 2.1, 3.5),
            _t("user", "yes", 3.7, 4.0, truth="yes"),
            _t("agent", "Done.", 4.2, 5.0, actions=[Action("refund", {"amount": 200}, True)]),
        ],
        policy={"max_refund": 50},
    )
    assert "policy_violation" in _codes(inter)


def test_slow_response_is_caught():
    """Invisible in a transcript. A four-second silence is a failed call."""
    inter = Interaction(
        id="t",
        turns=[
            _t("user", "hello", 0, 1, truth="hello"),
            _t("agent", "Hi there.", 5.0, 6.0),
        ],
    )
    findings = [f for f in analyse(inter) if f.check == "slow_response"]
    assert findings and findings[0].severity == "high", "4s gap should be high, it is 2x+ the budget"


def test_prompt_response_is_not_flagged():
    inter = Interaction(
        id="t", turns=[_t("user", "hello", 0, 1, truth="hello"), _t("agent", "Hi.", 1.3, 2.0)]
    )
    assert "slow_response" not in _codes(inter)


def test_talking_over_user_is_caught():
    inter = Interaction(
        id="t",
        turns=[
            _t("agent", "Let me read you the whole policy...", 0, 8),
            _t("user", "stop", 5.0, 5.5, truth="stop"),
        ],
    )
    assert "talked_over_user" in _codes(inter)


def test_tiny_overlap_is_normal_turn_taking_not_a_fault():
    """Humans overlap by a couple hundred milliseconds constantly. Flagging that is noise."""
    inter = Interaction(
        id="t",
        turns=[
            _t("agent", "Anything else?", 0, 2.0),
            _t("user", "no thanks", 1.9, 3.0, truth="no thanks"),
        ],
    )
    assert "talked_over_user" not in _codes(inter)


def test_dead_air_is_caught():
    inter = load(FIXTURES / "slow_call.json")
    assert "dead_air" in _codes(inter)


def test_incomplete_call_is_flagged():
    inter = load(FIXTURES / "slow_call.json")
    assert "incomplete" in _codes(inter)


def test_misheard_without_ground_truth_is_undetectable():
    """Documents the limitation honestly.

    Without `truth`, mis-hearing is invisible by construction. This is why scripted test calls
    earn their keep, and why this library cannot save you in production on its own.
    """
    inter = Interaction(
        id="t",
        turns=[
            _t("user", "refund fifty", 0, 2),  # no truth field
            _t("agent", "Just to confirm, fifty?", 2.1, 3.5),
            _t("user", "yes", 3.7, 4.0),
            _t("agent", "Done.", 4.2, 5.0, actions=[Action("refund", {"amount": 50}, True)]),
        ],
        policy={"max_refund": 50},
    )
    assert "misheard_number" not in _codes(inter)




def test_non_numeric_stt_mismatch_is_medium_misheard_not_number():
    """STT can mangle wording without touching amounts.

    When ground truth differs but the *numbers* still match, the expensive
    misheard_number path must not fire — the code has a separate medium
    `misheard` finding for that case. That branch was untested.
    """
    inter = Interaction(
        id="t",
        turns=[
            _t(
                "user",
                "refund fifteen dollars for the shirt please",
                0,
                2,
                truth="refund fifteen dollars for the shoes please",
            ),
            _t("agent", "Just to confirm, fifteen dollars?", 2.2, 4.0),
            _t("user", "yes", 4.2, 4.6, truth="yes"),
            _t(
                "agent",
                "Done.",
                4.8,
                5.4,
                actions=[Action("refund", {"amount": 15}, True)],
            ),
        ],
        policy={"max_refund": 50},
    )
    findings = analyse(inter)
    codes = {f.check for f in findings}
    assert "misheard_number" not in codes
    misheard = [f for f in findings if f.check == "misheard"]
    assert len(misheard) == 1
    assert misheard[0].severity == "medium"
    assert misheard[0].turn_index == 0

# --------------------------------------------------------------------------- regression diff


def test_diff_detects_a_regression():
    """The point of the whole thing: did my prompt change make it worse?"""
    good = load(FIXTURES / "good_call.json")
    bad = load(FIXTURES / "misheard_call.json")
    bad.id = good.id  # same test case, worse behaviour after a change

    before = run_suite([good], label="v1")
    after = run_suite([bad], label="v2")
    d = diff(before, after)

    assert d["verdict"] == "REGRESSION"
    assert good.id in d["regressed"]
    assert d["pass_rate_before"] == 1.0
    assert d["pass_rate_after"] == 0.0


def test_diff_detects_a_fix():
    good = load(FIXTURES / "good_call.json")
    bad = load(FIXTURES / "misheard_call.json")
    bad.id = good.id
    d = diff(run_suite([bad], "v1"), run_suite([good], "v2"))
    assert d["verdict"] == "IMPROVED"


def test_diff_reports_no_change_when_nothing_moved():
    good = load(FIXTURES / "good_call.json")
    d = diff(run_suite([good], "v1"), run_suite([good], "v2"))
    assert d["verdict"] == "NO CHANGE"


def test_from_dict_roundtrip():
    inter = from_dict(
        {
            "id": "x",
            "turns": [{"speaker": "user", "text": "hi", "start_s": 0, "end_s": 1}],
        }
    )
    assert inter.turns[0].speaker == "user"
    assert inter.policy == {}
