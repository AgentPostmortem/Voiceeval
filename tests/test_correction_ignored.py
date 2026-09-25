"""Tests for check_correction_ignored."""

import pytest
from voiceeval.checks import check_correction_ignored
from voiceeval.turns import Interaction, Turn


def make_call(agent_init, user_corr, agent_followup):
    """Helper to build a 3-turn sequence: agent proposes -> user corrects -> agent responds."""
    turns = [
        Turn("agent", agent_init, 0.0, 1.0),
        Turn("user", user_corr, 1.2, 2.0),
        Turn("agent", agent_followup, 2.2, 3.2),
    ]
    return Interaction("correction_test", turns)


def test_correction_ignored_detected():
    """Agent repeats the outdated number 50 after caller corrects to 15."""
    inter = make_call(
        agent_init="You want to refund fifty dollars, correct?",
        user_corr="No, I said fifteen dollars, not fifty.",
        agent_followup="Alright, processing refund for 50 dollars.",
    )
    findings = check_correction_ignored(inter)
    assert len(findings) == 1
    assert findings[0].check == "correction_ignored"
    assert findings[0].severity == "high"
    assert findings[0].turn_index == 2


def test_correction_respected_clean():
    """Agent accepts the correction and uses the new number 15; no findings expected."""
    inter = make_call(
        agent_init="You want to refund fifty dollars, correct?",
        user_corr="No, actually fifteen.",
        agent_followup="Got it, processing fifteen dollars now.",
    )
    findings = check_correction_ignored(inter)
    assert len(findings) == 0


@pytest.mark.parametrize(
    "phrase",
    [
        "No, actually fifteen.",
        "Wrong, I said fifteen.",
        "Not that is wrong, make it fifteen instead.",
    ],
)
def test_correction_phrases_trigger(phrase):
    """Various correction expressions should trigger detection if ignored."""
    inter = make_call(
        agent_init="Confirming 50?",
        user_corr=phrase,
        agent_followup="Okay, confirming 50.",
    )
    findings = check_correction_ignored(inter)
    assert len(findings) == 1
    assert findings[0].check == "correction_ignored"