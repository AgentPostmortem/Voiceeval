"""Wrong-number confirmations must stay tied to the caller's current request."""

from pathlib import Path

import pytest

from voiceeval.checks import analyse, check_confirmed_wrong_value
from voiceeval.score import score
from voiceeval.turns import Action, Interaction, Turn, load


def call(heard, truth, confirmation, amount=None, later=()):
    turns = [Turn("user", heard, 0, 1, truth), Turn("agent", confirmation, 1.1, 2)]
    if amount is not None:
        turns += [
            Turn("user", "yes", 2.1, 2.4),
            Turn("agent", "Done", 2.5, 3, actions=[Action("refund", {"amount": amount}, True)]),
        ]
    return Interaction("confirmation", turns + list(later))


@pytest.mark.parametrize(
    "confirmation", ["Confirming fifty?", "Fifty, correct?", "Did you say 50?"]
)
def test_confirmation_phrases(confirmation):
    findings = check_confirmed_wrong_value(call("fifty", "fifteen", confirmation, 50))
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert findings[0].turn_index == 1


@pytest.mark.parametrize(
    "heard,truth,confirmation",
    [
        ("50", "15", "Confirm fifty?"),
        ("fifty", "15", "Confirm 50.00?"),
        ("50.25", "15.25", "Confirm 50.250?"),
        ("FIFTY", "FIFTEEN", "CONFIRM FIFTY?"),
        ("50 for order 123", "15 for order 123", "Confirm 50 for order 123?"),
        ("refund 50", "refund please", "Confirm 50?"),
    ],
)
def test_wrong_values(heard, truth, confirmation):
    assert len(check_confirmed_wrong_value(call(heard, truth, confirmation))) == 1


@pytest.mark.parametrize(
    "heard,truth,confirmation",
    [
        ("fifty", "fifteen", "Confirm fifteen?"),
        ("fifty", "fifty", "Confirm fifty?"),
        ("50.00", "fifty", "Confirm 50?"),
        ("fifty", None, "Confirm fifty?"),
        ("fifty", "", "Confirm fifty?"),
        ("refund shoes", "refund shirt", "Confirm refund?"),
        ("50 for order 123", "15 for order 123", "Confirm order 123?"),
        ("fifty", "fifteen", "Refunding fifty now."),
        ("fifty", "fifteen", "Confirm refund?"),
        ("fifty", "fifteen", "Confirm sixty?"),
    ],
)
def test_clean_or_insufficient_evidence(heard, truth, confirmation):
    assert check_confirmed_wrong_value(call(heard, truth, confirmation)) == []


def test_new_user_request_ends_old_mismatch():
    inter = call(
        "fifty",
        "fifteen",
        "I heard you.",
        later=[
            Turn("user", "Actually make it fifty", 2.1, 3, "Actually make it fifty"),
            Turn("agent", "Confirm fifty?", 3.1, 4),
        ],
    )
    assert check_confirmed_wrong_value(inter) == []


def test_does_not_duplicate_one_confirmation_for_old_requests():
    inter = Interaction(
        "repeated",
        [
            Turn("user", "fifty", 0, 1, "fifteen"),
            Turn("user", "fifty", 1.1, 2, "fifteen"),
            Turn("agent", "Confirm fifty?", 2.1, 3),
        ],
    )
    findings = check_confirmed_wrong_value(inter)
    assert len(findings) == 1
    assert findings[0].turn_index == 2


def test_can_confirm_after_agent_filler():
    inter = call(
        "fifty",
        "fifteen",
        "Let me check.",
        later=[
            Turn("agent", "Confirm fifty?", 2.1, 3),
        ],
    )
    assert len(check_confirmed_wrong_value(inter)) == 1


def test_action_mismatch_without_truth():
    findings = check_confirmed_wrong_value(call("50", None, "Confirm fifty?", "15"))
    assert len(findings) == 1
    assert "action" in findings[0].message.lower()


def test_equivalent_spelling_does_not_pull_in_later_action():
    inter = call(
        "15",
        "fifteen",
        "Confirm 15?",
        later=[
            Turn("user", "Now refund twenty for the other item", 2.1, 3),
            Turn("agent", "Done", 3.1, 4, actions=[Action("refund", {"amount": 20}, True)]),
        ],
    )
    assert check_confirmed_wrong_value(inter) == []


@pytest.mark.parametrize("amount", [True, "NaN", "Infinity", "", "not an amount", {}, []])
def test_invalid_action_amount_is_ignored(amount):
    assert check_confirmed_wrong_value(call("50", None, "Confirm fifty?", amount)) == []


@pytest.mark.parametrize(
    "value", ["twenty five", "twenty-five", "two hundred", "1,000", "-50", "+50"]
)
def test_unsupported_numeric_forms_are_not_split_into_unrelated_values(value):
    assert check_confirmed_wrong_value(call(value, "25", "Confirm " + value + "?")) == []


def test_headline_fixture_and_corrected_confirmation():
    inter = load(Path(__file__).resolve().parents[1] / "fixtures/confirmed_wrong_value_call.json")
    findings = check_confirmed_wrong_value(inter)
    assert len(findings) == 1
    assert findings[0].turn_index == 2
    assert findings[0].severity == "high"
    assert "no_confirmation" not in {f.check for f in analyse(inter)}
    assert score(inter).passed is False
    assert any(f["check"] == "confirmed_wrong_value" for f in score(inter).findings)
    inter.turns[2].text = "Just to confirm, fifteen dollars?"
    inter.turns[4].actions[0].args["amount"] = 15
    assert check_confirmed_wrong_value(inter) == []


def test_action_amount_does_not_compare_an_order_number():
    inter = call("refund 50 for order 123", "refund 50 for order 123", "Confirm order 123?", 50)
    assert check_confirmed_wrong_value(inter) == []


def test_action_on_confirmation_turn():
    inter = call("50", None, "Confirm 50?")
    inter.turns[1].actions = [Action("refund", {"amount": 15}, True)]
    assert len(check_confirmed_wrong_value(inter)) == 1


def test_next_confirmation_stops_action_association():
    inter = call(
        "50",
        None,
        "Confirm 50?",
        later=[
            Turn("agent", "Actually confirm 15?", 2.1, 3),
            Turn("agent", "Done", 3.1, 4, actions=[Action("refund", {"amount": 15}, True)]),
        ],
    )
    assert check_confirmed_wrong_value(inter) == []


def test_non_consequential_and_ambiguous_actions_are_not_amount_evidence():
    inter = call("50", None, "Confirm 50?")
    inter.turns[1].actions = [Action("lookup", {"amount": 15}, False)]
    assert check_confirmed_wrong_value(inter) == []
    inter.turns[1].actions = [Action("refund", {"amount": a}, True) for a in (15, 20)]
    assert check_confirmed_wrong_value(inter) == []


def test_shared_confirmation_phrases_prevent_missing_confirmation():
    for phrase in ("Confirming fifty?", "Fifty, correct?"):
        inter = call("fifty", "fifteen", phrase, 50)
        assert "no_confirmation" not in {f.check for f in analyse(inter)}
