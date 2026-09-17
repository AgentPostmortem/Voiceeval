import pytest

from voiceeval.turns import from_dict


def _turn(**overrides) -> dict:
    turn = {"speaker": "user", "text": "hello", "start_s": 0.0, "end_s": 1.0}
    turn.update(overrides)
    return turn


@pytest.mark.parametrize(
    "turn,field",
    [
        ({}, "speaker"),
        (_turn(speaker="robot"), "speaker"),
        (_turn(text=None), "text"),
        ({"speaker": "user", "text": "hello", "end_s": 1.0}, "start_s"),
        (_turn(start_s="soon"), "start_s"),
        (_turn(start_s=None), "start_s"),
        ({"speaker": "user", "text": "hello", "start_s": 0.0}, "end_s"),
        (_turn(end_s="gone"), "end_s"),
    ],
)
def test_from_dict_malformed_turn_raises_descriptive_value_error(turn, field):
    with pytest.raises(ValueError, match=f"turn 0.*{field}"):
        from_dict({"turns": [turn]})


def test_from_dict_malformed_second_turn_names_turn_index():
    turns = [_turn(), {"speaker": "agent", "text": "ok", "start_s": "later", "end_s": 5.0}]
    with pytest.raises(ValueError, match="turn 1.*start_s"):
        from_dict({"turns": turns})


def test_from_dict_missing_turns_key_is_descriptive():
    with pytest.raises(ValueError, match="turns"):
        from_dict({})