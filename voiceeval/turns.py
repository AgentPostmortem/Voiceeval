"""The data model: a voice interaction as timed turns.

A text agent's transcript is a list of messages. A voice agent's transcript is a list of messages
*with clocks attached*, and the clocks are where the failures live.

This distinction is the entire reason this library exists. If you evaluate a voice agent by
reading its transcript, you are evaluating a text agent that happens to have been spoken. You will
score a call as perfect when the caller hung up during a four-second silence, or when the agent
confidently refunded fifteen dollars because it misheard fifty.

So every turn carries:
  - when it started and ended (latency, dead air, overlap)
  - what the STT *heard* vs what was *said*, when you have ground truth (mis-hearing)
  - what the agent *did*, not just what it said (actions are what cost money)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Speaker = Literal["user", "agent"]


@dataclass
class Action:
    """Something the agent did in the real world. This is what makes a mistake expensive."""

    name: str
    args: dict = field(default_factory=dict)
    # Consequential = costs money, moves data, or is hard to undo. Refunds, bookings, cancellations.
    # A lookup is not consequential. A charge is.
    consequential: bool = False


@dataclass
class Turn:
    speaker: Speaker
    text: str  # what the STT heard (agent turns: what it said)
    start_s: float
    end_s: float
    # What was actually said, when you have ground truth (a scripted test call, a human label).
    # Its absence is why mis-hearing goes undetected in production.
    truth: str | None = None
    actions: list[Action] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Interaction:
    """One call."""

    id: str
    turns: list[Turn]
    # Policy the agent was supposed to obey, e.g. {"max_refund": 50}. Checks read this rather than
    # hardcoding thresholds, because "what is allowed" is a business decision, not a library one.
    policy: dict = field(default_factory=dict)
    completed: bool = True  # did the call reach its goal, or did it just end

    def pairs(self) -> list[tuple[Turn, Turn]]:
        """(user turn, the agent turn that answered it). Where response latency lives."""
        out = []
        for i, t in enumerate(self.turns[:-1]):
            nxt = self.turns[i + 1]
            if t.speaker == "user" and nxt.speaker == "agent":
                out.append((t, nxt))
        return out

    @property
    def duration_s(self) -> float:
        return self.turns[-1].end_s - self.turns[0].start_s if self.turns else 0.0


def load(path: str | Path) -> Interaction:
    """Load an interaction from JSON.

    The format is deliberately dumb: whatever produced your call (LiveKit, Vapi, Twilio, a test
    script) can emit this with a few lines of glue. A harness that only works with one vendor's
    SDK is a harness nobody uses.
    """
    data = json.loads(Path(path).read_text())
    return from_dict(data)


def from_dict(data: dict) -> Interaction:
    if not isinstance(data, dict):
        raise ValueError("interaction must be a JSON object with a 'turns' list")
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("interaction 'turns' must be a list")

    def turn_field(turn, i, field, *, convert=None):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {i} must be an object")
        if field not in turn:
            raise ValueError(f"turn {i} missing field '{field}'")
        value = turn[field]
        if convert is not None:
            try:
                return convert(value)
            except (TypeError, ValueError):
                raise ValueError(f"turn {i} field '{field}' must be a number, got {value!r}") from None
        if field == "speaker" and value not in ("user", "agent"):
            raise ValueError(f"turn {i} field 'speaker' must be 'user' or 'agent', got {value!r}")
        if field == "text" and not isinstance(value, str):
            raise ValueError(f"turn {i} field 'text' must be a string, got {value!r}")
        return value

    turns = [
        Turn(
            speaker=turn_field(t, i, "speaker"),
            text=turn_field(t, i, "text"),
            start_s=turn_field(t, i, "start_s", convert=float),
            end_s=turn_field(t, i, "end_s", convert=float),
            truth=t.get("truth"),
            actions=[
                Action(
                    name=a["name"],
                    args=a.get("args", {}),
                    consequential=bool(a.get("consequential", False)),
                )
                for a in t.get("actions", [])
            ],
        )
        for i, t in enumerate(raw_turns)
    ]
    return Interaction(
        id=data.get("id", "unnamed"),
        turns=turns,
        policy=data.get("policy", {}),
        completed=bool(data.get("completed", True)),
    )
