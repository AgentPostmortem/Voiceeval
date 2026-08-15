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
    # BCP-47-ish tag used to select a locale pack. Default English.
    language: str = "en"

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
    turns = [
        Turn(
            speaker=t["speaker"],
            text=t["text"],
            start_s=float(t["start_s"]),
            end_s=float(t["end_s"]),
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
        for t in data["turns"]
    ]
    return Interaction(
        id=data.get("id", "unnamed"),
        turns=turns,
        policy=data.get("policy", {}),
        completed=bool(data.get("completed", True)),
        language=str(data.get("language") or "en"),
    )
