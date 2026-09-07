"""Which model runs a turn, decided on evidence instead of on a guess.

The obvious way to route between a cheap model and an expensive one is to ask
a third model first: classify the user's message as simple or complex, then
dispatch. That design has two problems this app is unusually well placed to
see.

It pays latency on *every* turn. A classification hop is serial, so it lands
in front of the first token -- the one number that decides how fast a reply
feels -- to save a fraction of a cent.

And it guesses. What makes a turn hard here is not the surface of the text:
"si, dale" is three words and a write to the database, and "cambiamela para
las 22" needs a cancel *and* a rebook, which is the case a big model has
already been observed getting wrong. To classify that correctly you need the
conversation state, which means the classifier is neither small nor cheap.

So the decision runs the other way around. Every turn starts on the draft
model, and the turn changes hands the moment the draft produces evidence that
it is out of its depth. The evidence is a tool call that has already been
streamed back -- a real name with real arguments -- not a prediction about
what the user might have meant. In the common case (a menu question) nothing
extra is spent at all.

What counts as evidence:

``write``
    The draft asked for a tool that mutates the database. Reads are cheap to
    get wrong and easy to correct; a booking is not.
``bad_arguments``
    The arguments do not parse, or cannot be coerced into the declared schema.
    A model that cannot fill in the form is not the model to trust with it.
``repeat``
    The draft asked for a call it already ran this turn. That is the shape a
    stuck model has before it becomes a runaway one.
``runaway``
    It exhausted its round budget. The backstop for when ``repeat`` misses,
    because the call varies slightly each time.

The trade to state plainly: escalation happens *before* the round runs, so the
round the draft produced is thrown away -- its tokens are spent and counted,
and its work is not used. That is why the checks all run before a single tool
executes rather than after.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from assistant import tools as tools_module
from assistant.config import ModelSpec

Trigger = Literal["write", "bad_arguments", "repeat", "runaway"]

#: Why a turn changed hands, in words a person can read. The UI shows these.
REASONS: dict[Trigger, str] = {
    "write": "pidió escribir en la base",
    "bad_arguments": "mandó argumentos que no se pueden usar",
    "repeat": "repitió una llamada que ya había hecho",
    "runaway": "se quedó sin rondas",
}


@dataclass(frozen=True)
class EscalationPolicy:
    """Which kinds of evidence are worth changing models over.

    Every trigger is individually switchable so the Telemetry tab can answer
    "which of these actually fires, and what does each one buy" rather than
    leaving it as an assertion.
    """

    on_write: bool = True
    on_bad_arguments: bool = True
    on_repeat: bool = True
    on_runaway: bool = True

    def allows(self, trigger: Trigger) -> bool:
        return bool(getattr(self, f"on_{trigger}"))

    @classmethod
    def off(cls) -> EscalationPolicy:
        """Never escalate. The turn runs entirely on the model it started on."""
        return cls(False, False, False, False)


DEFAULT_POLICY = EscalationPolicy()


@dataclass(frozen=True)
class Call:
    """A tool call the model asked for, already reassembled from the stream.

    ``arguments`` is ``None`` when the streamed JSON did not parse -- which is
    itself a signal, so it travels rather than being filtered out first.
    """

    name: str
    arguments: dict[str, object] | None


def signature(call: Call) -> str:
    """A stable identity for "the same call again", used by the repeat check."""
    return f"{call.name}({json.dumps(call.arguments or {}, sort_keys=True, default=str)})"


def can_escalate(draft: ModelSpec, strong: ModelSpec | None) -> bool:
    """Whether handing a turn from ``draft`` to ``strong`` is possible at all.

    Escalating to a model that cannot call tools would take a turn that is
    mid-tool-call and hand it to something that can only talk about it.
    """
    if strong is None or strong.key == draft.key:
        return False
    return strong.supports_tools


def trigger_for(
    calls: Sequence[Call],
    already_run: Iterable[str] = (),
    policy: EscalationPolicy = DEFAULT_POLICY,
) -> Trigger | None:
    """The first piece of evidence in this round that warrants a stronger model.

    ``already_run`` holds the signatures of calls this turn has executed, which
    is what makes the repeat check possible.
    """
    seen = set(already_run)
    for call in calls:
        # Order matters only for the label, never for the outcome: a call that
        # trips two checks escalates either way. "The draft asked to write" is
        # the more useful thing to read in the telemetry breakdown afterwards,
        # so it wins over "and its arguments were junk too".
        if policy.allows("write") and tools_module.writes(call.name):
            return "write"
        if call.arguments is None:
            if policy.allows("bad_arguments"):
                return "bad_arguments"
            continue
        if policy.allows("bad_arguments") and tools_module.prepare(call.name, call.arguments)[1]:
            return "bad_arguments"
        if policy.allows("repeat") and signature(call) in seen:
            return "repeat"
    return None


def default_draft(models: Sequence[ModelSpec]) -> ModelSpec | None:
    """The model a turn should *start* on: one that is not already the target.

    Defaulting both ends of a route to the same model leaves a control that
    claims to escalate and cannot, which is worse than not offering it.
    """
    return next(
        (model for model in models if model.supports_tools and not model.trusted_for_writes), None
    )


def default_target(models: Sequence[ModelSpec]) -> ModelSpec | None:
    """The model a turn should escalate *to*, given what is available.

    The first one the registry declares trustworthy with a write. ``None`` when
    there is no such model, in which case the UI hides the whole feature rather
    than offering a route that goes nowhere.
    """
    return next((model for model in models if model.supports_tools and model.trusted_for_writes), None)


def describe(draft: ModelSpec, strong: ModelSpec | None, trigger: Trigger) -> str:
    """One line for the chat transcript and the status bar."""
    target = strong.label if strong else "?"
    return f"{draft.label} {REASONS[trigger]} → {target}"
