"""The orchestration loop: streaming and tool calling at the same time.

Most tutorials do one or the other. Together they are harder, because in a
streamed response a tool call arrives in pieces: the id, the function name and
the JSON arguments are split across chunks and have to be reassembled by index
before anything can be executed.

The loop emits typed events instead of writing to a UI, so it can be driven by
Gradio, by a CLI, or by a test with a scripted backend.

Invariants worth keeping:

* Every ``tool_call`` gets exactly one ``role: "tool"`` reply. Skipping one --
  because the arguments failed to parse, say -- makes the next request invalid.
* A tool failure is content, not an exception. The model gets the message and
  can offer an alternative.
* ``max_rounds`` bounds the loop. Small models do get stuck calling the same
  tool forever.
* A turn may change hands mid-flight, from a draft model to a stronger one --
  see :mod:`assistant.routing`. The check runs *before* the round is committed
  to the transcript, because a discarded round that left its ``tool_calls``
  behind would invalidate the next request.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from assistant import routing
from assistant import tools as tools_module
from assistant.config import ModelSpec
from assistant.llm import ChatBackend, Message, Usage, describe_error
from assistant.routing import EscalationPolicy
from assistant.tools import ToolResult

# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    """A fragment of the assistant's visible answer."""

    text: str


@dataclass(frozen=True)
class ToolStarted:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolFinished:
    name: str
    result: ToolResult
    elapsed_ms: int


@dataclass(frozen=True)
class ApprovalRequested:
    """A write is about to run and the caller asked to be consulted first.

    The driver answers with ``generator.send(True)`` or ``send(False)``. A
    driver that just iterates sends ``None``, which denies -- failing closed is
    the only safe default for something that mutates the database.
    """

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolRejected:
    """A write the user declined in the confirmation panel."""

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class Escalated:
    """The turn changed hands to a stronger model, and what made it.

    Not a *request*: nothing pauses here. The evidence is already in -- the
    draft model asked for something it should not be finishing -- and a dialog
    asking permission to use a better model would cost more attention than it
    saves. The write itself still stops at :class:`ApprovalRequested` if that
    is on, only now with the stronger model's arguments to look at.

    The round that triggered this is discarded, ``discarded_text`` included:
    the driver has already streamed that text and has to take it back.
    """

    from_model: ModelSpec
    to_model: ModelSpec
    trigger: routing.Trigger
    discarded_text: str = ""

    @property
    def reason(self) -> str:
        return routing.REASONS[self.trigger]


@dataclass(frozen=True)
class TurnFinished:
    messages: list[Message]
    usage: Usage
    rounds: int
    text: str
    #: Usage per model key. More than one entry means the turn escalated, and
    #: the draft's tokens are in here too -- they were really spent, even
    #: though the round they paid for was thrown away.
    spend: dict[str, Usage] = field(default_factory=dict)
    #: Model keys in the order they ran.
    route: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopAborted:
    reason: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    spend: dict[str, Usage] = field(default_factory=dict)
    route: tuple[str, ...] = ()


Event = (
    TextDelta
    | ToolStarted
    | ToolFinished
    | ApprovalRequested
    | ToolRejected
    | Escalated
    | TurnFinished
    | LoopAborted
)


# --------------------------------------------------------------------------
# streaming helpers
# --------------------------------------------------------------------------


@dataclass
class _PendingCall:
    """A tool call being reassembled from stream fragments."""

    id: str = ""
    name: str = ""
    arguments: str = ""

    def parse(self) -> tuple[dict[str, object] | None, str]:
        raw = self.arguments.strip() or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            return None, f"argumentos JSON inválidos ({error.msg}): {raw[:120]}"
        if not isinstance(parsed, dict):
            return None, f"los argumentos no son un objeto JSON: {raw[:120]}"
        return parsed, ""


def _assistant_message(text: str, calls: Sequence[_PendingCall]) -> Message:
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                # The raw string goes back untouched: re-serialising can change
                # it, and some providers compare it against what they sent.
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
            for call in calls
        ],
    }


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _schemas_for(model: ModelSpec, tools_enabled: bool) -> list[dict[str, object]] | None:
    """Recomputed whenever the turn changes hands: capabilities differ per model."""
    return tools_module.openai_schemas() if (tools_enabled and model.supports_tools) else None


def run_turn(
    messages: Sequence[Message],
    model: ModelSpec,
    backend: ChatBackend,
    *,
    tools_enabled: bool = True,
    max_rounds: int = 6,
    require_approval: bool = False,
    escalate_to: ModelSpec | None = None,
    policy: EscalationPolicy | None = None,
    path: Path | None = None,
) -> Iterator[Event]:
    """Drive one user turn to completion, yielding events as they happen.

    With ``require_approval`` the loop pauses before every write and yields
    :class:`ApprovalRequested`; answer it with ``generator.send(True | False)``.

    With ``escalate_to`` the turn *starts* on ``model`` and is handed to the
    stronger one the moment the draft gives evidence it should not be finishing
    this turn -- a write, unusable arguments, a repeated call, a runaway loop.
    See :mod:`assistant.routing` for why the decision is made this way round
    rather than by classifying the user's message up front. A turn escalates at
    most once, which is what bounds the whole thing.
    """
    conversation: list[Message] = list(messages)
    policy = policy or routing.DEFAULT_POLICY
    current = model
    escalation_available = routing.can_escalate(current, escalate_to)
    schemas = _schemas_for(current, tools_enabled)
    total = Usage()
    spend: dict[str, Usage] = {}
    route: list[str] = [current.key]
    #: Signatures of the calls this turn has actually run, for the repeat check.
    already_run: set[str] = set()
    answer = ""
    budget = max_rounds
    round_number = 0

    while True:
        if round_number >= budget:
            if escalation_available and policy.allows("runaway") and escalate_to is not None:
                yield Escalated(current, escalate_to, "runaway")
                current, escalation_available = escalate_to, False
                schemas = _schemas_for(current, tools_enabled)
                route.append(current.key)
                # A fresh allowance rather than a shared one: the draft spent
                # the original budget, so sharing it would escalate into zero
                # rounds. A turn escalates at most once, so the worst case for
                # the whole turn is exactly ``2 * max_rounds``.
                budget = round_number + max_rounds
                continue
            yield LoopAborted(
                f"El modelo siguió pidiendo herramientas después de {round_number} rondas.",
                conversation,
                total,
                dict(spend),
                tuple(route),
            )
            return

        round_number += 1
        chunks: list[object] = []
        pending: dict[object, _PendingCall] = {}
        text = ""

        try:
            for chunk in backend.stream(conversation, current, schemas):
                chunks.append(chunk)
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue  # the final usage-only chunk carries no choices
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                content = getattr(delta, "content", None)
                if content:
                    text += content
                    yield TextDelta(content)

                for fragment in getattr(delta, "tool_calls", None) or []:
                    # Providers that stream tool calls always send an index; the
                    # fallback covers the ones that send each call in one piece.
                    key = getattr(fragment, "index", None)
                    if key is None:
                        key = getattr(fragment, "id", None) or 0
                    call = pending.setdefault(key, _PendingCall())
                    call.id += getattr(fragment, "id", None) or ""
                    function = getattr(fragment, "function", None)
                    if function is not None:
                        call.name += getattr(function, "name", None) or ""
                        call.arguments += getattr(function, "arguments", None) or ""
        except AssertionError:
            raise  # a broken test script, not a provider problem
        except Exception as error:  # noqa: BLE001 - rate limits, outages, dead model ids
            yield LoopAborted(describe_error(error), conversation, total, dict(spend), tuple(route))
            return

        round_usage = backend.usage(chunks, conversation, current)
        total = total + round_usage
        spend[current.key] = spend.get(current.key, Usage()) + round_usage

        if not pending:
            answer = text
            conversation.append({"role": "assistant", "content": text})
            yield TurnFinished(conversation, total, round_number, answer, dict(spend), tuple(route))
            return

        calls = [pending[key] for key in sorted(pending, key=str)]
        prepared = [(call, *call.parse()) for call in calls]

        if escalation_available and escalate_to is not None:
            trigger = routing.trigger_for(
                [routing.Call(call.name, arguments) for call, arguments, _ in prepared],
                already_run,
                policy,
            )
            if trigger is not None:
                # Before a single tool runs and before the round is committed to
                # the transcript. Discarding it afterwards would leave tool_calls
                # nobody answered, which invalidates the very next request.
                yield Escalated(current, escalate_to, trigger, text)
                current, escalation_available = escalate_to, False
                schemas = _schemas_for(current, tools_enabled)
                route.append(current.key)
                continue

        conversation.append(_assistant_message(text, calls))

        for call, arguments, error in prepared:
            if arguments is None:
                result = ToolResult(f"No pude leer los argumentos: {error}", ok=False)
                yield ToolFinished(call.name, result, 0)
                conversation.append(_tool_message(call, result))
                continue

            if require_approval and tools_module.writes(call.name):
                approved = yield ApprovalRequested(call.name, arguments)
                if not approved:
                    yield ToolRejected(call.name, arguments)
                    result = ToolResult(
                        "El cliente no autorizó esta acción. No la ejecutes de nuevo sin permiso.",
                        ok=False,
                    )
                    conversation.append(_tool_message(call, result))
                    continue

            yield ToolStarted(call.name, arguments)
            started = time.perf_counter()
            result = tools_module.execute(call.name, arguments, path=path)
            elapsed = int((time.perf_counter() - started) * 1000)
            yield ToolFinished(call.name, result, elapsed)
            already_run.add(routing.signature(routing.Call(call.name, arguments)))

            # One reply per tool_call id, always -- including the failures.
            conversation.append(_tool_message(call, result))


def _tool_message(call: _PendingCall, result: ToolResult) -> Message:
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result.text}
