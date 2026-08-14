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
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from assistant import tools as tools_module
from assistant.config import ModelSpec
from assistant.llm import ChatBackend, Message, Usage, describe_error
from assistant.tools import ToolResult

#: Called with (tool_name, arguments); returning False blocks the call.
ApprovalHook = Callable[[str, dict[str, object]], bool]


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
class ToolRejected:
    """A write the user declined in the confirmation panel."""

    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class TurnFinished:
    messages: list[Message]
    usage: Usage
    rounds: int
    text: str


@dataclass(frozen=True)
class LoopAborted:
    reason: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


Event = TextDelta | ToolStarted | ToolFinished | ToolRejected | TurnFinished | LoopAborted


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


def run_turn(
    messages: Sequence[Message],
    model: ModelSpec,
    backend: ChatBackend,
    *,
    tools_enabled: bool = True,
    max_rounds: int = 6,
    approve: ApprovalHook | None = None,
    path: Path | None = None,
) -> Iterator[Event]:
    """Drive one user turn to completion, yielding events as they happen."""
    conversation: list[Message] = list(messages)
    schemas = tools_module.openai_schemas() if (tools_enabled and model.supports_tools) else None
    total = Usage()
    answer = ""

    for round_number in range(1, max_rounds + 1):
        chunks: list[object] = []
        pending: dict[object, _PendingCall] = {}
        text = ""

        try:
            for chunk in backend.stream(conversation, model, schemas):
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
            yield LoopAborted(describe_error(error), conversation, total)
            return

        total = total + backend.usage(chunks, conversation, model)

        if not pending:
            answer = text
            conversation.append({"role": "assistant", "content": text})
            yield TurnFinished(conversation, total, round_number, answer)
            return

        calls = [pending[key] for key in sorted(pending, key=str)]
        conversation.append(_assistant_message(text, calls))

        for call in calls:
            arguments, error = call.parse()
            if arguments is None:
                result = ToolResult(f"No pude leer los argumentos: {error}", ok=False)
                yield ToolFinished(call.name, result, 0)
            elif approve is not None and _needs_approval(call.name) and not approve(call.name, arguments):
                yield ToolRejected(call.name, arguments)
                result = ToolResult(
                    "El cliente no autorizó esta acción. No la ejecutes de nuevo sin permiso.", ok=False
                )
            else:
                yield ToolStarted(call.name, arguments)
                started = time.perf_counter()
                result = tools_module.execute(call.name, arguments, path=path)
                elapsed = int((time.perf_counter() - started) * 1000)
                yield ToolFinished(call.name, result, elapsed)

            # One reply per tool_call id, always -- including the failures.
            conversation.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result.text}
            )

    yield LoopAborted(
        f"El modelo siguió pidiendo herramientas después de {max_rounds} rondas.", conversation, total
    )


def _needs_approval(name: str) -> bool:
    tool = tools_module.REGISTRY.get(name)
    return tool is not None and tool.writes
