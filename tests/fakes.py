"""A scripted chat backend.

Lets the whole tool loop be exercised -- fragmented arguments, parallel calls,
chained rounds, runaway loops -- with no API key, no network and no cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from assistant.llm import Usage


def fragment(index: int, id: str = "", name: str = "", arguments: str = "") -> SimpleNamespace:
    """One piece of a streamed tool call, shaped like an OpenAI delta."""
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def chunk(
    content: str | None = None,
    tool_calls: Sequence[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=list(tool_calls) if tool_calls else None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def usage_only_chunk() -> SimpleNamespace:
    """The trailing chunk providers send with ``stream_options.include_usage``."""
    return SimpleNamespace(choices=[])


def say(text: str) -> list[SimpleNamespace]:
    """A round that streams ``text`` one word at a time and stops."""
    words = text.split(" ")
    chunks = [chunk(content=word if i == 0 else f" {word}") for i, word in enumerate(words)]
    return [*chunks, chunk(finish_reason="stop"), usage_only_chunk()]


class FakeBackend:
    """Replays canned rounds and records what the loop sent."""

    def __init__(
        self,
        rounds: Sequence[Sequence[SimpleNamespace]],
        usage: Usage | None = None,
        usage_by_model: dict[str, Usage] | None = None,
    ) -> None:
        self._rounds = [list(round_) for round_ in rounds]
        self._usage = usage or Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001)
        #: Per-model override, for turns that change hands mid-flight.
        self._usage_by_model = usage_by_model or {}
        #: The messages passed on each call, so tests can assert the transcript.
        self.seen_messages: list[list[dict]] = []
        #: The tool schemas passed on each call (``None`` when tools are off).
        self.seen_tools: list[object] = []
        #: Which model each round was actually sent to.
        self.seen_models: list[str] = []

    def stream(self, messages, model, tools):  # noqa: ANN001, ANN201 - protocol impl
        self.seen_messages.append([dict(message) for message in messages])
        self.seen_tools.append(tools)
        self.seen_models.append(model.key)
        if not self._rounds:
            raise AssertionError("the loop asked for more rounds than the script provides")
        return iter(self._rounds.pop(0))

    def usage(self, chunks, messages, model) -> Usage:  # noqa: ANN001 - protocol impl
        return self._usage_by_model.get(model.key, self._usage)

    @property
    def rounds_left(self) -> int:
        return len(self._rounds)


class ArenaBackend:
    """Thread-safe fake keyed by model: several contenders stream at once."""

    def __init__(
        self,
        scripts: dict[str, Sequence[SimpleNamespace]],
        usage: dict[str, Usage] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self._scripts = {key: list(value) for key, value in scripts.items()}
        self._usage = usage or {}
        self._errors = errors or {}

    def stream(self, messages, model, tools):  # noqa: ANN001, ANN201 - protocol impl
        if model.key in self._errors:
            raise self._errors[model.key]
        return iter(self._scripts.get(model.key, []))

    def usage(self, chunks, messages, model) -> Usage:  # noqa: ANN001 - protocol impl
        return self._usage.get(model.key, Usage(prompt_tokens=50, completion_tokens=10, cost_usd=0.0001))


class RateLimitError(Exception):
    """Named after the LiteLLM exception so the error mapping recognises it."""


class ExplodingBackend:
    """A provider that is down, throttled, or pointed at a dead model id."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def stream(self, messages, model, tools):  # noqa: ANN001, ANN201 - protocol impl
        raise self._error

    def usage(self, chunks, messages, model) -> Usage:  # noqa: ANN001 - protocol impl
        return Usage()


class ScriptedByModel:
    """A backend that answers differently depending on which model is asking.

    What the bench needs: "the cheap one double-books, the good one cancels
    first". Each model key gets its own queue of rounds, so a conversation can
    be replayed under three policies from one script.
    """

    def __init__(
        self,
        scripts: dict[str, Sequence[Sequence[SimpleNamespace]]],
        usage: dict[str, Usage] | None = None,
    ) -> None:
        # A round may be a callable taking the transcript: that is how a
        # scripted model "reads" a reservation code it could not know up front.
        self._scripts = {
            key: [round_ if callable(round_) else list(round_) for round_ in value]
            for key, value in scripts.items()
        }
        self._usage = usage or {}
        self.seen_models: list[str] = []

    def stream(self, messages, model, tools):  # noqa: ANN001, ANN201 - protocol impl
        self.seen_models.append(model.key)
        rounds = self._scripts.get(model.key)
        if not rounds:
            raise AssertionError(f"no script left for {model.key}")
        round_ = rounds.pop(0)
        return iter(round_(messages) if callable(round_) else round_)

    def usage(self, chunks, messages, model) -> Usage:  # noqa: ANN001 - protocol impl
        return self._usage.get(
            model.key, Usage(prompt_tokens=1000, completion_tokens=50, cost_usd=0.0002)
        )
