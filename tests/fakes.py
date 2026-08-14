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

    def __init__(self, rounds: Sequence[Sequence[SimpleNamespace]], usage: Usage | None = None) -> None:
        self._rounds = [list(round_) for round_ in rounds]
        self._usage = usage or Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001)
        #: The messages passed on each call, so tests can assert the transcript.
        self.seen_messages: list[list[dict]] = []
        #: The tool schemas passed on each call (``None`` when tools are off).
        self.seen_tools: list[object] = []

    def stream(self, messages, model, tools):  # noqa: ANN001, ANN201 - protocol impl
        self.seen_messages.append([dict(message) for message in messages])
        self.seen_tools.append(tools)
        if not self._rounds:
            raise AssertionError("the loop asked for more rounds than the script provides")
        return iter(self._rounds.pop(0))

    def usage(self, chunks, messages, model) -> Usage:  # noqa: ANN001 - protocol impl
        return self._usage

    @property
    def rounds_left(self) -> int:
        return len(self._rounds)


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
