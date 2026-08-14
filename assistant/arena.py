"""Run one prompt against several models at once.

No tools here on purpose: the Arena compares raw model behaviour -- speed,
verbosity, price, tone -- and tool schemas would add two thousand tokens of
identical prefix to every contender.

Each model streams on its own thread and pushes updates into a shared queue.
The generator drains that queue and yields a snapshot of every column, so the
UI sees all of them filling in at the same time rather than one after another.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from assistant.config import ModelSpec
from assistant.llm import ChatBackend, Message, Usage, describe_error

#: Nothing is allowed to hold the whole comparison hostage.
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class ArenaSlot:
    """One contender's column."""

    model: ModelSpec
    text: str = ""
    usage: Usage | None = None
    #: Time to first token: the number that decides how fast a model *feels*.
    first_token_seconds: float | None = None
    seconds: float = 0.0
    error: str | None = None
    done: bool = False

    @property
    def tokens_per_second(self) -> float:
        if self.usage is None or self.seconds <= 0:
            return 0.0
        return self.usage.completion_tokens / self.seconds


def _worker(
    index: int,
    model: ModelSpec,
    messages: Sequence[Message],
    backend: ChatBackend,
    outbox: queue.Queue,
) -> None:
    started = time.perf_counter()
    chunks: list[object] = []
    first_token: float | None = None
    try:
        for chunk in backend.stream(messages, model, None):
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if not content:
                continue
            if first_token is None:
                first_token = time.perf_counter() - started
                outbox.put((index, "first_token", first_token))
            outbox.put((index, "text", content))
        outbox.put((index, "done", (backend.usage(chunks, messages, model), time.perf_counter() - started)))
    except Exception as error:  # noqa: BLE001 - one contender failing is a result, not a crash
        outbox.put((index, "error", (describe_error(error), time.perf_counter() - started)))


def run_arena(
    prompt: str,
    models: Sequence[ModelSpec],
    backend: ChatBackend,
    *,
    system: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[list[ArenaSlot]]:
    """Yield a snapshot of every column as the models stream in parallel."""
    slots = [ArenaSlot(model=model) for model in models]
    if not slots:
        yield []
        return

    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    outbox: queue.Queue = queue.Queue()
    for index, model in enumerate(models):
        threading.Thread(
            target=_worker, args=(index, model, messages, backend, outbox), daemon=True
        ).start()

    deadline = time.perf_counter() + timeout
    remaining = len(slots)
    yield list(slots)

    while remaining:
        try:
            index, kind, payload = outbox.get(timeout=max(0.1, deadline - time.perf_counter()))
        except queue.Empty:
            slots = [
                slot if slot.done else replace(slot, done=True, error="Se acabó el tiempo de espera.")
                for slot in slots
            ]
            yield list(slots)
            return

        slot = slots[index]
        if kind == "text":
            slots[index] = replace(slot, text=slot.text + payload)
        elif kind == "first_token":
            slots[index] = replace(slot, first_token_seconds=payload)
        elif kind == "done":
            usage, elapsed = payload
            slots[index] = replace(slot, usage=usage, seconds=elapsed, done=True)
            remaining -= 1
        elif kind == "error":
            message, elapsed = payload
            slots[index] = replace(slot, error=message, seconds=elapsed, done=True)
            remaining -= 1

        yield list(slots)


HEADERS = ("Modelo", "1er token", "Total", "Tokens out", "Tok/s", "Costo")


def table_rows(slots: Sequence[ArenaSlot]) -> list[list[str]]:
    """Ranked by time to first token; failures sink to the bottom."""
    ranked = sorted(
        slots,
        key=lambda slot: (slot.error is not None, slot.first_token_seconds or float("inf")),
    )
    rows = []
    for slot in ranked:
        if slot.error:
            rows.append([slot.model.label, "—", f"{slot.seconds:.1f} s", "—", "—", "falló"])
            continue
        usage = slot.usage
        cost = "n/d" if usage is None or usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢"
        rows.append(
            [
                slot.model.label,
                "—" if slot.first_token_seconds is None else f"{slot.first_token_seconds:.2f} s",
                f"{slot.seconds:.1f} s",
                str(usage.completion_tokens) if usage else "—",
                f"{slot.tokens_per_second:.0f}",
                cost,
            ]
        )
    return rows
