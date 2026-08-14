"""Parallel comparison. Threads are real; the providers are not."""

from __future__ import annotations

from fakes import ArenaBackend, RateLimitError, say

from assistant.arena import HEADERS, run_arena, table_rows
from assistant.config import get_model
from assistant.llm import Usage

GPT = get_model("gpt-4.1-mini")
GEMINI = get_model("gemini-flash-lite")
GROQ = get_model("groq-oss")


def final(prompt: str, models, backend, **kwargs) -> list:  # noqa: ANN001
    snapshots = list(run_arena(prompt, models, backend, **kwargs))
    return snapshots[-1]


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_every_contender_answers() -> None:
    backend = ArenaBackend(
        {
            "gpt-4.1-mini": say("respuesta de gpt"),
            "gemini-flash-lite": say("respuesta de gemini"),
            "groq-oss": say("respuesta de groq"),
        }
    )
    slots = final("hola", [GPT, GEMINI, GROQ], backend)

    assert [slot.model.key for slot in slots] == ["gpt-4.1-mini", "gemini-flash-lite", "groq-oss"]
    assert [slot.text for slot in slots] == [
        "respuesta de gpt",
        "respuesta de gemini",
        "respuesta de groq",
    ]
    assert all(slot.done and slot.error is None for slot in slots)


def test_columns_are_ordered_by_the_caller_not_by_who_finished() -> None:
    """Threads race; the layout must not."""
    backend = ArenaBackend({"groq-oss": say("rapido"), "gpt-4.1-mini": say("lento")})
    slots = final("hola", [GPT, GROQ], backend)
    assert [slot.model.key for slot in slots] == ["gpt-4.1-mini", "groq-oss"]


def test_snapshots_stream_rather_than_arriving_all_at_once() -> None:
    backend = ArenaBackend({"gpt-4.1-mini": say("una respuesta bastante larga")})
    snapshots = list(run_arena("hola", [GPT], backend))
    assert len(snapshots) > 3
    lengths = [len(snapshot[0].text) for snapshot in snapshots]
    assert lengths == sorted(lengths), "text must only ever grow"


def test_usage_and_timing_are_recorded() -> None:
    backend = ArenaBackend(
        {"gpt-4.1-mini": say("hola mundo")},
        usage={"gpt-4.1-mini": Usage(prompt_tokens=40, completion_tokens=200, cost_usd=0.002)},
    )
    slot = final("hola", [GPT], backend)[0]
    assert slot.usage.completion_tokens == 200
    assert slot.first_token_seconds is not None
    assert slot.seconds > 0
    assert slot.tokens_per_second > 0


def test_empty_lineup() -> None:
    assert final("hola", [], ArenaBackend({})) == []


# --------------------------------------------------------------------------
# failures are results, not crashes
# --------------------------------------------------------------------------


def test_one_provider_failing_does_not_take_the_others_down() -> None:
    backend = ArenaBackend(
        {"gpt-4.1-mini": say("todo bien"), "gemini-flash-lite": say("yo tambien")},
        errors={"groq-oss": RateLimitError("Limit 8000 tokens per minute")},
    )
    slots = final("hola", [GPT, GROQ, GEMINI], backend)

    assert slots[0].text == "todo bien" and slots[0].error is None
    assert slots[2].text == "yo tambien" and slots[2].error is None
    assert "límite de uso" in slots[1].error
    assert all(slot.done for slot in slots)


def test_a_hung_provider_is_abandoned() -> None:
    class NeverEnding:
        def stream(self, messages, model, tools):  # noqa: ANN001, ANN201
            import time

            while True:
                time.sleep(0.05)

        def usage(self, chunks, messages, model) -> Usage:  # noqa: ANN001
            return Usage()

    slots = final("hola", [GPT], NeverEnding(), timeout=0.3)
    assert slots[0].done and "tiempo de espera" in slots[0].error


# --------------------------------------------------------------------------
# the results table
# --------------------------------------------------------------------------


def test_table_ranks_by_time_to_first_token() -> None:
    backend = ArenaBackend({"gpt-4.1-mini": say("a"), "groq-oss": say("b")})
    slots = final("hola", [GPT, GROQ], backend)
    fast, slow = slots[1], slots[0]
    object.__setattr__(fast, "first_token_seconds", 0.1)
    object.__setattr__(slow, "first_token_seconds", 0.9)

    rows = table_rows(slots)
    assert rows[0][0] == fast.model.label
    assert all(len(row) == len(HEADERS) for row in rows)


def test_failed_contenders_sink_to_the_bottom() -> None:
    backend = ArenaBackend(
        {"gpt-4.1-mini": say("bien")}, errors={"groq-oss": RateLimitError("nope")}
    )
    rows = table_rows(final("hola", [GROQ, GPT], backend))
    assert rows[0][0] == GPT.label
    assert rows[1][5] == "falló"
