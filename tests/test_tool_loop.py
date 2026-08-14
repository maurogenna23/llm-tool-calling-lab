"""The tool loop, driven by a scripted backend. No network, no API key, no cost."""

from __future__ import annotations

from pathlib import Path

from conftest import slot
from fakes import ExplodingBackend, FakeBackend, RateLimitError, chunk, fragment, say, usage_only_chunk

from assistant import db, tool_loop
from assistant.config import get_model
from assistant.llm import Usage, describe_error
from assistant.tool_loop import (
    ApprovalRequested,
    LoopAborted,
    TextDelta,
    ToolFinished,
    ToolRejected,
    ToolStarted,
    TurnFinished,
)

WITH_TOOLS = get_model("gpt-4.1-mini")
WITHOUT_TOOLS = get_model("deepseek-r1")
TUESDAY = 1

USER = [{"role": "user", "content": "hola"}]


def run(backend: FakeBackend, model=WITH_TOOLS, **kwargs) -> list:  # noqa: ANN001
    return list(tool_loop.run_turn(USER, model, backend, **kwargs))


def only(events: list, kind: type) -> list:
    return [event for event in events if isinstance(event, kind)]


# --------------------------------------------------------------------------
# plain conversation
# --------------------------------------------------------------------------


def test_streamed_text_is_emitted_and_assembled() -> None:
    events = run(FakeBackend([say("Buenas, bienvenido")]))

    assert "".join(event.text for event in only(events, TextDelta)) == "Buenas, bienvenido"
    finished = only(events, TurnFinished)[0]
    assert finished.text == "Buenas, bienvenido"
    assert finished.rounds == 1
    assert finished.messages[-1] == {"role": "assistant", "content": "Buenas, bienvenido"}


def test_usage_only_chunk_does_not_break_the_stream() -> None:
    """The trailing chunk from ``include_usage`` has no ``choices``."""
    events = run(FakeBackend([[chunk(content="ok"), usage_only_chunk()]]))
    assert only(events, TurnFinished)[0].text == "ok"


def test_tools_are_withheld_from_models_that_cannot_use_them() -> None:
    backend = FakeBackend([say("no puedo consultar nada")])
    run(backend, model=WITHOUT_TOOLS)
    assert backend.seen_tools == [None]


def test_tools_can_be_disabled_explicitly() -> None:
    backend = FakeBackend([say("hola")])
    run(backend, tools_enabled=False)
    assert backend.seen_tools == [None]


def test_tool_schemas_are_sent_when_supported() -> None:
    backend = FakeBackend([say("hola")])
    run(backend)
    names = {entry["function"]["name"] for entry in backend.seen_tools[0]}
    assert "check_availability" in names


# --------------------------------------------------------------------------
# reassembling streamed tool calls
# --------------------------------------------------------------------------


def test_arguments_split_across_chunks_are_reassembled(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    rounds = [
        [
            chunk(tool_calls=[fragment(0, id="call_1", name="check_availa")]),
            chunk(tool_calls=[fragment(0, name="bility", arguments='{"date": "')]),
            chunk(tool_calls=[fragment(0, arguments=f'{moment:%Y-%m-%d}", "time": "')]),
            chunk(tool_calls=[fragment(0, arguments=f'{moment:%H:%M}", "party_size": 4}}')]),
            chunk(finish_reason="tool_calls"),
        ],
        say("Sí, tenemos mesa"),
    ]
    events = run(FakeBackend(rounds), path=db_path)

    started = only(events, ToolStarted)[0]
    assert started.name == "check_availability"
    assert started.arguments == {
        "date": f"{moment:%Y-%m-%d}",
        "time": f"{moment:%H:%M}",
        "party_size": 4,
    }
    assert only(events, ToolFinished)[0].result.ok


def test_two_tool_calls_in_one_response_both_run(db_path: Path) -> None:
    rounds = [
        [
            chunk(
                tool_calls=[
                    fragment(0, id="a", name="get_menu", arguments='{"category": "postre"}'),
                    fragment(1, id="b", name="get_menu", arguments='{"tag": "vegano"}'),
                ]
            ),
            chunk(finish_reason="tool_calls"),
        ],
        say("listo"),
    ]
    events = run(FakeBackend(rounds), path=db_path)

    assert len(only(events, ToolFinished)) == 2
    tool_messages = [m for m in only(events, TurnFinished)[0].messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]


def test_tool_calls_chain_across_rounds(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    rounds = [
        [
            chunk(
                tool_calls=[
                    fragment(
                        0,
                        id="a",
                        name="check_availability",
                        arguments=f'{{"date": "{moment:%Y-%m-%d}", "time": "{moment:%H:%M}",'
                        f' "party_size": 2}}',
                    )
                ]
            ),
            chunk(finish_reason="tool_calls"),
        ],
        [
            chunk(
                tool_calls=[
                    fragment(
                        0,
                        id="b",
                        name="make_reservation",
                        arguments=f'{{"customer_name": "Mauro", "party_size": 2,'
                        f' "date": "{moment:%Y-%m-%d}", "time": "{moment:%H:%M}"}}',
                    )
                ]
            ),
            chunk(finish_reason="tool_calls"),
        ],
        say("Listo Mauro"),
    ]
    events = run(FakeBackend(rounds), path=db_path)

    assert [event.name for event in only(events, ToolStarted)] == [
        "check_availability",
        "make_reservation",
    ]
    assert only(events, TurnFinished)[0].rounds == 3
    assert db.available_tables(moment, 2, path=db_path)  # the booking really landed
    code = only(events, ToolFinished)[-1].result.payload["reservation_code"]
    assert db.get_reservation(code, path=db_path).customer_name == "Mauro"


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_unparseable_arguments_still_get_a_tool_reply(db_path: Path) -> None:
    """A missing tool reply makes the *next* API request invalid, so it must exist."""
    rounds = [
        [
            chunk(tool_calls=[fragment(0, id="a", name="get_menu", arguments="{not json")]),
            chunk(finish_reason="tool_calls"),
        ],
        say("perdón, se me trabó"),
    ]
    events = run(FakeBackend(rounds), path=db_path)

    finished = only(events, ToolFinished)[0]
    assert not finished.result.ok and "JSON" in finished.result.text
    assert not only(events, ToolStarted)  # never attempted

    messages = only(events, TurnFinished)[0].messages
    assert _tool_replies_match_tool_calls(messages)


def test_tool_failure_is_content_not_an_exception(db_path: Path) -> None:
    monday = slot(0, 21)
    rounds = [
        [
            chunk(
                tool_calls=[
                    fragment(
                        0,
                        id="a",
                        name="make_reservation",
                        arguments=f'{{"customer_name": "Ana", "party_size": 2,'
                        f' "date": "{monday:%Y-%m-%d}", "time": "21:00"}}',
                    )
                ]
            ),
            chunk(finish_reason="tool_calls"),
        ],
        say("Los lunes cerramos"),
    ]
    events = run(FakeBackend(rounds), path=db_path)

    result = only(events, ToolFinished)[0].result
    assert not result.ok and "cerrado" in result.text
    assert only(events, TurnFinished)  # the turn completed anyway


def test_runaway_tool_calling_is_bounded(db_path: Path) -> None:
    loop_round = [
        chunk(tool_calls=[fragment(0, id="a", name="get_menu", arguments="{}")]),
        chunk(finish_reason="tool_calls"),
    ]
    backend = FakeBackend([loop_round] * 10)
    events = run(backend, max_rounds=3, path=db_path)

    aborted = only(events, LoopAborted)
    assert aborted and "3 rondas" in aborted[0].reason
    assert not only(events, TurnFinished)
    assert backend.rounds_left == 7  # it really stopped at three


# --------------------------------------------------------------------------
# approval hook (the confirmation toggle in the UI)
# --------------------------------------------------------------------------


def _drive(generator, decisions: list[bool]) -> list:
    """Iterate the loop, answering each approval request in order.

    This is exactly what the UI does, minus the pause: it sends the decision
    back into the generator with ``send``.
    """
    events, decision, answers = [], None, list(decisions)
    while True:
        try:
            event = generator.send(decision)
        except StopIteration:
            return events
        events.append(event)
        decision = answers.pop(0) if (isinstance(event, ApprovalRequested) and answers) else None


def _write_round(moment, call_id: str = "a"):
    return [
        chunk(
            tool_calls=[
                fragment(
                    0,
                    id=call_id,
                    name="make_reservation",
                    arguments=f'{{"customer_name": "Ana", "party_size": 2,'
                    f' "date": "{moment:%Y-%m-%d}", "time": "{moment:%H:%M}"}}',
                )
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]


def test_reads_are_never_put_up_for_approval(db_path: Path) -> None:
    rounds = [
        [
            chunk(tool_calls=[fragment(0, id="a", name="get_menu", arguments="{}")]),
            chunk(finish_reason="tool_calls"),
        ],
        say("ahí va la carta"),
    ]
    generator = tool_loop.run_turn(
        USER, WITH_TOOLS, FakeBackend(rounds), require_approval=True, path=db_path
    )
    events = _drive(generator, [])
    assert not only(events, ApprovalRequested)
    assert only(events, ToolFinished)[0].result.ok


def test_a_declined_write_never_touches_the_database(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    rounds = [_write_round(moment), say("Entendido, no reservo")]
    generator = tool_loop.run_turn(
        USER, WITH_TOOLS, FakeBackend(rounds), require_approval=True, path=db_path
    )
    events = _drive(generator, [False])

    assert only(events, ApprovalRequested)[0].name == "make_reservation"
    assert only(events, ToolRejected)[0].name == "make_reservation"
    assert not only(events, ToolStarted)
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0


def test_an_approved_write_goes_through(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    rounds = [_write_round(moment), say("Listo Ana")]
    generator = tool_loop.run_turn(
        USER, WITH_TOOLS, FakeBackend(rounds), require_approval=True, path=db_path
    )
    events = _drive(generator, [True])

    assert not only(events, ToolRejected)
    assert only(events, ToolFinished)[0].result.ok
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == 1


def test_a_driver_that_ignores_the_request_denies_it(db_path: Path) -> None:
    """Plain iteration sends None. For something that mutates data, that has
    to mean no."""
    moment = slot(TUESDAY, 21)
    rounds = [_write_round(moment), say("bueno")]
    events = list(
        tool_loop.run_turn(
            USER, WITH_TOOLS, FakeBackend(rounds), require_approval=True, path=db_path
        )
    )
    assert only(events, ToolRejected)
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0


def test_the_model_is_told_it_was_refused(db_path: Path) -> None:
    """Otherwise it retries the same write on the next round."""
    moment = slot(TUESDAY, 21)
    rounds = [_write_round(moment), say("ok")]
    generator = tool_loop.run_turn(
        USER, WITH_TOOLS, FakeBackend(rounds), require_approval=True, path=db_path
    )
    events = _drive(generator, [False])
    messages = only(events, TurnFinished)[0].messages
    refusal = next(m for m in messages if m["role"] == "tool")
    assert "no autorizó" in refusal["content"]
    assert _tool_replies_match_tool_calls(messages)


def test_approval_is_off_by_default(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    rounds = [_write_round(moment), say("listo")]
    events = list(tool_loop.run_turn(USER, WITH_TOOLS, FakeBackend(rounds), path=db_path))
    assert not only(events, ApprovalRequested)
    assert only(events, ToolFinished)[0].result.ok


# --------------------------------------------------------------------------
# transcript shape and accounting
# --------------------------------------------------------------------------


def test_usage_accumulates_across_rounds(db_path: Path) -> None:
    rounds = [
        [
            chunk(tool_calls=[fragment(0, id="a", name="get_menu", arguments="{}")]),
            chunk(finish_reason="tool_calls"),
        ],
        say("ahí va"),
    ]
    backend = FakeBackend(rounds, usage=Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.001))
    finished = only(run(backend, path=db_path), TurnFinished)[0]

    assert finished.usage.prompt_tokens == 200
    assert finished.usage.completion_tokens == 40
    assert finished.usage.cost_usd == 0.002
    assert finished.usage.total_tokens == 240


def test_conversation_stays_valid_for_the_next_request(db_path: Path) -> None:
    rounds = [
        [
            chunk(content="Dejame ver. "),
            chunk(
                tool_calls=[
                    fragment(0, id="a", name="get_menu", arguments='{"category": "bebida"}'),
                    fragment(1, id="b", name="get_menu", arguments='{"category": "postre"}'),
                ]
            ),
            chunk(finish_reason="tool_calls"),
        ],
        say("eso tenemos"),
    ]
    backend = FakeBackend(rounds)
    messages = only(run(backend, path=db_path), TurnFinished)[0].messages

    assert _tool_replies_match_tool_calls(messages)
    assistant = next(m for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant["content"] == "Dejame ver. "  # text before a tool call is preserved
    # The second request must carry the tool results back to the model.
    assert any(m["role"] == "tool" for m in backend.seen_messages[1])


# --------------------------------------------------------------------------
# provider failures
# --------------------------------------------------------------------------


def test_provider_failure_becomes_a_readable_abort() -> None:
    backend = ExplodingBackend(RateLimitError("Limit 8000, Used 7173"))
    events = run(backend)

    aborted = only(events, LoopAborted)[0]
    assert "límite de uso" in aborted.reason
    assert "8000" in aborted.reason  # the original detail is kept for debugging
    assert not only(events, TurnFinished)


def test_error_messages_are_mapped_per_failure_kind() -> None:
    class NotFoundError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    assert "deprecado" in describe_error(NotFoundError("no such model"))
    assert "API key" in describe_error(AuthenticationError("bad key"))
    assert "Falló la llamada" in describe_error(ValueError("something odd"))


def _tool_replies_match_tool_calls(messages: list[dict]) -> bool:
    requested = [
        call["id"]
        for message in messages
        if message["role"] == "assistant"
        for call in (message.get("tool_calls") or [])
    ]
    answered = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    return requested == answered
