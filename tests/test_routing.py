"""Deterministic escalation: a turn changes hands on evidence, not on a guess.

Everything here runs against the scripted backend, so the interesting cases --
a draft model asking for a write, sending junk arguments, or going in circles
-- are reproducible and free.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import slot
from fakes import FakeBackend, chunk, fragment, say

from assistant import db, routing, tool_loop
from assistant.config import get_model
from assistant.llm import Usage
from assistant.routing import Call, EscalationPolicy
from assistant.tool_loop import (
    ApprovalRequested,
    Escalated,
    LoopAborted,
    ToolStarted,
    TurnFinished,
)

DRAFT = get_model("groq-oss")  # fast, cheap, not trusted with a booking
STRONG = get_model("gpt-4.1-mini")
NO_TOOLS = get_model("deepseek-r1")
TUESDAY = 1

USER = [{"role": "user", "content": "hola"}]


def run(backend: FakeBackend, **kwargs) -> list:
    return list(tool_loop.run_turn(USER, DRAFT, backend, escalate_to=STRONG, **kwargs))


def only(events: list, kind: type) -> list:
    return [event for event in events if isinstance(event, kind)]


def write_round(moment, call_id: str = "a", text: str | None = None) -> list:
    body = [
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
    return [chunk(content=text), *body] if text else body


def menu_round(call_id: str = "a", category: str = "postre") -> list:
    return [
        chunk(
            tool_calls=[
                fragment(0, id=call_id, name="get_menu", arguments=f'{{"category": "{category}"}}')
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]


# --------------------------------------------------------------------------
# the policy in isolation
# --------------------------------------------------------------------------


def test_a_read_is_not_evidence_of_anything() -> None:
    assert routing.trigger_for([Call("get_menu", {"category": "postre"})]) is None


def test_a_write_is() -> None:
    call = Call(
        "make_reservation",
        {"customer_name": "Ana", "party_size": 2, "date": "2026-01-06", "time": "21:00"},
    )
    assert routing.trigger_for([call]) == "write"


def test_a_write_reports_as_a_write_even_when_its_arguments_are_junk_too() -> None:
    """Both would escalate; the label should be the more useful of the two."""
    assert routing.trigger_for([Call("make_reservation", {})]) == "write"


def test_unparseable_arguments_are() -> None:
    assert routing.trigger_for([Call("get_menu", None)]) == "bad_arguments"


def test_so_are_arguments_that_cannot_be_coerced() -> None:
    """``party_size: "dos"`` is not a number in any reading of the schema."""
    call = Call("check_availability", {"date": "2026-01-06", "time": "21:00", "party_size": "dos"})
    assert routing.trigger_for([call]) == "bad_arguments"


def test_coercible_arguments_are_not() -> None:
    """``"2"`` is what small models send for an integer. It is not a failure."""
    call = Call("check_availability", {"date": "2026-01-06", "time": "21:00", "party_size": "2"})
    assert routing.trigger_for([call]) is None


def test_a_call_already_made_this_turn_is() -> None:
    call = Call("get_menu", {"category": "postre"})
    assert routing.trigger_for([call], already_run=[routing.signature(call)]) == "repeat"


def test_the_same_tool_with_different_arguments_is_not() -> None:
    ran = routing.signature(Call("get_menu", {"category": "postre"}))
    assert routing.trigger_for([Call("get_menu", {"category": "entrada"})], already_run=[ran]) is None


def test_signatures_ignore_key_order() -> None:
    """Providers do not promise a key order, so neither can the repeat check."""
    assert routing.signature(Call("t", {"a": 1, "b": 2})) == routing.signature(Call("t", {"b": 2, "a": 1}))


@pytest.mark.parametrize(
    ("policy", "call"),
    [
        (
            EscalationPolicy(on_write=False, on_bad_arguments=False),
            Call("make_reservation", {}),
        ),
        (EscalationPolicy(on_bad_arguments=False), Call("get_menu", None)),
    ],
)
def test_each_trigger_can_be_switched_off(policy: EscalationPolicy, call: Call) -> None:
    assert routing.trigger_for([call], policy=policy) is None


def test_switching_everything_off_never_escalates() -> None:
    call = Call("make_reservation", {})
    assert routing.trigger_for([call], policy=EscalationPolicy.off()) is None


def test_a_route_that_goes_nowhere_is_refused() -> None:
    assert not routing.can_escalate(DRAFT, None)
    assert not routing.can_escalate(DRAFT, DRAFT)  # itself
    assert not routing.can_escalate(DRAFT, NO_TOOLS)  # cannot finish a tool call
    assert routing.can_escalate(DRAFT, STRONG)


def test_the_default_target_is_declared_not_guessed() -> None:
    target = routing.default_target([DRAFT, NO_TOOLS, STRONG])
    assert target is STRONG
    assert routing.default_target([DRAFT, NO_TOOLS]) is None


# --------------------------------------------------------------------------
# escalation inside a real turn
# --------------------------------------------------------------------------


def test_a_read_stays_on_the_draft_model(db_path: Path) -> None:
    backend = FakeBackend([menu_round(), say("ahí va la carta")])
    events = run(backend, path=db_path)

    assert not only(events, Escalated)
    assert backend.seen_models == [DRAFT.key, DRAFT.key]
    assert only(events, TurnFinished)[0].route == (DRAFT.key,)


def test_a_write_changes_hands_before_it_runs(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    backend = FakeBackend([write_round(moment), write_round(moment, "b"), say("Listo Ana")])
    events = run(backend, path=db_path)

    escalation = only(events, Escalated)[0]
    assert escalation.trigger == "write"
    assert escalation.from_model is DRAFT and escalation.to_model is STRONG

    # The draft asked; the strong model is the one that actually booked.
    assert backend.seen_models == [DRAFT.key, STRONG.key, STRONG.key]
    assert len(only(events, ToolStarted)) == 1
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == 1


def test_the_discarded_round_leaves_no_trace_in_the_transcript(db_path: Path) -> None:
    """A dropped round that kept its tool_calls would invalidate the next request."""
    moment = slot(TUESDAY, 21)
    backend = FakeBackend(
        [
            write_round(moment, text="Dale, te la reservo. "),
            write_round(moment, "b"),
            say("Listo Ana"),
        ]
    )
    messages = only(run(backend, path=db_path), TurnFinished)[0].messages

    requested = [
        call["id"]
        for message in messages
        if message["role"] == "assistant"
        for call in (message.get("tool_calls") or [])
    ]
    answered = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    assert requested == answered == ["b"]
    assert not any("te la reservo" in str(message.get("content")) for message in messages)


def test_junk_arguments_change_hands(db_path: Path) -> None:
    broken = [
        chunk(tool_calls=[fragment(0, id="a", name="get_menu", arguments="{not json")]),
        chunk(finish_reason="tool_calls"),
    ]
    backend = FakeBackend([broken, menu_round("b"), say("ahí va")])
    events = run(backend, path=db_path)

    assert only(events, Escalated)[0].trigger == "bad_arguments"
    assert backend.seen_models == [DRAFT.key, STRONG.key, STRONG.key]


def test_going_in_circles_changes_hands(db_path: Path) -> None:
    """The same call twice is what a stuck model looks like before it runs away."""
    backend = FakeBackend([menu_round(), menu_round("b"), menu_round("c"), say("ahí va")])
    events = run(backend, path=db_path)

    escalation = only(events, Escalated)[0]
    assert escalation.trigger == "repeat"
    # The draft got two rounds: the first call ran, and asking for the very same
    # one again is what took the turn off it.
    assert backend.seen_models == [DRAFT.key, DRAFT.key, STRONG.key, STRONG.key]


def test_a_runaway_draft_gets_the_turn_taken_off_it(db_path: Path) -> None:
    """The backstop for when the repeat check misses because the args vary."""
    rounds = [menu_round("a", "postre"), menu_round("b", "entrada"), menu_round("c"), say("listo")]
    backend = FakeBackend(rounds)
    events = run(
        backend, max_rounds=2, policy=EscalationPolicy(on_repeat=False), path=db_path
    )

    assert only(events, Escalated)[0].trigger == "runaway"
    # Two rounds on the draft, then a fresh allowance on the strong model.
    assert backend.seen_models == [DRAFT.key, DRAFT.key, STRONG.key, STRONG.key]
    assert only(events, TurnFinished)


def test_a_runaway_that_cannot_escalate_still_aborts(db_path: Path) -> None:
    backend = FakeBackend([menu_round("a", "postre"), menu_round("b", "entrada")] * 4)
    events = list(
        tool_loop.run_turn(
            USER, DRAFT, backend, max_rounds=2, policy=EscalationPolicy(on_repeat=False), path=db_path
        )
    )
    assert only(events, LoopAborted) and not only(events, TurnFinished)


def test_a_turn_changes_hands_at_most_once(db_path: Path) -> None:
    """Otherwise a stubborn strong model could ping-pong the turn forever."""
    moment = slot(TUESDAY, 21)
    backend = FakeBackend([write_round(moment), write_round(moment, "b"), say("listo")])
    events = run(backend, path=db_path)

    assert len(only(events, Escalated)) == 1
    assert only(events, TurnFinished)[0].route == (DRAFT.key, STRONG.key)


def test_escalating_is_off_unless_a_target_is_given(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    backend = FakeBackend([write_round(moment), say("listo Ana")])
    events = list(tool_loop.run_turn(USER, DRAFT, backend, path=db_path))

    assert not only(events, Escalated)
    assert backend.seen_models == [DRAFT.key, DRAFT.key]  # the draft did the booking
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == 1


def test_a_target_that_cannot_call_tools_is_ignored(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    backend = FakeBackend([write_round(moment), say("listo")])
    events = list(tool_loop.run_turn(USER, DRAFT, backend, escalate_to=NO_TOOLS, path=db_path))

    assert not only(events, Escalated)
    assert backend.seen_models == [DRAFT.key, DRAFT.key]


def test_the_policy_can_wave_writes_through(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    backend = FakeBackend([write_round(moment), say("listo")])
    events = run(backend, policy=EscalationPolicy(on_write=False), path=db_path)

    assert not only(events, Escalated)
    assert backend.seen_models == [DRAFT.key, DRAFT.key]


# --------------------------------------------------------------------------
# accounting and the approval gate
# --------------------------------------------------------------------------


def test_the_draft_pays_for_the_round_that_was_thrown_away(db_path: Path) -> None:
    """The round was discarded. The tokens were still spent, so they are counted."""
    moment = slot(TUESDAY, 21)
    backend = FakeBackend(
        [write_round(moment), write_round(moment, "b"), say("listo")],
        usage_by_model={
            DRAFT.key: Usage(prompt_tokens=800, completion_tokens=30, cost_usd=0.00002),
            STRONG.key: Usage(prompt_tokens=900, completion_tokens=40, cost_usd=0.0004),
        },
    )
    finished = only(run(backend, path=db_path), TurnFinished)[0]

    assert set(finished.spend) == {DRAFT.key, STRONG.key}
    assert finished.spend[DRAFT.key].prompt_tokens == 800  # one round
    assert finished.spend[STRONG.key].prompt_tokens == 1800  # two
    assert finished.usage.prompt_tokens == 2600
    assert finished.usage.cost_usd == pytest.approx(0.00082)


def test_the_write_still_stops_for_approval_after_changing_hands(db_path: Path) -> None:
    """Escalation picks who runs the write. It does not decide whether to."""
    moment = slot(TUESDAY, 21)
    backend = FakeBackend([write_round(moment), write_round(moment, "b"), say("bueno")])
    generator = tool_loop.run_turn(
        USER, DRAFT, backend, escalate_to=STRONG, require_approval=True, path=db_path
    )

    events, decision = [], None
    while True:
        try:
            event = generator.send(decision)
        except StopIteration:
            break
        events.append(event)
        decision = False if isinstance(event, ApprovalRequested) else None

    kinds = [type(event) for event in events]
    assert kinds.index(Escalated) < kinds.index(ApprovalRequested)
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == 0


def test_the_two_ends_of_a_default_route_are_different_models() -> None:
    """A control that claims to escalate onto itself is worse than no control."""
    models = [STRONG, DRAFT, NO_TOOLS]
    draft, target = routing.default_draft(models), routing.default_target(models)

    assert draft is DRAFT and target is STRONG
    assert routing.can_escalate(draft, target)


def test_there_is_no_default_draft_when_every_model_is_already_trusted() -> None:
    assert routing.default_draft([STRONG]) is None
    assert routing.default_draft([NO_TOOLS]) is None  # cannot call tools, cannot draft
