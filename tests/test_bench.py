"""The policy bench: does routing actually buy anything, and can it prove it.

The scripted models here are caricatures on purpose -- one always double-books,
one always cancels first -- because what is under test is the *bench*, not the
models. If it cannot tell those two apart it is not measuring anything.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest
from fakes import ScriptedByModel, chunk, fragment, say

from assistant import bench, db
from assistant.bench import Expectation, PolicySpec
from assistant.config import get_model
from assistant.llm import Usage

DRAFT = get_model("groq-oss")
STRONG = get_model("gpt-4.1-mini")

MODIFICATION = bench.SCENARIOS_BY_KEY["modification"]
MENU = bench.SCENARIOS_BY_KEY["menu"]


# --------------------------------------------------------------------------
# scripted models
# --------------------------------------------------------------------------


def book(moment: datetime, hour: int, call_id: str, text: str | None = None) -> list:
    body = [
        chunk(
            tool_calls=[
                fragment(
                    0,
                    id=call_id,
                    name="make_reservation",
                    arguments=json.dumps(
                        {
                            "customer_name": "Ana",
                            "party_size": 2,
                            "date": f"{moment:%Y-%m-%d}",
                            "time": f"{hour:02d}:00",
                        }
                    ),
                )
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]
    return [chunk(content=text), *body] if text else body


def cancel_whatever_is_booked(call_id: str = "c"):
    """A round that reads the reservation code out of the transcript, as a model would."""

    def build(messages) -> list:  # noqa: ANN001 - the fake passes the transcript
        codes = re.findall(r"R-[A-Z0-9]{4}", " ".join(str(m.get("content")) for m in messages))
        return [
            chunk(
                tool_calls=[
                    fragment(
                        0,
                        id=call_id,
                        name="cancel_reservation",
                        arguments=json.dumps({"code": codes[-1] if codes else "R-XXXX"}),
                    )
                ]
            ),
            chunk(finish_reason="tool_calls"),
        ]

    return build


def careless(moment: datetime) -> list:
    """Books the new time and never cancels the old one. Two tables, one party."""
    return [
        book(moment, 21, "a", "Perfecto, "),
        say("Listo Ana, quedó reservada."),
        book(moment, 22, "b"),
        say("Listo, te la cambié."),
    ]


def careful(moment: datetime) -> list:
    """Cancels first, then rebooks. The behaviour the system prompt asks for."""
    return [
        book(moment, 21, "a"),
        say("Listo Ana."),
        cancel_whatever_is_booked(),
        book(moment, 22, "b"),
        say("Listo, te la cambié."),
    ]


@pytest.fixture
def moment(db_path: Path) -> datetime:
    return bench.next_open_slot(db_path, MODIFICATION.hour)


# --------------------------------------------------------------------------
# the expectation, on its own
# --------------------------------------------------------------------------


def test_two_live_reservations_is_a_failure_not_a_detail(db_path: Path) -> None:
    """The exact bug this repo caught by hand: a polite reply over a wrong state."""
    moment = bench.next_open_slot(db_path, 21)
    db.create_reservation("Ana", 2, moment, path=db_path)
    db.create_reservation("Ana", 2, moment.replace(hour=22), path=db_path)

    outcome = Expectation(confirmed=1, cancelled=1, at_hour=22).check(db_path)
    assert not outcome.ok
    assert "2 vigente(s)" in outcome.detail


def test_the_right_state_passes(db_path: Path) -> None:
    moment = bench.next_open_slot(db_path, 21)
    first = db.create_reservation("Ana", 2, moment, path=db_path)
    db.cancel_reservation(first.code, path=db_path)
    db.create_reservation("Ana", 2, moment.replace(hour=22), path=db_path)

    assert Expectation(confirmed=1, cancelled=1, at_hour=22).check(db_path).ok


def test_the_right_count_at_the_wrong_hour_is_still_a_failure(db_path: Path) -> None:
    moment = bench.next_open_slot(db_path, 21)
    db.create_reservation("Ana", 2, moment, path=db_path)
    outcome = Expectation(confirmed=1, at_hour=22).check(db_path)
    assert not outcome.ok and "21:00" in outcome.detail


def test_a_menu_question_must_not_book_anything(db_path: Path) -> None:
    assert MENU.expect.check(db_path).ok
    db.create_reservation("Ana", 2, bench.next_open_slot(db_path, 21), path=db_path)
    assert not MENU.expect.check(db_path).ok


# --------------------------------------------------------------------------
# scenarios and policies
# --------------------------------------------------------------------------


def test_the_script_lands_on_a_night_the_place_is_open(db_path: Path) -> None:
    moment = bench.next_open_slot(db_path, 21)
    assert db.is_open_at(moment, db_path)
    assert f"{moment:%d/%m}" in MODIFICATION.script(moment)[0]
    assert "{day}" not in MODIFICATION.script(moment)[0]


def test_the_three_arms_are_the_ceiling_the_floor_and_the_question() -> None:
    strong, draft, routed = bench.policies_for(DRAFT, STRONG)

    assert (strong.draft, strong.strong) == (STRONG, None)
    assert (draft.draft, draft.strong) == (DRAFT, None)
    assert (routed.draft, routed.strong) == (DRAFT, STRONG)
    assert routed.models == "GPT-OSS 120B → GPT-4.1 mini"


# --------------------------------------------------------------------------
# the comparison it exists to make
# --------------------------------------------------------------------------


def test_the_cheap_model_double_books_and_the_bench_says_so(db_path: Path, moment: datetime) -> None:
    backend = ScriptedByModel({DRAFT.key: careless(moment)})
    result = bench.run_policy(MODIFICATION, PolicySpec("draft", "Siempre el chico", DRAFT), backend, db_path)

    assert not result.outcome.ok
    assert "2 vigente(s)" in result.outcome.detail
    assert not result.escalations


def test_the_strong_model_gets_it_right(db_path: Path, moment: datetime) -> None:
    backend = ScriptedByModel({STRONG.key: careful(moment)})
    result = bench.run_policy(
        MODIFICATION, PolicySpec("strong", "Siempre el grande", STRONG), backend, db_path
    )
    assert result.outcome.ok, result.outcome.detail


def test_routing_reaches_the_strong_models_outcome_from_the_cheap_one(
    db_path: Path, moment: datetime
) -> None:
    """The whole argument, in one assertion."""
    backend = ScriptedByModel(
        {
            # Both turns start on the draft and both open with a write, so both
            # change hands on the first round.
            DRAFT.key: [book(moment, 21, "x", "Perfecto, "), book(moment, 22, "y")],
            STRONG.key: careful(moment),
        }
    )
    result = bench.run_policy(
        MODIFICATION, PolicySpec("routed", "Ruteado", DRAFT, STRONG), backend, db_path
    )

    assert result.outcome.ok, result.outcome.detail
    assert result.escalations == ("write", "write")
    assert backend.seen_models[0] == DRAFT.key
    assert backend.seen_models[-1] == STRONG.key


def test_the_flicker_of_a_withdrawn_answer_is_counted(db_path: Path, moment: datetime) -> None:
    """Text shown then taken back is the price of deciding late. It gets a number."""
    backend = ScriptedByModel(
        {
            DRAFT.key: [book(moment, 21, "x", "Perfecto, "), book(moment, 22, "y")],
            STRONG.key: careful(moment),
        }
    )
    result = bench.run_policy(
        MODIFICATION, PolicySpec("routed", "Ruteado", DRAFT, STRONG), backend, db_path
    )
    assert result.discarded_chars == len("Perfecto, ")


def test_a_read_only_scenario_never_changes_hands(db_path: Path) -> None:
    backend = ScriptedByModel(
        {
            DRAFT.key: [
                [
                    chunk(
                        tool_calls=[
                            fragment(0, id="a", name="get_menu", arguments='{"tag": "vegano"}')
                        ]
                    ),
                    chunk(finish_reason="tool_calls"),
                ],
                say("Tenemos sorbete de maracuyá."),
                say("También el curry, sin gluten."),
            ]
        }
    )
    result = bench.run_policy(MENU, PolicySpec("routed", "Ruteado", DRAFT, STRONG), backend, db_path)

    assert result.outcome.ok
    assert not result.escalations
    assert STRONG.key not in backend.seen_models  # the expensive model was never woken up


def test_a_dead_provider_is_a_failed_arm_not_a_crash(db_path: Path, moment: datetime) -> None:
    backend = ScriptedByModel({DRAFT.key: [book(moment, 21, "a")]})  # runs out mid-conversation
    with pytest.raises(AssertionError):
        # The fake shouts when the script runs dry; a real provider would abort
        # the loop instead, which is covered by the tool loop's own tests.
        bench.run_policy(MODIFICATION, PolicySpec("draft", "chico", DRAFT), backend, db_path)


# --------------------------------------------------------------------------
# accounting and rendering
# --------------------------------------------------------------------------


def test_the_draft_round_that_was_thrown_away_is_still_paid_for(
    db_path: Path, moment: datetime
) -> None:
    backend = ScriptedByModel(
        {
            DRAFT.key: [book(moment, 21, "x"), book(moment, 22, "y")],
            STRONG.key: careful(moment),
        },
        usage={
            DRAFT.key: Usage(prompt_tokens=500, completion_tokens=10, cost_usd=0.00001),
            STRONG.key: Usage(prompt_tokens=1000, completion_tokens=50, cost_usd=0.0002),
        },
    )
    result = bench.run_policy(
        MODIFICATION, PolicySpec("routed", "Ruteado", DRAFT, STRONG), backend, db_path
    )

    per_model = {model for record in result.records for model in record.spend}
    assert per_model == {DRAFT.key, STRONG.key}
    assert result.cost_usd == pytest.approx(0.00002 + 0.001)  # 2 draft rounds + 5 strong


def _result(key: str, ok: bool, cost: float | None, label: str = "x") -> bench.PolicyResult:
    from assistant.telemetry import TurnRecord

    records = (
        TurnRecord(
            at="21:00:00",
            model="m",
            usage=Usage(prompt_tokens=100, completion_tokens=10, cost_usd=cost),
            seconds=1.0,
            rounds=1,
            first_token_seconds=0.5,
        ),
    )
    return bench.PolicyResult(
        PolicySpec(key, label, DRAFT), records, bench.Outcome(ok, "detalle")
    )


def test_the_verdict_is_cheapest_among_the_correct_ones() -> None:
    results = [
        _result("draft", ok=False, cost=0.00001, label="Siempre el chico"),
        _result("routed", ok=True, cost=0.0001, label="Ruteado"),
        _result("strong", ok=True, cost=0.001, label="Siempre el grande"),
    ]
    text = bench.verdict(results)
    assert "**Ruteado**" in text  # cheaper than the big one, and it got there
    assert "Falló: Siempre el chico" in text


def test_the_verdict_refuses_to_pick_when_nothing_worked() -> None:
    assert "Ninguna política" in bench.verdict([_result("draft", ok=False, cost=0.001)])


def test_the_verdict_says_so_when_routing_bought_nothing() -> None:
    results = [
        _result("draft", ok=True, cost=0.00001, label="Siempre el chico"),
        _result("routed", ok=True, cost=0.0001, label="Ruteado"),
    ]
    text = bench.verdict(results)
    assert "**Siempre el chico**" in text and "Todas llegaron bien" in text


def test_the_report_is_pasteable_and_names_the_loser() -> None:
    results = [
        _result("draft", ok=False, cost=0.00001, label="Siempre el chico"),
        _result("routed", ok=True, cost=0.0001, label="Ruteado"),
    ]
    text = bench.markdown_report(MODIFICATION, results)
    assert text.startswith(f"### {MODIFICATION.title}")
    assert "| Política |" in text
    assert "**Siempre el chico** terminó mal" in text
    assert "MAL" in text


def test_unpriced_arms_are_reported_as_unknown_not_as_free() -> None:
    """A local model has no price in LiteLLM's map. Zero would be a lie."""
    row = bench.table_rows([_result("draft", ok=True, cost=None)])[0]
    assert row[bench.HEADERS.index("Costo")] == "n/d"


# --------------------------------------------------------------------------
# driving it from a UI: partial results
# --------------------------------------------------------------------------


def test_an_arm_still_running_reports_no_verdict(db_path: Path, moment: datetime) -> None:
    """Halfway through the modification scenario there is one live booking, and
    it is the wrong one. Scoring that as ``ok`` would be a lie with a timestamp."""
    backend = ScriptedByModel({DRAFT.key: careless(moment)})
    partials = list(
        bench.iter_policy(MODIFICATION, PolicySpec("draft", "chico", DRAFT), backend, db_path)
    )

    assert len(partials) == len(MODIFICATION.messages) + 1  # one per turn, plus the finished one
    assert [result.done for result in partials] == [False, False, True]
    assert bench.table_rows(partials[:1])[0][bench.HEADERS.index("Resultado")] == "…"
    assert not partials[-1].outcome.ok


def test_a_partial_arm_still_carries_real_numbers(db_path: Path, moment: datetime) -> None:
    backend = ScriptedByModel({DRAFT.key: careless(moment)})
    first = next(
        iter(bench.iter_policy(MODIFICATION, PolicySpec("draft", "chico", DRAFT), backend, db_path))
    )
    assert len(first.records) == 1 and first.tokens > 0


def test_the_board_grows_a_row_at_a_time(db_path: Path, tmp_path: Path) -> None:
    moment = bench.next_open_slot(db_path, MENU.hour)
    menu = [
        [chunk(tool_calls=[fragment(0, id="a", name="get_menu", arguments="{}")]),
         chunk(finish_reason="tool_calls")],
        say("Sorbete de maracuyá."),
        say("Y el curry."),
    ]
    policies = bench.policies_for(DRAFT, STRONG)
    backend = ScriptedByModel({DRAFT.key: menu * 2, STRONG.key: menu})

    boards = list(
        bench.stream_scenario(MENU, policies, backend, lambda name: tmp_path / f"{name}.db")
    )
    assert [len(board) for board in boards] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert bench.all_done(boards[-1], len(policies))
    assert not bench.all_done(boards[0], len(policies))
    assert all(result.outcome.ok for result in boards[-1])
    assert moment  # the scenario resolved a real evening


def test_a_board_is_not_done_while_its_last_arm_is_running() -> None:
    finished = _result("strong", ok=True, cost=0.001)
    running = bench.PolicyResult(
        PolicySpec("draft", "chico", DRAFT), (), bench.Outcome(False, "…"), done=False
    )
    assert not bench.all_done([finished, running], 2)
    assert bench.all_done([finished], 1)
