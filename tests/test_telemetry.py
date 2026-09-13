"""Accounting maths. The interesting cases are the ones with missing prices."""

from __future__ import annotations

from assistant.llm import Usage
from assistant.media import MediaEvent
from assistant.telemetry import (
    HEADERS,
    TurnRecord,
    by_model,
    by_model_markdown,
    media_markdown,
    plot_frame,
    routing_markdown,
    summary_markdown,
    table_rows,
    totals,
)


def turn(
    model: str = "GPT-4.1 mini · OpenAI",
    prompt: int = 1000,
    completion: int = 100,
    cached: int = 0,
    cost: float | None = 0.001,
    seconds: float = 2.0,
    rounds: int = 1,
    tools: tuple[str, ...] = (),
    at: str = "21:00:00",
) -> TurnRecord:
    return TurnRecord(
        at=at,
        model=model,
        usage=Usage(
            prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached, cost_usd=cost
        ),
        seconds=seconds,
        rounds=rounds,
        tools=tools,
    )


# --------------------------------------------------------------------------
# totals
# --------------------------------------------------------------------------


def test_empty_session() -> None:
    figures = totals([])
    assert figures.turns == 0
    assert figures.cached_share == 0.0
    assert figures.average_seconds == 0.0
    assert "Todavía no hay turnos" in summary_markdown([])


def test_counts_agree_in_number() -> None:
    one = summary_markdown([turn(tools=("get_menu",))])
    assert "**1** turno ·" in one and "**1** llamada a herramientas" in one

    many = summary_markdown([turn(tools=("a", "b")), turn(tools=("c",))])
    assert "**2** turnos ·" in many and "**3** llamadas a herramientas" in many


def test_totals_add_up() -> None:
    figures = totals([turn(tools=("get_menu",)), turn(cached=400, tools=("a", "b"))])
    assert figures.turns == 2
    assert figures.prompt_tokens == 2000
    assert figures.completion_tokens == 200
    assert figures.cached_tokens == 400
    assert figures.cost_usd == 0.002
    assert figures.tool_calls == 3
    assert figures.average_seconds == 2.0
    assert figures.cached_share == 0.2


def test_unpriced_turns_are_counted_but_not_guessed() -> None:
    """Local models have no price in LiteLLM's map. Zero is a fact, not a fill-in."""
    figures = totals([turn(cost=0.001), turn(model="Llama 3.2 3B · local", cost=None)])
    assert figures.turns == 2
    assert figures.priced_turns == 1
    assert figures.cost_usd == 0.001

    text = summary_markdown([turn(cost=0.001), turn(cost=None)])
    assert "1 de 2 turnos sin precio conocido" in text


def test_summary_mentions_caching_only_when_it_happened() -> None:
    assert "cacheados" not in summary_markdown([turn()])
    assert "50% de la entrada" in summary_markdown([turn(cached=500)])


# --------------------------------------------------------------------------
# per model
# --------------------------------------------------------------------------


def test_models_are_ranked_by_cost() -> None:
    records = [
        turn(model="cheap", cost=0.0001),
        turn(model="pricey", cost=0.01),
        turn(model="cheap", cost=0.0001),
    ]
    summaries = by_model(records)
    assert [summary.model for summary in summaries] == ["pricey", "cheap"]
    assert summaries[1].turns == 2
    assert summaries[1].cost_cents == 0.02


def test_local_model_summary_is_flagged_as_unpriced() -> None:
    summaries = by_model([turn(model="local", cost=None)])
    assert summaries[0].priced is False
    assert summaries[0].cost_usd == 0.0


def test_throughput_is_output_tokens_over_wall_clock() -> None:
    assert turn(completion=300, seconds=3.0).tokens_per_second == 100.0
    assert turn(seconds=0.0).tokens_per_second == 0.0


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_table_is_newest_first_and_matches_the_headers() -> None:
    rows = table_rows([turn(at="21:00:00"), turn(at="21:05:00")])
    assert [row[0] for row in rows] == ["21:05:00", "21:00:00"]
    assert all(len(row) == len(HEADERS) for row in rows)


def _cell(row: list[str], header: str) -> str:
    """Look a cell up by column name -- adding a column should not break a test."""
    return row[HEADERS.index(header)]


def test_table_marks_missing_values_instead_of_faking_them() -> None:
    row = table_rows([turn(cost=None, cached=0, tools=())])[0]
    assert _cell(row, "Cacheados") == "—"
    assert _cell(row, "Costo") == "n/d"
    assert _cell(row, "Tools") == "—"


def test_plot_frame_charts_tokens_not_cents() -> None:
    """A few turns cost a fraction of a cent; a bar chart of that reads "0"."""
    frame = plot_frame([turn(model="GPT-4.1 mini · OpenAI", prompt=1000, completion=100)])
    assert frame == [{"modelo": "GPT-4.1 mini", "tokens": 1100}]


def test_per_model_table_carries_the_exact_money() -> None:
    text = by_model_markdown([turn(model="GPT-4.1 mini", cost=0.000978)])
    assert "0.0978 ¢" in text
    assert "| GPT-4.1 mini | 1 | 1,100 |" in text


def test_per_model_table_marks_unpriced_models() -> None:
    assert "n/d" in by_model_markdown([turn(model="Llama 3.2 · local", cost=None)])
    assert by_model_markdown([]) == ""


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------


def test_media_counts_separate_cached_from_generated() -> None:
    text = media_markdown(
        [
            MediaEvent("image", "Risotto"),
            MediaEvent("image", "Risotto", cached=True),
            MediaEvent("speech", "120 caracteres"),
        ]
    )
    assert "**2** imágenes (1 desde caché)" in text
    assert "**1** audios generados" in text
    assert "aparte de los tokens" in text


def test_media_summary_when_nothing_happened() -> None:
    assert "Sin llamadas multimodales" in media_markdown([])


# --------------------------------------------------------------------------
# turns that changed hands
# --------------------------------------------------------------------------

DRAFT = "GPT-OSS 120B · Groq"
STRONG = "GPT-4.1 mini · OpenAI"


def escalated_turn(
    draft_cost: float = 0.0001,
    strong_cost: float = 0.0009,
    trigger: str = "write",
    seconds: float = 3.0,
) -> TurnRecord:
    return TurnRecord(
        at="21:10:00",
        model=STRONG,
        usage=Usage(prompt_tokens=1800, completion_tokens=140, cost_usd=draft_cost + strong_cost),
        seconds=seconds,
        rounds=3,
        tools=("make_reservation",),
        route=(DRAFT, STRONG),
        trigger=trigger,
        spend={
            DRAFT: Usage(prompt_tokens=800, completion_tokens=40, cost_usd=draft_cost),
            STRONG: Usage(prompt_tokens=1000, completion_tokens=100, cost_usd=strong_cost),
        },
        could_escalate=True,
    )


def test_the_draft_keeps_its_own_bill() -> None:
    """Folding the draft's tokens into whoever finished would make this lie twice."""
    rows = {summary.model: summary for summary in by_model([escalated_turn()])}

    assert rows[DRAFT].prompt_tokens == 800
    assert rows[STRONG].prompt_tokens == 1000
    assert rows[DRAFT].cost_usd == 0.0001
    assert rows[STRONG].cost_usd == 0.0009


def test_a_model_that_only_drafted_reports_no_throughput() -> None:
    """There is one clock per turn and it belongs to the model that finished."""
    text = by_model_markdown([escalated_turn()])
    draft_row = next(line for line in text.splitlines() if line.startswith(f"| {DRAFT} "))
    assert draft_row.endswith("| — |")


def test_a_plain_turn_still_bills_one_model() -> None:
    summaries = by_model([turn(model=STRONG, prompt=1000, completion=100)])
    assert len(summaries) == 1
    assert summaries[0].model == STRONG
    assert summaries[0].prompt_tokens == 1000


def test_the_escalation_rate_ignores_turns_that_could_not_escalate() -> None:
    """Turns run with routing off are not evidence that routing never fires."""
    figures = totals([escalated_turn(), turn(), turn()])
    assert figures.turns == 3
    assert figures.escalated == 1
    assert figures.routable == 1
    assert figures.escalation_rate == 1.0


def test_the_table_shows_the_route_and_what_set_it_off() -> None:
    row = table_rows([escalated_turn()])[0]
    assert _cell(row, "Modelo") == "GPT-OSS 120B → GPT-4.1 mini"
    assert _cell(row, "Escaló") == "write"


def test_the_table_marks_a_turn_that_never_changed_hands() -> None:
    row = table_rows([turn(model=STRONG)])[0]
    assert _cell(row, "Modelo") == STRONG
    assert _cell(row, "Escaló") == "—"


def test_routing_summary_says_when_it_was_never_on() -> None:
    assert "apagado" in routing_markdown([turn(), turn()])


def test_routing_summary_counts_the_triggers() -> None:
    records = [escalated_turn(trigger="write"), escalated_turn(trigger="write")]
    text = routing_markdown(records)
    assert "**2** de **2**" in text and "100%" in text
    assert "**2** write" in text


def test_no_escalation_is_reported_as_a_result_not_a_blank() -> None:
    quiet = TurnRecord(
        at="21:00:00",
        model=DRAFT,
        usage=Usage(prompt_tokens=900, completion_tokens=60, cost_usd=0.0001),
        seconds=1.0,
        rounds=1,
        route=(DRAFT,),
        could_escalate=True,
    )
    text = routing_markdown([quiet])
    assert "**0** de **1**" in text
    assert "Ningún turno necesitó" in text
