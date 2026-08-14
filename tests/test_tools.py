"""The tool registry: schema hygiene and executor behaviour."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from conftest import slot

from assistant import tools

TUESDAY, MONDAY = 1, 0


# --------------------------------------------------------------------------
# schema hygiene -- the cheapest bugs to prevent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(tools.REGISTRY))
def test_schema_is_well_formed(name: str) -> None:
    tool = tools.REGISTRY[name]
    schema = tool.parameters
    assert tool.name == name
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) <= set(schema["properties"]), "required lists an unknown property"
    for prop, spec in schema["properties"].items():
        assert spec.get("description"), f"{name}.{prop} has no description for the model to read"


def test_openai_payload_matches_the_registry() -> None:
    payload = tools.openai_schemas()
    assert {entry["function"]["name"] for entry in payload} == set(tools.REGISTRY)
    assert all(entry["type"] == "function" for entry in payload)


def test_write_tools_are_flagged_as_high_risk() -> None:
    writes = {name for name, tool in tools.REGISTRY.items() if tool.writes}
    assert writes == {"make_reservation", "cancel_reservation"}


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_unknown_tool_is_reported_not_raised(db_path: Path) -> None:
    result = tools.execute("delete_everything", {}, path=db_path)
    assert not result.ok and "no existe" in result.text


def test_missing_required_argument_is_reported(db_path: Path) -> None:
    result = tools.execute("check_availability", {"date": "2026-08-20"}, path=db_path)
    assert not result.ok and "time" in result.text and "party_size" in result.text


def test_unexpected_arguments_are_dropped(db_path: Path) -> None:
    """Models occasionally invent an extra field; that must not crash the turn."""
    result = tools.execute("get_menu", {"category": "postre", "sort_by": "price"}, path=db_path)
    assert result.ok and "Flan mixto" in result.text


# --------------------------------------------------------------------------
# argument coercion -- what small local models actually send
# --------------------------------------------------------------------------


def test_numbers_arriving_as_strings_are_coerced(db_path: Path) -> None:
    """Llama 3.2 sends party_size as "2"; comparing that to an int used to crash."""
    result = tools.execute(
        "check_availability",
        {"date": f"{slot(TUESDAY, 21):%Y-%m-%d}", "time": "21:00", "party_size": "2"},
        path=db_path,
    )
    assert result.ok and "mesa(s) para 2" in result.text


def test_integer_written_as_float_string(db_path: Path) -> None:
    result = tools.execute(
        "check_availability",
        {"date": f"{slot(TUESDAY, 21):%Y-%m-%d}", "time": "21:00", "party_size": "4.0"},
        path=db_path,
    )
    assert result.ok and "mesa(s) para 4" in result.text


def test_uncoercible_number_is_explained(db_path: Path) -> None:
    result = tools.execute(
        "check_availability",
        {"date": "2026-08-20", "time": "21:00", "party_size": "dos"},
        path=db_path,
    )
    assert not result.ok and "número" in result.text and "party_size" in result.text


def test_fractional_party_size_is_rejected(db_path: Path) -> None:
    result = tools.execute(
        "check_availability",
        {"date": "2026-08-20", "time": "21:00", "party_size": "2.5"},
        path=db_path,
    )
    assert not result.ok and "entero" in result.text


def test_empty_optional_argument_means_unspecified(db_path: Path) -> None:
    """Llama sends zone="" instead of omitting it; that is not a zone named ''."""
    result = tools.execute(
        "make_reservation",
        {
            "customer_name": "Mauro",
            "party_size": "2",
            "date": f"{slot(TUESDAY, 21):%Y-%m-%d}",
            "time": "21:00",
            "zone": "",
        },
        path=db_path,
    )
    assert result.ok and "Reserva confirmada" in result.text


def test_enum_matching_ignores_case_and_accents(db_path: Path) -> None:
    result = tools.execute(
        "check_availability",
        {"date": f"{slot(TUESDAY, 21):%Y-%m-%d}", "time": "21:00", "party_size": 2, "zone": "Salón"},
        path=db_path,
    )
    assert result.ok and "salon" in result.text


def test_value_outside_the_enum_is_explained(db_path: Path) -> None:
    result = tools.execute(
        "check_availability",
        {"date": "2026-08-20", "time": "21:00", "party_size": 2, "zone": "azotea"},
        path=db_path,
    )
    assert not result.ok and "salon" in result.text


# --------------------------------------------------------------------------
# menu
# --------------------------------------------------------------------------


def test_get_menu_returns_readable_lines(db_path: Path) -> None:
    result = tools.execute("get_menu", {"tag": "vegano"}, path=db_path)
    assert result.ok
    assert "Curry de garbanzos" in result.text
    assert "Bife de chorizo Aurora" not in result.text


def test_get_menu_price_filter_uses_currency_units_not_cents(db_path: Path) -> None:
    result = tools.execute("get_menu", {"category": "entrada", "max_price": 9.5}, path=db_path)
    assert "Empanadas de osobuco" in result.text  # $8.90
    assert "Rabas con alioli de limón" not in result.text  # $12.90


def test_get_menu_with_no_matches_is_not_an_error_message_the_model_misreads(db_path: Path) -> None:
    result = tools.execute("get_menu", {"category": "postre", "max_price": 0.5}, path=db_path)
    assert not result.ok and "No hay platos" in result.text


# --------------------------------------------------------------------------
# availability and booking
# --------------------------------------------------------------------------


def _args(moment, **extra) -> dict[str, object]:
    return {"date": f"{moment:%Y-%m-%d}", "time": f"{moment:%H:%M}", **extra}


def test_check_availability_reports_open_slot(db_path: Path) -> None:
    result = tools.execute("check_availability", _args(slot(TUESDAY, 21), party_size=4), path=db_path)
    assert result.ok and "mesa(s) para 4" in result.text


def test_check_availability_explains_closed_days(db_path: Path) -> None:
    result = tools.execute("check_availability", _args(slot(MONDAY, 21), party_size=2), path=db_path)
    assert not result.ok and "cerrado" in result.text


def test_check_availability_suggests_alternatives_when_full(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    for name in ("Ana", "Bruno"):
        tools.execute("make_reservation", _args(moment, customer_name=name, party_size=8), path=db_path)
    result = tools.execute("check_availability", _args(moment, party_size=8), path=db_path)
    assert not result.ok and "Horarios cercanos" in result.text


def test_booking_lifecycle(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    booked = tools.execute(
        "make_reservation",
        _args(moment, customer_name="Mauro", party_size=4, zone="terraza"),
        path=db_path,
    )
    assert booked.ok
    code = booked.payload["reservation_code"]
    assert code in booked.text

    found = tools.execute("lookup_reservation", {"code": code}, path=db_path)
    assert found.ok and "vigente" in found.text and "Mauro" in found.text

    cancelled = tools.execute("cancel_reservation", {"code": code}, path=db_path)
    assert cancelled.ok and "cancelada" in cancelled.text

    after = tools.execute("lookup_reservation", {"code": code}, path=db_path)
    assert after.ok and "cancelada" in after.text


def test_booking_failures_come_back_as_text(db_path: Path) -> None:
    """A BookingError must reach the model as content, never as an exception."""
    result = tools.execute(
        "make_reservation",
        _args(slot(MONDAY, 21), customer_name="Ana", party_size=2),
        path=db_path,
    )
    assert not result.ok and "cerrado" in result.text

    oversized = tools.execute(
        "make_reservation",
        _args(slot(TUESDAY, 21), customer_name="Ana", party_size=20),
        path=db_path,
    )
    assert not oversized.ok and "encargado" in oversized.text


def test_a_crashing_tool_is_reported_not_propagated(
    monkeypatch: pytest.MonkeyPatch, db_path: Path
) -> None:
    """A provider outage inside a tool must come back as content, not an exception."""

    def boom(**_: object) -> tools.ToolResult:
        raise RuntimeError("the image provider is down")

    monkeypatch.setitem(tools.REGISTRY, "get_menu", replace(tools.REGISTRY["get_menu"], run=boom))
    result = tools.execute("get_menu", {}, path=db_path)
    assert not result.ok and "RuntimeError" in result.text and "provider is down" in result.text


def test_lookup_of_unknown_code(db_path: Path) -> None:
    result = tools.execute("lookup_reservation", {"code": "R-ZZZZ"}, path=db_path)
    assert not result.ok and "No existe" in result.text
