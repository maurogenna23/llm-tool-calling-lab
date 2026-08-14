"""Booking rules. No API keys, no network, no LLM."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from conftest import slot

from assistant import config, db
from assistant.db import BookingError

MONDAY, TUESDAY, FRIDAY, SATURDAY, SUNDAY = 0, 1, 4, 5, 6


# --------------------------------------------------------------------------
# menu
# --------------------------------------------------------------------------


def test_bootstrap_is_idempotent(db_path: Path) -> None:
    before = len(db.list_dishes(path=db_path))
    db.bootstrap(db_path)
    assert len(db.list_dishes(path=db_path)) == before


def test_unavailable_dishes_are_hidden(db_path: Path) -> None:
    names = {dish.name for dish in db.list_dishes(path=db_path)}
    assert "Milanesa napolitana" not in names  # seeded as available = 0


def test_filter_by_tag_matches_whole_tags_only(db_path: Path) -> None:
    vegan = db.list_dishes(tag="vegano", path=db_path)
    assert vegan, "expected vegan dishes in the seed data"
    assert all("vegano" in dish.tags for dish in vegan)
    # 'vegano' must not match the 'vegetariano' rows
    assert all(dish.tags != ("vegetariano",) for dish in vegan)


def test_filter_by_category_and_price(db_path: Path) -> None:
    starters = db.list_dishes(category="entrada", max_price_cents=1000, path=db_path)
    assert starters
    assert all(dish.category == "entrada" and dish.price_cents <= 1000 for dish in starters)


def test_find_dish_falls_back_to_substring(db_path: Path) -> None:
    assert db.find_dish("risotto", path=db_path).name == "Risotto de hongos"
    assert db.find_dish("no existe este plato", path=db_path) is None


def test_price_is_formatted_for_humans(db_path: Path) -> None:
    dish = db.find_dish("Flan mixto", path=db_path)
    assert dish.price == "$7.20"


# --------------------------------------------------------------------------
# opening hours
# --------------------------------------------------------------------------


def test_closed_on_mondays(db_path: Path) -> None:
    assert db.opening_hours(MONDAY, db_path) is None
    assert not db.is_open_at(slot(MONDAY, 21), db_path)


def test_open_inside_service_hours(db_path: Path) -> None:
    assert db.is_open_at(slot(TUESDAY, 13), db_path)
    assert not db.is_open_at(slot(TUESDAY, 11), db_path)
    assert not db.is_open_at(slot(TUESDAY, 23, 45), db_path)  # closes 23:30


def test_service_running_past_midnight_stays_open(db_path: Path) -> None:
    # Friday and Saturday close at 00:30, so the small hours belong to the previous service.
    assert db.is_open_at(slot(SATURDAY, 0, 15), db_path)
    assert db.is_open_at(slot(SUNDAY, 0, 15), db_path)
    assert not db.is_open_at(slot(SATURDAY, 1, 0), db_path)  # after last call
    # Thursday closes at 23:30, so Friday does not inherit a late window.
    assert not db.is_open_at(slot(FRIDAY, 0, 15), db_path)


def test_schedule_summary_lists_every_day(db_path: Path) -> None:
    summary = db.schedule_summary(db_path)
    assert summary.count("\n") == 6
    assert "lunes: cerrado" in summary


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------


def test_smallest_fitting_table_is_offered_first(db_path: Path) -> None:
    options = db.available_tables(slot(TUESDAY, 20), party_size=2, path=db_path)
    assert options[0].seats == 2
    assert all(option.seats >= 2 for option in options)


def test_large_party_only_sees_large_tables(db_path: Path) -> None:
    options = db.available_tables(slot(TUESDAY, 20), party_size=7, path=db_path)
    assert {option.seats for option in options} == {8}


def test_zone_filter(db_path: Path) -> None:
    options = db.available_tables(slot(TUESDAY, 20), party_size=4, zone="terraza", path=db_path)
    assert options and all(option.zone == "terraza" for option in options)


def test_booking_blocks_the_table_for_the_whole_window(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    reservation = db.create_reservation("Ana", 8, moment, path=db_path)

    def ids(at) -> set[int]:
        return {option.table_id for option in db.available_tables(at, 8, path=db_path)}

    assert reservation.table_id not in ids(moment)
    assert reservation.table_id not in ids(moment + timedelta(minutes=60))  # inside the 90' window
    assert reservation.table_id in ids(moment + timedelta(minutes=90))  # exactly at the edge
    assert reservation.table_id in ids(moment - timedelta(minutes=90))


def test_venue_fills_up(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    db.create_reservation("Ana", 8, moment, path=db_path)
    db.create_reservation("Bruno", 8, moment, path=db_path)
    with pytest.raises(BookingError, match="No queda mesa"):
        db.create_reservation("Clara", 8, moment, path=db_path)


def test_nearby_slots_are_suggested_when_full(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    db.create_reservation("Ana", 8, moment, path=db_path)
    db.create_reservation("Bruno", 8, moment, path=db_path)
    alternatives = db.nearby_slots(moment, 8, path=db_path)
    assert moment + timedelta(minutes=90) in alternatives
    assert all(db.is_open_at(candidate, db_path) for candidate in alternatives)


# --------------------------------------------------------------------------
# reservations
# --------------------------------------------------------------------------


def test_reservation_round_trip(db_path: Path) -> None:
    created = db.create_reservation("Mauro", 4, slot(TUESDAY, 21), zone="terraza", path=db_path)
    assert created.code.startswith("R-") and len(created.code) == 6
    assert created.zone == "terraza"

    found = db.get_reservation(created.code.lower(), path=db_path)  # lookup is case-insensitive
    assert found is not None and found.customer_name == "Mauro" and found.status == "CONFIRMED"


def test_codes_are_unique(db_path: Path) -> None:
    codes = {db.create_reservation(f"C{i}", 2, slot(TUESDAY, 20), path=db_path).code for i in range(4)}
    assert len(codes) == 4


def test_cannot_book_when_closed(db_path: Path) -> None:
    with pytest.raises(BookingError, match="cerrado"):
        db.create_reservation("Ana", 2, slot(MONDAY, 21), path=db_path)
    with pytest.raises(BookingError, match="lugar|abre"):
        db.create_reservation("Ana", 2, slot(TUESDAY, 9), path=db_path)


def test_party_size_limits(db_path: Path) -> None:
    with pytest.raises(BookingError, match="encargado"):
        db.create_reservation("Grupo", 12, slot(TUESDAY, 21), path=db_path)


def test_bad_date_format_is_explained(db_path: Path) -> None:
    with pytest.raises(BookingError, match="AAAA-MM-DD"):
        db.parse_slot("20 de agosto", "21hs")


def test_cancelling_frees_the_table(db_path: Path) -> None:
    moment = slot(TUESDAY, 21)
    first = db.create_reservation("Ana", 8, moment, path=db_path)
    db.create_reservation("Bruno", 8, moment, path=db_path)
    assert not db.available_tables(moment, 8, path=db_path)

    cancelled = db.cancel_reservation(first.code, path=db_path)
    assert cancelled.status == "CANCELLED"
    assert len(db.available_tables(moment, 8, path=db_path)) == 1


def test_cancelling_twice_is_rejected(db_path: Path) -> None:
    code = db.create_reservation("Ana", 2, slot(TUESDAY, 21), path=db_path).code
    db.cancel_reservation(code, path=db_path)
    with pytest.raises(BookingError, match="ya estaba cancelada"):
        db.cancel_reservation(code, path=db_path)


def test_cancelling_an_unknown_code_is_rejected(db_path: Path) -> None:
    with pytest.raises(BookingError, match="No encuentro"):
        db.cancel_reservation("R-ZZZZ", path=db_path)


def test_demo_seed_runs_once(db_path: Path) -> None:
    """Restarting the app must not keep booking tables until the place is full."""
    first = db.seed_demo_reservations(db_path)
    assert first > 0
    assert db.seed_demo_reservations(db_path) == 0

    with db.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM reservations").fetchone()[0] == first


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_configured_paths_resolve_against_the_project(tmp_path: Path) -> None:
    """A relative path must not follow the shell's working directory around."""
    default = tmp_path / "fallback.db"
    assert config._resolve(None, default) == default
    assert config._resolve("arnie.db", default) == config.ROOT / "arnie.db"
    assert config._resolve("/tmp/other.db", default) == Path("/tmp/other.db")
