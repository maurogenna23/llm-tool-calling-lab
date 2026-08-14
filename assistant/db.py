"""SQLite data access for the restaurant.

Every tool the model can call ends up here. Keeping the persistence layer free
of any LLM concepts means the booking rules -- overlap, capacity, opening hours
-- can be tested exhaustively without spending a single API token.

Design notes:

* Timestamps are stored as ``YYYY-MM-DDTHH:MM`` strings. That format sorts
  lexicographically, so window comparisons work directly in SQL.
* A booking occupies the table for ``BUSINESS.booking_window_minutes``. Since
  every booking has the same duration, two bookings for the same table overlap
  exactly when their start times are less than one window apart.
* We validate the *seating* time against opening hours, not the end of the
  meal. A real venue would model a separate last-seating time.
"""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from assistant.config import BUSINESS, DATA_DIR, DB_PATH

TIME_FORMAT = "%Y-%m-%dT%H:%M"
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no look-alike characters
WEEKDAY_NAMES_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


class BookingError(Exception):
    """A rule the caller broke, phrased so it can be shown to the customer."""


@dataclass(frozen=True)
class Dish:
    name: str
    category: str
    price_cents: int
    description: str
    tags: tuple[str, ...]

    @property
    def price(self) -> str:
        return f"{BUSINESS.currency_symbol}{self.price_cents / 100:,.2f}"


@dataclass(frozen=True)
class TableOption:
    table_id: int
    seats: int
    zone: str


@dataclass(frozen=True)
class Reservation:
    code: str
    customer_name: str
    party_size: int
    starts_at: datetime
    table_id: int
    zone: str
    seats: int
    status: str


# --------------------------------------------------------------------------
# connection handling
# --------------------------------------------------------------------------


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def bootstrap(path: Path | None = None) -> None:
    """Create the schema and load reference data. Safe to call on every start."""
    schema = (DATA_DIR / "schema.sql").read_text(encoding="utf-8")
    seed = (DATA_DIR / "seed.sql").read_text(encoding="utf-8")
    with connect(path) as conn:
        conn.executescript(schema)
        conn.executescript(seed)


# --------------------------------------------------------------------------
# opening hours
# --------------------------------------------------------------------------


def _minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def opening_hours(weekday: int, path: Path | None = None) -> tuple[str, str] | None:
    """Return ``(opens_at, closes_at)`` or ``None`` when closed that day."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT opens_at, closes_at FROM opening_hours WHERE weekday = ?", (weekday,)
        ).fetchone()
    if row is None or row["opens_at"] is None:
        return None
    return row["opens_at"], row["closes_at"]


def is_open_at(moment: datetime, path: Path | None = None) -> bool:
    """True when the venue is seating at ``moment``, including past-midnight service."""
    now = moment.hour * 60 + moment.minute

    today = opening_hours(moment.weekday(), path)
    if today is not None:
        opens, closes = _minutes(today[0]), _minutes(today[1])
        spans_midnight = closes <= opens
        if opens <= now and (spans_midnight or now < closes):
            return True

    # A service that closes after midnight keeps seating into the next day.
    yesterday = opening_hours((moment.weekday() - 1) % 7, path)
    if yesterday is not None:
        opens, closes = _minutes(yesterday[0]), _minutes(yesterday[1])
        if closes <= opens and now < closes:
            return True

    return False


def schedule_summary(path: Path | None = None) -> str:
    """One line per day, for the system prompt."""
    lines = []
    for weekday, label in enumerate(WEEKDAY_NAMES_ES):
        hours = opening_hours(weekday, path)
        lines.append(f"{label}: cerrado" if hours is None else f"{label}: {hours[0]} a {hours[1]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# menu
# --------------------------------------------------------------------------


def list_dishes(
    category: str | None = None,
    max_price_cents: int | None = None,
    tag: str | None = None,
    path: Path | None = None,
) -> list[Dish]:
    query = "SELECT name, category, price_cents, description, tags FROM dishes WHERE available = 1"
    params: list[object] = []
    if category:
        query += " AND category = ?"
        params.append(category.strip().lower())
    if max_price_cents is not None:
        query += " AND price_cents <= ?"
        params.append(max_price_cents)
    if tag:
        query += " AND (',' || tags || ',') LIKE ?"
        params.append(f"%,{tag.strip().lower()},%")
    query += " ORDER BY category, price_cents"

    with connect(path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        Dish(
            name=row["name"],
            category=row["category"],
            price_cents=row["price_cents"],
            description=row["description"],
            tags=tuple(t for t in row["tags"].split(",") if t),
        )
        for row in rows
    ]


def find_dish(name: str, path: Path | None = None) -> Dish | None:
    """Case-insensitive lookup, falling back to a substring match."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT name, category, price_cents, description, tags FROM dishes"
            " WHERE lower(name) = lower(?)",
            (name.strip(),),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT name, category, price_cents, description, tags FROM dishes"
                " WHERE lower(name) LIKE lower(?) ORDER BY length(name) LIMIT 1",
                (f"%{name.strip()}%",),
            ).fetchone()
    if row is None:
        return None
    return Dish(
        name=row["name"],
        category=row["category"],
        price_cents=row["price_cents"],
        description=row["description"],
        tags=tuple(t for t in row["tags"].split(",") if t),
    )


# --------------------------------------------------------------------------
# availability and reservations
# --------------------------------------------------------------------------


def parse_slot(date: str, time: str) -> datetime:
    try:
        return datetime.strptime(f"{date.strip()}T{time.strip()}", TIME_FORMAT)
    except ValueError:
        raise BookingError(
            f"No entiendo la fecha y hora '{date} {time}'. Usá el formato AAAA-MM-DD y HH:MM."
        ) from None


def _validate_slot(moment: datetime, party_size: int, path: Path | None) -> None:
    if party_size < BUSINESS.min_party_size or party_size > BUSINESS.max_party_size:
        raise BookingError(
            f"Tomamos reservas de {BUSINESS.min_party_size} a {BUSINESS.max_party_size} personas. "
            f"Para {party_size} hay que hablar con el encargado."
        )
    if not is_open_at(moment, path):
        day = WEEKDAY_NAMES_ES[moment.weekday()]
        hours = opening_hours(moment.weekday(), path)
        detail = "está cerrado" if hours is None else f"abre de {hours[0]} a {hours[1]}"
        raise BookingError(f"El {day} {detail}, así que no podemos tomar la reserva a las {moment:%H:%M}.")


def available_tables(
    moment: datetime, party_size: int, zone: str | None = None, path: Path | None = None
) -> list[TableOption]:
    """Tables that seat the party and are free for the whole booking window."""
    window = timedelta(minutes=BUSINESS.booking_window_minutes)
    lower = (moment - window).strftime(TIME_FORMAT)
    upper = (moment + window).strftime(TIME_FORMAT)

    query = """
        SELECT t.id, t.seats, t.zone
        FROM dining_tables t
        WHERE t.seats >= ?
          AND t.id NOT IN (
              SELECT r.table_id FROM reservations r
              WHERE r.status = 'CONFIRMED' AND r.starts_at > ? AND r.starts_at < ?
          )
    """
    params: list[object] = [party_size, lower, upper]
    if zone:
        query += " AND lower(t.zone) = lower(?)"
        params.append(zone.strip())
    query += " ORDER BY t.seats, t.id"  # smallest table that fits, so big ones stay free

    with connect(path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [TableOption(table_id=row["id"], seats=row["seats"], zone=row["zone"]) for row in rows]


def nearby_slots(
    moment: datetime, party_size: int, zone: str | None = None, path: Path | None = None
) -> list[datetime]:
    """Alternative seating times around a full slot, so the assistant can offer options."""
    offsets = (-60, -30, 30, 60, 90)
    found = []
    for offset in offsets:
        candidate = moment + timedelta(minutes=offset)
        if not is_open_at(candidate, path):
            continue
        if available_tables(candidate, party_size, zone, path):
            found.append(candidate)
    return found


def _new_code(conn: sqlite3.Connection) -> str:
    for _ in range(20):
        code = "R-" + "".join(secrets.choice(CODE_ALPHABET) for _ in range(4))
        if conn.execute("SELECT 1 FROM reservations WHERE code = ?", (code,)).fetchone() is None:
            return code
    raise BookingError("No pude generar un código de reserva. Probá de nuevo.")


def create_reservation(
    customer_name: str,
    party_size: int,
    moment: datetime,
    zone: str | None = None,
    path: Path | None = None,
) -> Reservation:
    _validate_slot(moment, party_size, path)

    options = available_tables(moment, party_size, zone, path)
    if not options:
        where = f" en {zone}" if zone else ""
        raise BookingError(
            f"No queda mesa para {party_size} el {moment:%d/%m} a las {moment:%H:%M}{where}."
        )

    table = options[0]
    with connect(path) as conn:
        code = _new_code(conn)
        conn.execute(
            "INSERT INTO reservations (code, customer_name, party_size, starts_at, table_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                code,
                customer_name.strip(),
                party_size,
                moment.strftime(TIME_FORMAT),
                table.table_id,
                datetime.now().strftime(TIME_FORMAT),
            ),
        )
    return Reservation(
        code=code,
        customer_name=customer_name.strip(),
        party_size=party_size,
        starts_at=moment,
        table_id=table.table_id,
        zone=table.zone,
        seats=table.seats,
        status="CONFIRMED",
    )


def get_reservation(code: str, path: Path | None = None) -> Reservation | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT r.code, r.customer_name, r.party_size, r.starts_at, r.status,"
            "       r.table_id, t.zone, t.seats"
            " FROM reservations r JOIN dining_tables t ON t.id = r.table_id"
            " WHERE upper(r.code) = upper(?)",
            (code.strip(),),
        ).fetchone()
    if row is None:
        return None
    return Reservation(
        code=row["code"],
        customer_name=row["customer_name"],
        party_size=row["party_size"],
        starts_at=datetime.strptime(row["starts_at"], TIME_FORMAT),
        table_id=row["table_id"],
        zone=row["zone"],
        seats=row["seats"],
        status=row["status"],
    )


def cancel_reservation(code: str, path: Path | None = None) -> Reservation:
    reservation = get_reservation(code, path)
    if reservation is None:
        raise BookingError(f"No encuentro ninguna reserva con el código {code.strip().upper()}.")
    if reservation.status == "CANCELLED":
        raise BookingError(f"La reserva {reservation.code} ya estaba cancelada.")
    with connect(path) as conn:
        conn.execute("UPDATE reservations SET status = 'CANCELLED' WHERE code = ?", (reservation.code,))
    return replace(reservation, status="CANCELLED")


def seed_demo_reservations(path: Path | None = None) -> int:
    """Fill part of tonight and tomorrow so availability questions have real answers.

    Called from the app, never from :func:`bootstrap`, so tests stay deterministic.
    """
    base = datetime.now().replace(minute=0, second=0, microsecond=0)
    plan = [
        ("Familia Rossi", 4, base.replace(hour=21)),
        ("Mesa Duarte", 2, base.replace(hour=21) + timedelta(minutes=30)),
        ("Grupo Ferrari", 6, base.replace(hour=22)),
        ("Cena Aguirre", 4, base.replace(hour=21) + timedelta(days=1)),
    ]
    created = 0
    for name, size, moment in plan:
        try:
            create_reservation(name, size, moment, path=path)
            created += 1
        except BookingError:
            continue  # closed that night, or already full -- nothing to do
    return created
