"""The tools the model is allowed to call.

Each tool declares its JSON Schema and its executor in the same place, and the
schemas sent to the API are *derived* from the registry -- they are never
written out a second time by hand. A schema that drifts from its executor is
the classic reason a model silently stops calling a tool.

Executors return a :class:`ToolResult`: ``text`` goes back to the model,
``payload`` carries anything only the UI cares about (an image path, the code
of a reservation to highlight). Failures are returned as text, not raised, so
the model can apologise and offer an alternative instead of the turn dying.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from assistant import db
from assistant.config import BUSINESS
from assistant.db import BookingError

Risk = Literal["low", "high"]


@dataclass(frozen=True)
class ToolResult:
    text: str
    payload: dict[str, object] = field(default_factory=dict)
    ok: bool = True


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, object]
    risk: Risk
    run: Callable[..., ToolResult]

    @property
    def writes(self) -> bool:
        return self.risk == "high"


REGISTRY: dict[str, Tool] = {}


def _register(tool: Tool) -> Tool:
    if tool.name in REGISTRY:
        raise ValueError(f"Duplicate tool name: {tool.name}")
    REGISTRY[tool.name] = tool
    return tool


def _schema(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


DATE_PROPERTY = {"type": "string", "description": "Fecha en formato AAAA-MM-DD."}
TIME_PROPERTY = {"type": "string", "description": "Hora en formato HH:MM, 24 horas."}
ZONE_PROPERTY = {
    "type": "string",
    "enum": ["salon", "terraza", "barra"],
    "description": "Zona preferida del salón. Omitir si al cliente le da igual.",
}


# --------------------------------------------------------------------------
# menu
# --------------------------------------------------------------------------


def _get_menu(
    category: str | None = None,
    max_price: float | None = None,
    tag: str | None = None,
    path: Path | None = None,
) -> ToolResult:
    dishes = db.list_dishes(
        category=category,
        max_price_cents=round(max_price * 100) if max_price is not None else None,
        tag=tag,
        path=path,
    )
    if not dishes:
        return ToolResult("No hay platos disponibles con ese criterio.", ok=False)

    lines = []
    for dish in dishes:
        tags = f" [{', '.join(dish.tags)}]" if dish.tags else ""
        lines.append(f"- {dish.name} ({dish.category}) {dish.price}{tags}: {dish.description}")
    return ToolResult("\n".join(lines))


_register(
    Tool(
        name="get_menu",
        description=(
            "Consulta la carta vigente. Devuelve nombre, categoría, precio, etiquetas dietarias "
            "y descripción de cada plato disponible. Usar siempre antes de hablar de precios."
        ),
        parameters=_schema(
            {
                "category": {
                    "type": "string",
                    "enum": ["entrada", "principal", "postre", "bebida"],
                    "description": "Filtrar por categoría. Omitir para traer la carta completa.",
                },
                "max_price": {
                    "type": "number",
                    "description": "Precio máximo por plato, en la moneda del local (ej. 20.5).",
                },
                "tag": {
                    "type": "string",
                    "enum": ["vegano", "vegetariano", "sin-gluten", "picante"],
                    "description": "Filtrar por restricción o preferencia alimentaria.",
                },
            },
            required=[],
        ),
        risk="low",
        run=_get_menu,
    )
)


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------


def _check_availability(
    date: str,
    time: str,
    party_size: int,
    zone: str | None = None,
    path: Path | None = None,
) -> ToolResult:
    moment = db.parse_slot(date, time)
    if not db.is_open_at(moment, path):
        hours = db.opening_hours(moment.weekday(), path)
        detail = "está cerrado" if hours is None else f"abre de {hours[0]} a {hours[1]}"
        return ToolResult(f"Ese día {detail}, no se puede sentar a las {moment:%H:%M}.", ok=False)

    options = db.available_tables(moment, party_size, zone, path)
    if options:
        zones = sorted({option.zone for option in options})
        return ToolResult(
            f"Hay {len(options)} mesa(s) para {party_size} el {moment:%d/%m} a las {moment:%H:%M}. "
            f"Zonas con lugar: {', '.join(zones)}."
        )

    alternatives = db.nearby_slots(moment, party_size, zone, path)
    if alternatives:
        times = ", ".join(f"{slot:%H:%M}" for slot in alternatives)
        return ToolResult(
            f"Sin mesa a las {moment:%H:%M} para {party_size}. Horarios cercanos con lugar: {times}.",
            ok=False,
        )
    return ToolResult(
        f"Sin disponibilidad para {party_size} el {moment:%d/%m} cerca de las {moment:%H:%M}.", ok=False
    )


_register(
    Tool(
        name="check_availability",
        description=(
            "Verifica si hay mesa libre para una fecha, hora y cantidad de personas. "
            "Si no hay, sugiere horarios cercanos. Llamar SIEMPRE antes de prometer una reserva."
        ),
        parameters=_schema(
            {
                "date": DATE_PROPERTY,
                "time": TIME_PROPERTY,
                "party_size": {
                    "type": "integer",
                    "description": (
                        f"Cantidad de comensales, entre {BUSINESS.min_party_size} "
                        f"y {BUSINESS.max_party_size}."
                    ),
                },
                "zone": ZONE_PROPERTY,
            },
            required=["date", "time", "party_size"],
        ),
        risk="low",
        run=_check_availability,
    )
)


# --------------------------------------------------------------------------
# reservations
# --------------------------------------------------------------------------


def _make_reservation(
    customer_name: str,
    party_size: int,
    date: str,
    time: str,
    zone: str | None = None,
    path: Path | None = None,
) -> ToolResult:
    moment = db.parse_slot(date, time)
    reservation = db.create_reservation(customer_name, party_size, moment, zone, path)
    return ToolResult(
        f"Reserva confirmada. Código {reservation.code}, a nombre de {reservation.customer_name}, "
        f"{reservation.party_size} personas, {moment:%d/%m} a las {moment:%H:%M}, zona {reservation.zone}.",
        payload={"reservation_code": reservation.code},
    )


_register(
    Tool(
        name="make_reservation",
        description=(
            "Confirma una reserva y devuelve el código. Requiere el nombre del cliente. "
            "Nunca inventar el nombre: si no lo dijo, preguntarlo antes de llamar esta herramienta."
        ),
        parameters=_schema(
            {
                "customer_name": {"type": "string", "description": "Nombre con el que queda la reserva."},
                "party_size": {"type": "integer", "description": "Cantidad de comensales."},
                "date": DATE_PROPERTY,
                "time": TIME_PROPERTY,
                "zone": ZONE_PROPERTY,
            },
            required=["customer_name", "party_size", "date", "time"],
        ),
        risk="high",
        run=_make_reservation,
    )
)


def _lookup_reservation(code: str, path: Path | None = None) -> ToolResult:
    reservation = db.get_reservation(code, path)
    if reservation is None:
        return ToolResult(f"No existe ninguna reserva con el código {code.strip().upper()}.", ok=False)
    state = "vigente" if reservation.status == "CONFIRMED" else "cancelada"
    return ToolResult(
        f"Reserva {reservation.code} ({state}): {reservation.customer_name}, "
        f"{reservation.party_size} personas, {reservation.starts_at:%d/%m %H:%M}, zona {reservation.zone}."
    )


_register(
    Tool(
        name="lookup_reservation",
        description="Consulta el estado de una reserva existente a partir de su código (formato R-XXXX).",
        parameters=_schema(
            {"code": {"type": "string", "description": "Código de reserva, por ejemplo R-7K2M."}},
            required=["code"],
        ),
        risk="low",
        run=_lookup_reservation,
    )
)


def _cancel_reservation(code: str, path: Path | None = None) -> ToolResult:
    reservation = db.cancel_reservation(code, path)
    return ToolResult(
        f"Reserva {reservation.code} cancelada. La mesa quedó libre para "
        f"el {reservation.starts_at:%d/%m} a las {reservation.starts_at:%H:%M}.",
        payload={"reservation_code": reservation.code},
    )


_register(
    Tool(
        name="cancel_reservation",
        description="Cancela una reserva vigente por su código y libera la mesa.",
        parameters=_schema(
            {"code": {"type": "string", "description": "Código de reserva, por ejemplo R-7K2M."}},
            required=["code"],
        ),
        risk="high",
        run=_cancel_reservation,
    )
)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def openai_schemas() -> list[dict[str, object]]:
    """The ``tools=[...]`` payload, derived from the registry."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in REGISTRY.values()
    ]


def _fold(value: str) -> str:
    """Lowercase and strip accents, so 'Salón' matches the enum value 'salon'."""
    decomposed = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _coerce(name: str, value: object, spec: dict[str, object]) -> tuple[object, str]:
    """Bend a loosely typed argument into the type the schema declares.

    Small models routinely send ``"2"`` where the schema says integer. Rejecting
    that is technically correct and practically useless, so we coerce what can
    be coerced and explain what cannot.
    """
    expected = spec.get("type")
    if expected in ("integer", "number"):
        if isinstance(value, bool):
            return None, f"'{name}' tiene que ser un número y llegó un booleano."
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None, f"'{name}' tiene que ser un número y llegó '{value}'."
        if expected == "integer":
            if number != int(number):
                return None, f"'{name}' tiene que ser un número entero y llegó '{value}'."
            return int(number), ""
        return number, ""

    if expected == "string":
        text = str(value).strip()
        options = spec.get("enum")
        if isinstance(options, list):
            match = next((option for option in options if _fold(str(option)) == _fold(text)), None)
            if match is None:
                return None, f"'{name}' tiene que ser uno de: {', '.join(map(str, options))}."
            return match, ""
        return text, ""

    return value, ""


def execute(name: str, arguments: dict[str, object], path: Path | None = None) -> ToolResult:
    """Run a tool. Every failure comes back as text the model can work with."""
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult(f"La herramienta '{name}' no existe.", ok=False)

    properties: dict[str, dict[str, object]] = tool.parameters["properties"]  # type: ignore[assignment]
    kwargs: dict[str, object] = {}
    for key, value in arguments.items():
        # Unknown keys are dropped: models invent extra fields. Empty values are
        # dropped too -- they mean "not specified", not "the empty string".
        if key not in properties or value is None or value == "":
            continue
        coerced, error = _coerce(key, value, properties[key])
        if error:
            return ToolResult(error, ok=False)
        kwargs[key] = coerced

    missing = [key for key in tool.parameters["required"] if key not in kwargs]  # type: ignore[union-attr]
    if missing:
        return ToolResult(f"Faltan datos obligatorios: {', '.join(missing)}.", ok=False)

    try:
        return tool.run(path=path, **kwargs)
    except BookingError as error:
        return ToolResult(str(error), ok=False)
    except (TypeError, ValueError) as error:
        return ToolResult(f"No pude ejecutar {name}: {error}", ok=False)
