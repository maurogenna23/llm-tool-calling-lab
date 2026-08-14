"""Per-turn accounting.

Two honest choices worth stating up front:

* Token cost comes from LiteLLM's price map, so it is the provider's real
  number, not an estimate. When LiteLLM does not know a model -- newer ids
  often, and every local model -- the turn is counted but left unpriced rather
  than guessed at, and the summary says how many turns that covers.
* Images, speech and transcription are billed separately and are **not** folded
  into that figure. They are reported as call counts. Inventing a price per
  image to make a prettier total would make the number worse, not better.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from assistant.llm import Usage


@dataclass(frozen=True)
class TurnRecord:
    """What one completed turn cost."""

    at: str  # HH:MM:SS
    model: str
    usage: Usage
    seconds: float
    rounds: int
    tools: tuple[str, ...] = ()

    @property
    def tokens_per_second(self) -> float:
        return self.usage.completion_tokens / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True)
class Totals:
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    #: Turns whose provider price is known. The rest are real but unpriced.
    priced_turns: int = 0
    seconds: float = 0.0
    tool_calls: int = 0

    @property
    def cached_share(self) -> float:
        """Fraction of prompt tokens served from the provider's cache."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def average_seconds(self) -> float:
        return self.seconds / self.turns if self.turns else 0.0


@dataclass(frozen=True)
class ModelSummary:
    model: str
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    tool_calls: int = 0
    priced: bool = True

    @property
    def cost_cents(self) -> float:
        return self.cost_usd * 100

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds else 0.0


def totals(records: Sequence[TurnRecord]) -> Totals:
    return Totals(
        turns=len(records),
        prompt_tokens=sum(record.usage.prompt_tokens for record in records),
        completion_tokens=sum(record.usage.completion_tokens for record in records),
        cached_tokens=sum(record.usage.cached_tokens for record in records),
        cost_usd=sum(record.usage.cost_usd or 0.0 for record in records),
        priced_turns=sum(1 for record in records if record.usage.cost_usd is not None),
        seconds=sum(record.seconds for record in records),
        tool_calls=sum(len(record.tools) for record in records),
    )


def by_model(records: Sequence[TurnRecord]) -> list[ModelSummary]:
    """One row per model, most expensive first, then busiest."""
    buckets: dict[str, list[TurnRecord]] = {}
    for record in records:
        buckets.setdefault(record.model, []).append(record)

    summaries = [
        ModelSummary(
            model=model,
            turns=len(rows),
            prompt_tokens=sum(row.usage.prompt_tokens for row in rows),
            completion_tokens=sum(row.usage.completion_tokens for row in rows),
            cost_usd=sum(row.usage.cost_usd or 0.0 for row in rows),
            seconds=sum(row.seconds for row in rows),
            tool_calls=sum(len(row.tools) for row in rows),
            priced=any(row.usage.cost_usd is not None for row in rows),
        )
        for model, rows in buckets.items()
    ]
    return sorted(summaries, key=lambda summary: (-summary.cost_usd, -summary.turns))


HEADERS = ("Hora", "Modelo", "In", "Out", "Cacheados", "Costo", "Seg", "Tok/s", "Rondas", "Tools")


def table_rows(records: Sequence[TurnRecord]) -> list[list[str]]:
    """Newest first -- the turn you just ran is the one you want to read."""
    rows = []
    for record in reversed(records):
        usage = record.usage
        rows.append(
            [
                record.at,
                record.model,
                f"{usage.prompt_tokens:,}",
                f"{usage.completion_tokens:,}",
                f"{usage.cached_tokens:,}" if usage.cached_tokens else "—",
                "n/d" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢",
                f"{record.seconds:.1f}",
                f"{record.tokens_per_second:.0f}",
                str(record.rounds),
                ", ".join(record.tools) or "—",
            ]
        )
    return rows


def _count(quantity: int, singular: str, plural: str) -> str:
    return f"**{quantity:,}** {singular if quantity == 1 else plural}"


def summary_markdown(records: Sequence[TurnRecord]) -> str:
    if not records:
        return "_Todavía no hay turnos. Charlá un poco en la pestaña Chat y volvé._"

    figures = totals(records)
    lines = [
        f"{_count(figures.turns, 'turno', 'turnos')} · "
        f"{_count(figures.tool_calls, 'llamada a herramientas', 'llamadas a herramientas')} · "
        f"**{figures.average_seconds:.1f} s** promedio por turno",
        "",
        f"**{figures.prompt_tokens:,}** tokens de entrada · "
        f"**{figures.completion_tokens:,}** de salida",
    ]

    if figures.cached_tokens:
        lines.append(
            f"**{figures.cached_tokens:,}** cacheados por el proveedor "
            f"({figures.cached_share:.0%} de la entrada) — esos van con descuento o gratis."
        )

    cost = f"### {figures.cost_usd * 100:.4f} ¢ en total"
    if figures.priced_turns < figures.turns:
        missing = figures.turns - figures.priced_turns
        cost += f"\n\n_{missing} de {figures.turns} turnos sin precio conocido (modelo local o id nuevo)._"
    lines += ["", cost]
    return "\n".join(lines)


def media_markdown(events: Sequence[object] = ()) -> str:
    """Media calls are billed apart from tokens; they are counted, never priced."""
    kinds: dict[str, list[bool]] = {}
    for event in events:
        kinds.setdefault(getattr(event, "kind", "?"), []).append(bool(getattr(event, "cached", False)))

    if not kinds:
        return "_Sin llamadas multimodales todavía._"

    labels = {"image": "imágenes", "speech": "audios generados", "transcription": "transcripciones"}
    parts = []
    for kind, cached_flags in kinds.items():
        label = labels.get(kind, kind)
        cached = sum(cached_flags)
        detail = f" ({cached} desde caché)" if cached else ""
        parts.append(f"**{len(cached_flags)}** {label}{detail}")
    return " · ".join(parts) + "\n\n_Se facturan aparte de los tokens y no entran en el total de arriba._"


def plot_frame(records: Sequence[TurnRecord]) -> list[dict[str, object]]:
    """Rows for the per-model chart.

    Tokens rather than cost: a handful of turns costs a fraction of a cent, and
    a bar chart of 0.0978 renders an axis that reads "0". The exact money lives
    in :func:`by_model_markdown`, where precision is free.
    """
    return [
        {
            "modelo": summary.model.split(" · ")[0],
            "tokens": summary.prompt_tokens + summary.completion_tokens,
        }
        for summary in by_model(records)
    ]


def by_model_markdown(records: Sequence[TurnRecord]) -> str:
    """The precise per-model breakdown, including money."""
    summaries = by_model(records)
    if not summaries:
        return ""

    lines = ["| Modelo | Turnos | Tokens | Costo | Tok/s |", "|---|--:|--:|--:|--:|"]
    for summary in summaries:
        tokens = summary.prompt_tokens + summary.completion_tokens
        cost = f"{summary.cost_cents:.4f} ¢" if summary.priced else "n/d"
        lines.append(
            f"| {summary.model} | {summary.turns} | {tokens:,} | {cost} "
            f"| {summary.tokens_per_second:.0f} |"
        )
    return "\n".join(lines)
