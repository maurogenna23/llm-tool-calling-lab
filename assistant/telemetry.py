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
from dataclasses import dataclass, field

from assistant.llm import Usage


@dataclass(frozen=True)
class TurnRecord:
    """What one completed turn cost.

    ``model`` is the model that *finished* the turn. When the turn escalated,
    the draft's share is in ``spend`` -- it is real money and it belongs to the
    draft, not to whoever picked up after it.
    """

    at: str  # HH:MM:SS
    model: str
    usage: Usage
    seconds: float
    rounds: int
    tools: tuple[str, ...] = ()
    #: Time to the first token the user *kept*. On a turn that changed hands
    #: the draft's text is withdrawn, so the clock restarts -- counting text
    #: that was taken off the screen would make routing look free.
    first_token_seconds: float | None = None
    #: Model labels in the order they ran. One entry unless the turn escalated.
    route: tuple[str, ...] = ()
    #: What made it escalate, if it did.
    trigger: str | None = None
    #: Usage per model label. Empty on a turn that never changed hands, where
    #: ``model`` and ``usage`` already say everything there is to say.
    spend: dict[str, Usage] = field(default_factory=dict)
    #: Whether escalation was even available on this turn. It is the
    #: denominator of the escalation rate: turns run with routing off are not
    #: evidence that routing does not fire.
    could_escalate: bool = False

    @property
    def tokens_per_second(self) -> float:
        return self.usage.completion_tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def escalated(self) -> bool:
        return len(self.route) > 1

    @property
    def route_label(self) -> str:
        """``"GPT-OSS 120B → GPT-4.1 mini"``, trimmed of the provider suffix."""
        return " → ".join(name.split(" · ")[0] for name in self.route)

    def per_model(self) -> dict[str, Usage]:
        """``spend`` when the turn recorded one, otherwise the whole turn."""
        return self.spend or {self.model: self.usage}


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
    #: Turns that changed hands mid-flight, and turns where they could have.
    escalated: int = 0
    routable: int = 0

    @property
    def escalation_rate(self) -> float:
        """Share of the turns that *could* escalate which actually did."""
        return self.escalated / self.routable if self.routable else 0.0

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
        escalated=sum(1 for record in records if record.escalated),
        routable=sum(1 for record in records if record.could_escalate),
    )


def by_model(records: Sequence[TurnRecord]) -> list[ModelSummary]:
    """One row per model, most expensive first, then busiest.

    Tokens and cost are split exactly across a turn that changed hands: the
    draft really spent what it spent, and folding that into the model that
    happened to finish would make this table lie about both.

    Seconds and tool calls cannot be split the same way -- there is only one
    wall clock per turn -- so they stay with the model that finished it, and a
    model that only ever drafted reports no throughput rather than a made-up
    one. ``turns`` counts turns the model took part in, which is why the column
    sums to more than the session total once anything has escalated.
    """
    spend: dict[str, list[Usage]] = {}
    finished: dict[str, list[TurnRecord]] = {}
    for record in records:
        for model, usage in record.per_model().items():
            spend.setdefault(model, []).append(usage)
        finished.setdefault(record.model, []).append(record)

    summaries = [
        ModelSummary(
            model=model,
            turns=len(usages),
            prompt_tokens=sum(usage.prompt_tokens for usage in usages),
            completion_tokens=sum(usage.completion_tokens for usage in usages),
            cost_usd=sum(usage.cost_usd or 0.0 for usage in usages),
            seconds=sum(row.seconds for row in finished.get(model, [])),
            tool_calls=sum(len(row.tools) for row in finished.get(model, [])),
            priced=any(usage.cost_usd is not None for usage in usages),
        )
        for model, usages in spend.items()
    ]
    return sorted(summaries, key=lambda summary: (-summary.cost_usd, -summary.turns))


HEADERS = (
    "Hora",
    "Modelo",
    "Escaló",
    "In",
    "Out",
    "Cacheados",
    "Costo",
    "Seg",
    "Tok/s",
    "Rondas",
    "Tools",
)


def table_rows(records: Sequence[TurnRecord]) -> list[list[str]]:
    """Newest first -- the turn you just ran is the one you want to read."""
    rows = []
    for record in reversed(records):
        usage = record.usage
        rows.append(
            [
                record.at,
                # The route, when there was one: what finished the turn is only
                # half the story if something else started it.
                record.route_label if record.escalated else record.model,
                record.trigger or "—",
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
        # No timed turn means this model only ever drafted. Throughput over a
        # clock that belongs to someone else is not a measurement.
        speed = f"{summary.tokens_per_second:.0f}" if summary.seconds else "—"
        lines.append(f"| {summary.model} | {summary.turns} | {tokens:,} | {cost} | {speed} |")
    return "\n".join(lines)


def routing_markdown(records: Sequence[TurnRecord]) -> str:
    """How often the turn changed hands, and what set it off.

    Deliberately not an estimated saving: what a turn *would* have cost had it
    stayed on the draft model is a counterfactual, and this module does not
    invent numbers. The only honest way to get that figure is to run the same
    conversation under both policies and compare these rows.
    """
    routable = [record for record in records if record.could_escalate]
    if not routable:
        return "_El ruteo automático estuvo apagado en todos los turnos._"

    escalated = [record for record in routable if record.escalated]
    figures = totals(records)
    lines = [
        f"**{len(escalated)}** de **{len(routable)}** turnos ruteados cambiaron de modelo "
        f"({figures.escalation_rate:.0%})."
    ]

    if escalated:
        triggers: dict[str, int] = {}
        for record in escalated:
            triggers[record.trigger or "?"] = triggers.get(record.trigger or "?", 0) + 1
        detail = " · ".join(
            f"**{count}** {trigger}" for trigger, count in sorted(triggers.items(), key=_by_count)
        )
        lines += ["", f"Por qué: {detail}"]
        lines += [
            "",
            "_Los tokens del modelo borrador están contados igual: la ronda se descartó, "
            "el gasto no._",
        ]
    else:
        lines += ["", "_Ningún turno necesitó al modelo grande. Ese también es un resultado._"]
    return "\n".join(lines)


def _by_count(item: tuple[str, int]) -> tuple[int, str]:
    trigger, count = item
    return (-count, trigger)
