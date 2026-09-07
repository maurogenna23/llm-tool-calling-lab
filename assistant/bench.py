"""Run the same conversation under three routing policies and compare them.

The escalation router in :mod:`assistant.routing` is an argument, and an
argument is worth exactly as much as the measurement behind it. This module is
the measurement.

The thing it exists to avoid is a benchmark that only weighs money and speed.
Under that scoring the cheapest model wins every time, which is precisely the
wrong answer: the reason a booking is not left to a 3B model is not that it is
slow, it is that it holds two tables for one person and says something
charming about it. So every scenario declares **what the database has to look
like when the conversation is over**, and a policy that arrives at the wrong
state has lost no matter what it cost.

Each policy runs against its own throwaway database. Sharing one would let the
first policy's booking occupy the table the next policy is about to ask for,
and the comparison would measure the order they ran in.

They run one after another rather than in parallel, unlike the Arena: three
tool-heavy conversations at once is how you find out what a free tier's
tokens-per-minute cap feels like.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from assistant import db, prompts
from assistant.config import ModelSpec
from assistant.llm import ChatBackend, Message
from assistant.routing import EscalationPolicy
from assistant.telemetry import TurnRecord, totals
from assistant.tool_loop import Escalated, LoopAborted, TextDelta, ToolFinished, TurnFinished, run_turn

# --------------------------------------------------------------------------
# what a policy has to get right
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    ok: bool
    detail: str


@dataclass(frozen=True)
class Expectation:
    """The state the database must be in once the conversation is over.

    Deliberately about *rows*, not about what the assistant said. A model that
    replies "listo, ya te la cambié" and leaves both tables booked has failed,
    however good the sentence was.
    """

    confirmed: int
    cancelled: int = 0
    #: The hour the surviving booking has to be at, when the scenario moves one.
    at_hour: int | None = None

    def check(self, path: Path) -> Outcome:
        with db.connect(path) as conn:
            rows = conn.execute(
                "SELECT status, starts_at FROM reservations ORDER BY starts_at"
            ).fetchall()

        live = [row for row in rows if row["status"] == "CONFIRMED"]
        dead = [row for row in rows if row["status"] == "CANCELLED"]
        found = (
            f"{len(live)} vigente(s)"
            + (f" a las {', '.join(row['starts_at'][-5:] for row in live)}" if live else "")
            + f", {len(dead)} cancelada(s)"
        )

        if len(live) != self.confirmed:
            return Outcome(False, f"esperaba {self.confirmed} vigente(s), hubo {found}")
        if len(dead) != self.cancelled:
            return Outcome(False, f"esperaba {self.cancelled} cancelada(s), hubo {found}")
        if self.at_hour is not None and live:
            hour = int(live[0]["starts_at"][-5:-3])
            if hour != self.at_hour:
                return Outcome(False, f"la reserva quedó a las {hour}:00 y no a las {self.at_hour}:00")
        return Outcome(True, found)


@dataclass(frozen=True)
class Scenario:
    """A fixed conversation plus the state it has to end in.

    ``messages`` carry ``{date}``, ``{day}`` and ``{time}`` placeholders so the
    script always lands on a night the restaurant is actually open -- a
    scenario that books on a Monday would measure the closing-hours check
    rather than the models.
    """

    key: str
    title: str
    messages: tuple[str, ...]
    expect: Expectation
    #: Hour the conversation books for. The scenario resolves the next open day.
    hour: int = 21

    def script(self, moment: datetime) -> list[str]:
        return [
            message.format(date=f"{moment:%Y-%m-%d}", day=f"{moment:%d/%m}", time=f"{moment:%H:%M}")
            for message in self.messages
        ]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="menu",
        title="Consulta de carta (sin escritura)",
        messages=(
            "Hola! Qué postres veganos tienen?",
            "Y algo sin gluten por menos de 20?",
        ),
        # The control case. Routing should cost nothing here, and a policy that
        # books a table during a menu question has invented one.
        expect=Expectation(confirmed=0),
    ),
    Scenario(
        key="booking",
        title="Reserva simple",
        messages=(
            "Hay mesa para 2 el {day} a las {time}?",
            "Dale, reservala a nombre de Ana.",
        ),
        expect=Expectation(confirmed=1, at_hour=21),
    ),
    Scenario(
        key="modification",
        title="Cambio de reserva ya confirmada",
        messages=(
            "Reservame mesa para 2 el {day} a las {time}, a nombre de Ana.",
            "Perdón, mejor a las 22. Cambiámela.",
        ),
        # The one that pays for this whole module. Getting it wrong does not
        # look like an error: it looks like a polite confirmation and two
        # tables held for one person.
        expect=Expectation(confirmed=1, cancelled=1, at_hour=22),
    ),
)

SCENARIOS_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}


# --------------------------------------------------------------------------
# the policies under test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicySpec:
    """One arm of the comparison."""

    key: str
    label: str
    draft: ModelSpec
    #: ``None`` means the turn never changes hands.
    strong: ModelSpec | None = None
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)

    @property
    def models(self) -> str:
        short = self.draft.label.split(" · ")[0]
        return f"{short} → {self.strong.label.split(' · ')[0]}" if self.strong else short


def policies_for(draft: ModelSpec, strong: ModelSpec) -> tuple[PolicySpec, ...]:
    """The three arms: the ceiling, the floor, and the thing being argued about."""
    return (
        PolicySpec("strong", "Siempre el grande", strong),
        PolicySpec("draft", "Siempre el chico", draft),
        PolicySpec("routed", "Ruteado", draft, strong),
    )


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    policy: PolicySpec
    records: tuple[TurnRecord, ...]
    outcome: Outcome
    #: Characters the draft streamed that were then taken off the screen. The
    #: honest cost of deciding late instead of guessing early.
    discarded_chars: int = 0
    aborted: str | None = None
    #: False while the conversation is still running. A half-finished arm has
    #: real numbers but no verdict: after the first turn of the modification
    #: scenario there genuinely is one live booking, and it is genuinely the
    #: wrong answer. Reporting that as "ok" would be a lie with a timestamp.
    done: bool = True

    @property
    def cost_usd(self) -> float | None:
        figures = totals(self.records)
        return None if figures.priced_turns < figures.turns else figures.cost_usd

    @property
    def seconds(self) -> float:
        return sum(record.seconds for record in self.records)

    @property
    def tokens(self) -> int:
        figures = totals(self.records)
        return figures.prompt_tokens + figures.completion_tokens

    @property
    def rounds(self) -> int:
        return sum(record.rounds for record in self.records)

    @property
    def first_token_seconds(self) -> float | None:
        """Mean time to the first token that survived, across the turns."""
        timings = [
            record.first_token_seconds
            for record in self.records
            if record.first_token_seconds is not None
        ]
        return sum(timings) / len(timings) if timings else None

    @property
    def escalations(self) -> tuple[str, ...]:
        return tuple(record.trigger for record in self.records if record.trigger)


# --------------------------------------------------------------------------
# running one arm
# --------------------------------------------------------------------------


def next_open_slot(path: Path, hour: int, now: datetime | None = None) -> datetime:
    """The next evening the restaurant is actually seating at ``hour``."""
    moment = (now or datetime.now()) + timedelta(days=1)
    for _ in range(8):
        candidate = moment.replace(hour=hour, minute=0, second=0, microsecond=0)
        if db.is_open_at(candidate, path):
            return candidate
        moment += timedelta(days=1)
    raise RuntimeError("no hay ningún día abierto en la próxima semana")


def iter_policy(
    scenario: Scenario,
    policy: PolicySpec,
    backend: ChatBackend,
    path: Path,
    *,
    max_rounds: int = 6,
    now: datetime | None = None,
) -> Iterator[PolicyResult]:
    """Play ``scenario`` under ``policy``, yielding a partial result per turn.

    A conversation against a real provider takes tens of seconds, so a driver
    that only got an answer at the end would have nothing to show meanwhile.
    Every yield but the last carries ``done=False`` and no verdict.
    """
    moment = next_open_slot(path, scenario.hour, now)
    # Built once, as the app does: a system prompt rebuilt every turn changes
    # its prefix every minute and can never be served from the provider's cache,
    # which would tax every arm of this comparison equally but pointlessly.
    conversation: list[Message] = [
        {"role": "system", "content": prompts.system_prompt(now=now, path=path)}
    ]

    records: list[TurnRecord] = []
    discarded = 0
    aborted: str | None = None

    for message in scenario.script(moment):
        conversation.append({"role": "user", "content": message})
        started = time.perf_counter()
        first_token: float | None = None
        tools_used: list[str] = []
        trigger: str | None = None
        finished: TurnFinished | None = None

        for event in run_turn(
            conversation,
            policy.draft,
            backend,
            escalate_to=policy.strong,
            policy=policy.escalation,
            max_rounds=max_rounds,
            path=path,
        ):
            if isinstance(event, TextDelta):
                if first_token is None:
                    first_token = time.perf_counter() - started
            elif isinstance(event, Escalated):
                # The draft's text is withdrawn, so the clock restarts. Keeping
                # it would credit routing with a first token nobody got to read.
                discarded += len(event.discarded_text)
                first_token = None
                trigger = event.trigger
            elif isinstance(event, ToolFinished):
                tools_used.append(event.name)
            elif isinstance(event, TurnFinished):
                finished = event
            elif isinstance(event, LoopAborted):
                aborted = event.reason

        if finished is None:
            break  # the turn died; the conversation cannot continue honestly

        conversation = finished.messages
        route = tuple(finished.route)
        records.append(
            TurnRecord(
                at=f"{datetime.now():%H:%M:%S}",
                model=route[-1],
                usage=finished.usage,
                seconds=time.perf_counter() - started,
                rounds=finished.rounds,
                tools=tuple(tools_used),
                first_token_seconds=first_token,
                route=route,
                trigger=trigger,
                spend=dict(finished.spend),
                could_escalate=policy.strong is not None,
            )
        )
        yield PolicyResult(
            policy, tuple(records), Outcome(False, "…"), discarded, aborted, done=False
        )

    outcome = (
        Outcome(False, f"la conversación se cortó: {aborted}")
        if aborted
        else scenario.expect.check(path)
    )
    yield PolicyResult(policy, tuple(records), outcome, discarded, aborted)


def run_policy(
    scenario: Scenario,
    policy: PolicySpec,
    backend: ChatBackend,
    path: Path,
    *,
    max_rounds: int = 6,
    now: datetime | None = None,
) -> PolicyResult:
    """Play ``scenario`` end to end and hand back only the finished result."""
    # iter_policy always yields the final result last, so keeping one is enough.
    tail = deque(iter_policy(scenario, policy, backend, path, max_rounds=max_rounds, now=now), maxlen=1)
    return tail[0]


def run_scenario(
    scenario: Scenario,
    policies: Sequence[PolicySpec],
    backend: ChatBackend,
    make_path,  # noqa: ANN001 - Callable[[str], Path], one fresh database per arm
    *,
    max_rounds: int = 6,
    now: datetime | None = None,
) -> Iterator[PolicyResult]:
    """Yield one result per policy, as each finishes."""
    for policy in policies:
        path = make_path(f"{scenario.key}-{policy.key}")
        db.bootstrap(path)
        yield run_policy(scenario, policy, backend, path, max_rounds=max_rounds, now=now)


def stream_scenario(
    scenario: Scenario,
    policies: Sequence[PolicySpec],
    backend: ChatBackend,
    make_path,  # noqa: ANN001 - Callable[[str], Path], one fresh database per arm
    *,
    max_rounds: int = 6,
    now: datetime | None = None,
) -> Iterator[list[PolicyResult]]:
    """Yield the whole board after every turn, for a driver that redraws a table.

    The arms run one after another, so the board grows a row at a time and the
    row in flight fills in as its conversation goes.
    """
    finished: list[PolicyResult] = []
    for policy in policies:
        path = make_path(f"{scenario.key}-{policy.key}")
        db.bootstrap(path)
        latest: PolicyResult | None = None
        for partial in iter_policy(scenario, policy, backend, path, max_rounds=max_rounds, now=now):
            latest = partial
            yield [*finished, partial]
        if latest is not None:
            finished.append(latest)


def all_done(results: Sequence[PolicyResult], expected: int) -> bool:
    """Whether every arm has finished, which is when a verdict becomes meaningful."""
    return len(results) == expected and all(result.done for result in results)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

HEADERS = ("Política", "Modelos", "Resultado", "Costo", "1er token", "Total", "Tokens", "Escaló")


def _cost(result: PolicyResult) -> str:
    return "n/d" if result.cost_usd is None else f"{result.cost_usd * 100:.4f} ¢"


def _first_token(result: PolicyResult) -> str:
    value = result.first_token_seconds
    return "—" if value is None else f"{value:.2f} s"


def table_rows(results: Sequence[PolicyResult]) -> list[list[str]]:
    rows = []
    for result in results:
        triggers = ", ".join(result.escalations) or "—"
        rows.append(
            [
                result.policy.label,
                result.policy.models,
                # An arm still mid-conversation has real tokens and no verdict.
                "…" if not result.done else ("ok" if result.outcome.ok else "MAL"),
                _cost(result),
                _first_token(result),
                f"{result.seconds:.1f} s",
                f"{result.tokens:,}",
                triggers,
            ]
        )
    return rows


def markdown_report(scenario: Scenario, results: Sequence[PolicyResult]) -> str:
    """A table you can paste into the README, plus what it actually shows."""
    lines = [
        f"### {scenario.title}",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "|" + "---|" * len(HEADERS),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in table_rows(results)]

    failures = [result for result in results if not result.outcome.ok]
    lines.append("")
    if failures:
        for result in failures:
            lines.append(f"- **{result.policy.label}** terminó mal: {result.outcome.detail}")
    else:
        lines.append("- Las tres políticas llegaron al estado correcto.")

    discarded = sum(result.discarded_chars for result in results)
    if discarded:
        lines.append(
            f"- El ruteo mostró y retiró **{discarded}** caracteres: el costo de decidir "
            "sobre evidencia en vez de adivinar antes de empezar."
        )
    return "\n".join(lines)


def verdict(results: Sequence[PolicyResult]) -> str:
    """The sentence the table is there to support -- or to refuse to support.

    Cheapest-that-is-correct, because a wrong answer has no price worth
    comparing. When the cheap model gets there on its own, routing bought
    nothing on this scenario and the report says so.
    """
    correct = [result for result in results if result.outcome.ok and result.cost_usd is not None]
    if not correct:
        return "Ninguna política llegó al estado correcto con un precio conocido."

    best = min(correct, key=lambda result: result.cost_usd or 0.0)
    wrong = [result.policy.label for result in results if not result.outcome.ok]
    detail = f" Falló: {', '.join(wrong)}." if wrong else " Todas llegaron bien."
    return f"Más barata entre las correctas: **{best.policy.label}** ({_cost(best)}).{detail}"
