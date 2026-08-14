"""Run a scripted conversation against real providers.

    python scripts/smoke.py gpt-4.1-mini
    python scripts/smoke.py gpt-4.1-mini gemini-flash groq-oss llama3.2

Costs a fraction of a cent per model. Uses a throwaway database so it never
touches the app's data.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import db, prompts  # noqa: E402
from assistant.config import get_model  # noqa: E402
from assistant.llm import default_backend  # noqa: E402
from assistant.tool_loop import (  # noqa: E402
    LoopAborted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    run_turn,
)

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def next_open_day(path: Path) -> datetime:
    day = datetime.now() + timedelta(days=1)
    while not db.is_open_at(day.replace(hour=21, minute=0), path):
        day += timedelta(days=1)
    return day.replace(hour=21, minute=0, second=0, microsecond=0)


def main(keys: list[str]) -> int:
    tmp = Path(tempfile.mkdtemp()) / "smoke.db"
    db.bootstrap(tmp)
    backend = default_backend()
    when = next_open_day(tmp)

    script = [
        "Hola! Tienen algo vegano de postre?",
        f"Perfecto. Reservame mesa para 2 el {when:%d/%m} a las 21, a nombre de Mauro.",
        # Models often ask for the zone before writing; this closes the loop so
        # the smoke run always exercises make_reservation.
        "Salón está bien. Confirmámela por favor.",
    ]

    failures = 0
    for key in keys:
        model = get_model(key)
        print(f"\n{BOLD}=== {model.label} ==={RESET}")
        if not model.available:
            print(f"{DIM}sin credenciales o sin Ollama, salteado{RESET}")
            continue

        messages = [{"role": "system", "content": prompts.system_prompt(path=tmp)}]
        tools_used: list[str] = []
        booked = False

        for user_message in script:
            print(f"\n{DIM}> {user_message}{RESET}")
            messages.append({"role": "user", "content": user_message})
            try:
                for event in run_turn(messages, model, backend, path=tmp):
                    if isinstance(event, TextDelta):
                        print(event.text, end="", flush=True)
                    elif isinstance(event, ToolStarted):
                        print(f"\n{DIM}  [tool] {event.name}({event.arguments}){RESET}")
                    elif isinstance(event, ToolFinished):
                        status = "ok" if event.result.ok else "fail"
                        first_line = event.result.text.splitlines()[0][:90]
                        print(f"{DIM}  [{status} {event.elapsed_ms}ms] {first_line}{RESET}")
                        tools_used.append(event.name)
                        booked = booked or "reservation_code" in event.result.payload
                    elif isinstance(event, TurnFinished):
                        messages = event.messages
                        usage = event.usage
                        cost = "n/d" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} c"
                        print(
                            f"\n{DIM}  {usage.prompt_tokens} in / {usage.completion_tokens} out"
                            f" / {usage.cached_tokens} cached · {cost} · {event.rounds} ronda(s){RESET}"
                        )
                    elif isinstance(event, LoopAborted):
                        print(f"\n  ABORTED: {event.reason}")
                        failures += 1
            except Exception as error:  # noqa: BLE001 - the point is to report, not to crash
                print(f"\n  ERROR: {type(error).__name__}: {error}")
                failures += 1
                break

        verdict = "OK" if booked else "no llegó a reservar"
        print(f"\n{BOLD}  -> tools: {tools_used or 'ninguna'} | {verdict}{RESET}")
        if not booked:
            failures += 1

    print(f"\n{BOLD}{'todo bien' if not failures else f'{failures} problema(s)'}{RESET}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["gpt-4.1-mini"]))
