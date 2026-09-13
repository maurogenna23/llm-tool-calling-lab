"""Run the same conversation under three routing policies, against real providers.

    python scripts/bench.py
    python scripts/bench.py --draft groq-oss --strong gpt-4.1-mini
    python scripts/bench.py --scenario modification --markdown

Costs a few cents in total: three policies times a two-turn conversation, times
however many scenarios you ask for. Every arm gets its own throwaway database,
so nothing here touches the app's data.

Exit code 1 means the *routed* policy reached the wrong database state -- the
arm under test failing is a real result. The cheap arm failing is not an error,
it is the finding the whole thing exists to produce.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import bench  # noqa: E402
from assistant.config import available_models, default_model, get_model  # noqa: E402
from assistant.llm import default_backend  # noqa: E402
from assistant.routing import default_target  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
RED, GREEN = "\033[31m", "\033[32m"


def _table(rows: list[list[str]], headers: tuple[str, ...]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows, strict=True)]
    line = "  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True))
    out = [f"{BOLD}{line}{RESET}", DIM + "  ".join("-" * width for width in widths) + RESET]
    for row in rows:
        painted = []
        for cell, width in zip(row, widths, strict=True):
            text = str(cell).ljust(width)
            if cell == "ok":
                text = f"{GREEN}{text}{RESET}"
            elif cell == "MAL":
                text = f"{RED}{text}{RESET}"
            painted.append(text)
        out.append("  ".join(painted))
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", default="groq-oss", help="modelo barato que arranca cada turno")
    parser.add_argument("--strong", default=None, help="modelo al que se escala (default: el declarado)")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.key for scenario in bench.SCENARIOS],
        help="correr solo este escenario; repetible",
    )
    parser.add_argument("--markdown", action="store_true", help="salida lista para pegar en el README")
    args = parser.parse_args(argv)

    draft = get_model(args.draft)
    strong = (
        get_model(args.strong)
        if args.strong
        else (default_target(available_models()) or default_model())
    )
    if strong is None:
        print("No hay ningún modelo disponible. Cargá una API key en .env.", file=sys.stderr)
        return 2
    for model in (draft, strong):
        if not model.available:
            print(f"{model.label}: sin credenciales o sin Ollama.", file=sys.stderr)
            return 2

    scenarios = [bench.SCENARIOS_BY_KEY[key] for key in args.scenario] if args.scenario else bench.SCENARIOS
    policies = bench.policies_for(draft, strong)
    backend = default_backend()
    root = Path(tempfile.mkdtemp(prefix="arnie-bench-"))

    failed = False
    for scenario in scenarios:
        if not args.markdown:
            print(f"\n{BOLD}=== {scenario.title} ==={RESET}")
        results = []
        for result in bench.run_scenario(scenario, policies, backend, lambda name: root / f"{name}.db"):
            results.append(result)
            if not args.markdown:
                mark = f"{GREEN}ok{RESET}" if result.outcome.ok else f"{RED}MAL{RESET}"
                print(f"  {result.policy.label:<20} {mark}  {DIM}{result.outcome.detail}{RESET}")
        if result_failed(results):
            failed = True

        if args.markdown:
            print("\n" + bench.markdown_report(scenario, results))
            print("\n" + bench.verdict(results))
        else:
            print()
            print(_table(bench.table_rows(results), bench.HEADERS))
            print(f"\n{bench.verdict(results)}")

    print(f"\n{DIM}bases de la corrida: {root}{RESET}", file=sys.stderr)
    return 1 if failed else 0


def result_failed(results: list[bench.PolicyResult]) -> bool:
    """Only the arm under test counts as a failure of the run."""
    return any(not result.outcome.ok for result in results if result.policy.key == "routed")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
