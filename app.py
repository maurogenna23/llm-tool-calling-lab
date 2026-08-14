"""arnie-lab -- the Gradio front end.

This module is deliberately thin: it turns the events coming out of
``tool_loop.run_turn`` into chat bubbles and status lines. No booking rule and
no provider detail lives here.

One thing worth knowing about the session state: the system prompt is built
*once* per conversation, not per turn. It contains the current time, and a
prefix that changes every minute can never be served from the provider's prompt
cache -- which is exactly the discount we want on turn two onwards.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import gradio as gr

from assistant import db, prompts
from assistant.config import (
    BUSINESS,
    DB_PATH,
    ModelSpec,
    available_models,
    default_model,
    get_model,
)
from assistant.llm import Usage, default_backend
from assistant.tool_loop import (
    LoopAborted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    run_turn,
)

BACKEND = default_backend()

#: What the customer-facing activity line says while a tool runs.
TOOL_LABELS = {
    "get_menu": "Mirando la carta",
    "check_availability": "Chequeando disponibilidad",
    "make_reservation": "Confirmando la reserva",
    "cancel_reservation": "Cancelando la reserva",
    "lookup_reservation": "Buscando la reserva",
}

EXAMPLES = [
    "Hola! Qué postres veganos tienen?",
    "Hay mesa para 4 el viernes a las 21?",
    "Quiero reservar para 2 mañana a las 21 en la terraza, a nombre de Mauro",
    "Qué principal me recomendás por menos de 20?",
]


# --------------------------------------------------------------------------
# display helpers
# --------------------------------------------------------------------------


def _tool_bubble(name: str, arguments: dict[str, object], body: str = "") -> dict[str, object]:
    label = TOOL_LABELS.get(name, name)
    detail = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return {
        "role": "assistant",
        "content": body,
        "metadata": {"title": f"🔧 {label}", "log": f"{name}({detail})"},
    }


def _status_line(usage: Usage, rounds: int, seconds: float, model: ModelSpec) -> str:
    cost = "sin precio" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢"
    cached = f" · {usage.cached_tokens} cacheados" if usage.cached_tokens else ""
    return (
        f"**{model.label}** · {usage.prompt_tokens} in / {usage.completion_tokens} out{cached}\n\n"
        f"{cost} · {seconds:.1f} s · {rounds} ronda(s) al modelo"
    )


def _model_note(model_key: str) -> str:
    model = get_model(model_key)
    if not model.supports_tools:
        return (
            "⚠️ Este modelo **no puede usar herramientas**: charla, pero no consulta la carta "
            "ni reserva. Está acá para que se note la diferencia."
        )
    return f"_{model.note}_" if model.note else ""


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------


def submit_message(message: str, display: list[dict]) -> tuple[str, list[dict]]:
    """Clear the box and show the user's message before the model starts."""
    if not message.strip():
        return "", display
    return "", [*display, {"role": "user", "content": message.strip()}]


def respond(
    display: list[dict],
    conversation: list[dict],
    model_key: str,
    telemetry: list[dict],
) -> Iterator[tuple[list[dict], list[dict], list[dict], str]]:
    """Stream one assistant turn, updating the transcript as events arrive."""
    if not display or display[-1]["role"] != "user":
        yield display, conversation, telemetry, ""
        return

    model = get_model(model_key)
    if not conversation:
        conversation = [{"role": "system", "content": prompts.system_prompt(path=DB_PATH)}]
    conversation = [*conversation, {"role": "user", "content": display[-1]["content"]}]

    display = list(display)
    started = time.perf_counter()
    answer_index: int | None = None
    tool_index: int | None = None

    for event in run_turn(conversation, model, BACKEND, path=DB_PATH):
        if isinstance(event, TextDelta):
            if answer_index is None:
                display.append({"role": "assistant", "content": ""})
                answer_index = len(display) - 1
            current = display[answer_index]
            display[answer_index] = {**current, "content": current["content"] + event.text}

        elif isinstance(event, ToolStarted):
            display.append(_tool_bubble(event.name, event.arguments, "…"))
            tool_index, answer_index = len(display) - 1, None

        elif isinstance(event, ToolFinished):
            body = event.result.text if event.result.ok else f"⚠️ {event.result.text}"
            body = f"{body}\n\n`{event.elapsed_ms} ms`"
            if tool_index is None:  # arguments failed to parse: there was no start
                display.append(_tool_bubble(event.name, {}, body))
            else:
                display[tool_index] = {**display[tool_index], "content": body}
            tool_index = None

        elif isinstance(event, TurnFinished):
            elapsed = time.perf_counter() - started
            conversation = event.messages
            telemetry = [
                *telemetry,
                {
                    "model": model.label,
                    "in": event.usage.prompt_tokens,
                    "out": event.usage.completion_tokens,
                    "cached": event.usage.cached_tokens,
                    "cost_usd": event.usage.cost_usd,
                    "seconds": elapsed,
                    "rounds": event.rounds,
                },
            ]
            yield display, conversation, telemetry, _status_line(
                event.usage, event.rounds, elapsed, model
            )
            return

        elif isinstance(event, LoopAborted):
            display.append({"role": "assistant", "content": f"⚠️ {event.reason}"})
            yield display, conversation, telemetry, "**Turno interrumpido**"
            return

        yield display, conversation, telemetry, ""


def reset() -> tuple[list[dict], list[dict], str]:
    return [], [], ""


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    models = available_models()
    initial = default_model()
    if initial is None:
        raise SystemExit(
            "No hay ningún modelo disponible. Cargá al menos una API key en .env "
            "o levantá Ollama, y volvé a probar con `python -m assistant.config`."
        )

    with gr.Blocks(title=f"{BUSINESS.name} · arnie-lab", fill_height=True) as ui:
        gr.Markdown(
            f"### {BUSINESS.name} "
            f"<span style='font-weight:400;opacity:.6'>· asistente con herramientas reales "
            "sobre SQLite</span>"
        )

        conversation = gr.State([])  # the raw transcript sent to the model
        telemetry = gr.State([])  # one record per completed turn

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        type="messages",
                        # Viewport-relative so the composer stays visible on a
                        # laptop without scrolling the whole page.
                        height="58vh",
                        show_label=False,
                        allow_tags=False,
                        placeholder="<center>Preguntá por la carta, o pedí una mesa.</center>",
                    )
                    with gr.Row():
                        message = gr.Textbox(
                            placeholder="Escribí tu mensaje…",
                            show_label=False,
                            scale=9,
                            autofocus=True,
                            # lines=1 keeps Enter as "send" instead of "newline".
                            lines=1,
                            max_lines=4,
                        )
                        send = gr.Button("Enviar", variant="primary", scale=1, min_width=90)
                    gr.Examples(examples=EXAMPLES, inputs=message, label="Probá con")

                with gr.Column(scale=1):
                    model_picker = gr.Dropdown(
                        choices=[(model.label, model.key) for model in models],
                        value=initial.key,
                        label="Modelo",
                    )
                    note = gr.Markdown(_model_note(initial.key))
                    status = gr.Markdown(label="Último turno")
                    clear = gr.Button("Reiniciar conversación", size="sm")

        # events
        model_picker.change(_model_note, inputs=model_picker, outputs=note)
        clear.click(reset, outputs=[chatbot, conversation, status])

        stream_inputs = [chatbot, conversation, model_picker, telemetry]
        stream_outputs = [chatbot, conversation, telemetry, status]
        for trigger in (message.submit, send.click):
            trigger(submit_message, [message, chatbot], [message, chatbot], queue=False).then(
                respond, stream_inputs, stream_outputs
            )

    return ui


def main() -> None:
    db.bootstrap()
    seeded = db.seed_demo_reservations()
    print(f"Base lista en {DB_PATH} ({seeded} reservas de ejemplo)")
    build_ui().launch(inbrowser=False)


if __name__ == "__main__":
    main()
