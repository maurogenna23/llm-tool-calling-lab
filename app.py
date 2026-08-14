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
from datetime import datetime

import gradio as gr
import pandas as pd

from assistant import db, media, prompts
from assistant import telemetry as tel
from assistant.config import (
    BUSINESS,
    DB_PATH,
    IMAGE_CACHE_DIR,
    IMAGES_ENABLED,
    ModelSpec,
    available_models,
    default_model,
    get_model,
    media_enabled,
)
from assistant.llm import Usage, default_backend
from assistant.telemetry import TurnRecord
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
    "dish_image": "Buscando una foto del plato",
}

EXAMPLES = [
    "Hola! Qué postres veganos tienen?",
    "Hay mesa para 4 el viernes a las 21?",
    "Quiero reservar para 2 mañana a las 21 en la terraza, a nombre de Mauro",
    "Qué principal me recomendás por menos de 20?",
    "Mostrame cómo es el risotto de hongos",
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


def transcribe_recording(audio_path: str | None) -> tuple[str, None]:
    """Whisper the clip into the message box, and clear the recorder."""
    try:
        return media.transcribe(audio_path), None
    except Exception as error:  # noqa: BLE001 - a failed transcription is not a crash
        gr.Warning(f"No pude transcribir el audio: {error}")
        return "", None


def respond(
    display: list[dict],
    conversation: list[dict],
    model_key: str,
    telemetry: list[TurnRecord],
    voice: bool,
) -> Iterator[tuple[list[dict], list[dict], list[TurnRecord], str, str | None, str | None]]:
    """Stream one assistant turn, updating the transcript as events arrive."""
    if not display or display[-1]["role"] != "user":
        yield display, conversation, telemetry, "", None, None
        return

    model = get_model(model_key)
    if not conversation:
        conversation = [{"role": "system", "content": prompts.system_prompt(path=DB_PATH)}]
    conversation = [*conversation, {"role": "user", "content": display[-1]["content"]}]

    display = list(display)
    started = time.perf_counter()
    answer_index: int | None = None
    tool_index: int | None = None
    photo: str | None = None
    used_tools: list[str] = []

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
            used_tools.append(event.name)
            image_path = event.result.payload.get("image_path")
            if image_path:
                photo = str(image_path)

        elif isinstance(event, TurnFinished):
            elapsed = time.perf_counter() - started
            conversation = event.messages
            telemetry = [
                *telemetry,
                TurnRecord(
                    at=datetime.now().strftime("%H:%M:%S"),
                    model=model.label,
                    usage=event.usage,
                    seconds=elapsed,
                    rounds=event.rounds,
                    tools=tuple(used_tools),
                ),
            ]
            line = _status_line(event.usage, event.rounds, elapsed, model)
            # Show the text first; speech takes another second or two.
            yield display, conversation, telemetry, line, photo, None
            if voice and event.text.strip():
                try:
                    spoken = media.speak(event.text)
                except Exception as error:  # noqa: BLE001 - never let TTS break a turn
                    gr.Warning(f"No pude generar el audio: {error}")
                    spoken = None
                if spoken is not None:
                    yield display, conversation, telemetry, line, photo, str(spoken)
            return

        elif isinstance(event, LoopAborted):
            display.append({"role": "assistant", "content": f"⚠️ {event.reason}"})
            yield display, conversation, telemetry, "**Turno interrumpido**", photo, None
            return

        yield display, conversation, telemetry, "", photo, None


def reset() -> tuple[list[dict], list[dict], str, None, None]:
    """Clear the conversation. Telemetry survives: it accounts for the session."""
    return [], [], "", None, None


EMPTY_PLOT = pd.DataFrame({"modelo": [], "tokens": []})


def render_telemetry(
    records: list[TurnRecord],
) -> tuple[str, str, list[list[str]], gr.BarPlot, str]:
    frame = tel.plot_frame(records)
    # Vega would otherwise start the axis near the smallest bar, which makes a
    # 46% difference look like 10x. Comparisons have to start at zero.
    top = max((row["tokens"] for row in frame), default=0)
    return (
        tel.summary_markdown(records),
        tel.by_model_markdown(records),
        tel.table_rows(records),
        gr.BarPlot(
            value=pd.DataFrame(frame) if frame else EMPTY_PLOT,
            x="modelo",
            y="tokens",
            y_lim=[0, int(top * 1.15) or 1],
        ),
        tel.media_markdown(media.EVENTS),
    )


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
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="…o hablale: se transcribe y se envía al soltar",
                        show_download_button=False,
                        visible=media_enabled(),
                    )
                    gr.Examples(examples=EXAMPLES, inputs=message, label="Probá con")

                with gr.Column(scale=1):
                    model_picker = gr.Dropdown(
                        choices=[(model.label, model.key) for model in models],
                        value=initial.key,
                        label="Modelo",
                    )
                    note = gr.Markdown(_model_note(initial.key))
                    voice = gr.Checkbox(
                        label="Responder con voz",
                        value=False,
                        info="Suma unos segundos y unos centésimos de centavo por respuesta.",
                        visible=media_enabled(),
                    )
                    dish_photo = gr.Image(
                        label="Plato", height=220, show_download_button=False, visible=IMAGES_ENABLED
                    )
                    reply_audio = gr.Audio(label="Respuesta", autoplay=True, visible=media_enabled())
                    status = gr.Markdown(label="Último turno")
                    clear = gr.Button("Reiniciar conversación", size="sm")

        with gr.Tab("Telemetría"):
            gr.Markdown(
                "Cada turno del chat, con el costo real que devuelve el proveedor. "
                "Se mantiene aunque reinicies la conversación."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    tel_summary = gr.Markdown(tel.summary_markdown([]))
                    tel_media = gr.Markdown(tel.media_markdown([]))
                with gr.Column(scale=3):
                    tel_plot = gr.BarPlot(
                        EMPTY_PLOT,
                        x="modelo",
                        y="tokens",
                        title="Tokens por modelo",
                        height=220,
                    )
                    tel_models = gr.Markdown()
            tel_table = gr.Dataframe(
                headers=list(tel.HEADERS),
                value=[],
                interactive=False,
                wrap=True,
                label="Turno por turno (el más reciente arriba)",
            )

        # events
        telemetry_outputs = [tel_summary, tel_models, tel_table, tel_plot, tel_media]
        model_picker.change(_model_note, inputs=model_picker, outputs=note)
        clear.click(reset, outputs=[chatbot, conversation, status, dish_photo, reply_audio])

        stream_inputs = [chatbot, conversation, model_picker, telemetry, voice]
        stream_outputs = [chatbot, conversation, telemetry, status, dish_photo, reply_audio]
        for trigger in (message.submit, send.click):
            trigger(submit_message, [message, chatbot], [message, chatbot], queue=False).then(
                respond, stream_inputs, stream_outputs
            ).then(render_telemetry, telemetry, telemetry_outputs, queue=False)

        # Voice in: transcribe, drop the text in the box, then run the same turn.
        mic.stop_recording(transcribe_recording, [mic], [message, mic], queue=False).then(
            submit_message, [message, chatbot], [message, chatbot], queue=False
        ).then(respond, stream_inputs, stream_outputs).then(
            render_telemetry, telemetry, telemetry_outputs, queue=False
        )

    return ui


def main() -> None:
    db.bootstrap()
    seeded = db.seed_demo_reservations()
    print(f"Base lista en {DB_PATH} ({seeded} reservas de ejemplo)")
    build_ui().launch(
        inbrowser=False,
        # Gradio only serves files from its working directory or the temp dir.
        # The image cache lives with the project, which is not the same place
        # when the app is launched from elsewhere.
        allowed_paths=[str(IMAGE_CACHE_DIR)],
    )


if __name__ == "__main__":
    main()
