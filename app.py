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
from dataclasses import dataclass, field
from datetime import datetime

import gradio as gr
import pandas as pd

from assistant import db, media, prompts
from assistant import telemetry as tel
from assistant.arena import HEADERS as ARENA_HEADERS
from assistant.arena import ArenaSlot, run_arena
from assistant.arena import table_rows as arena_table_rows
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
    ApprovalRequested,
    LoopAborted,
    TextDelta,
    ToolFinished,
    ToolRejected,
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


@dataclass
class TurnContext:
    """Everything a turn needs to survive being paused for a confirmation."""

    model_key: str
    display: list[dict]
    conversation: list[dict]
    started: float
    used_tools: list[str] = field(default_factory=list)
    photo: str | None = None
    answer_index: int | None = None
    tool_index: int | None = None


@dataclass
class ParkedTurn:
    """A turn frozen mid-flight, waiting for the user to allow a write."""

    generator: object
    context: TurnContext
    request: ApprovalRequested


def _confirm_text(request: ApprovalRequested) -> str:
    label = TOOL_LABELS.get(request.name, request.name)
    detail = "\n".join(f"- **{key}**: {value}" for key, value in request.arguments.items())
    return f"### ⏸️ {label}\n\nEl asistente quiere ejecutar `{request.name}`:\n\n{detail}"


def _frame(
    context: TurnContext,
    telemetry: list[TurnRecord],
    status: str,
    audio: str | None = None,
    parked: ParkedTurn | None = None,
) -> tuple:
    return (
        context.display,
        context.conversation,
        telemetry,
        status,
        context.photo,
        audio,
        parked,
        gr.update(visible=parked is not None),
        _confirm_text(parked.request) if parked else "",
    )


def _pump(
    generator: Iterator,
    decision: bool | None,
    context: TurnContext,
    telemetry: list[TurnRecord],
    voice: bool,
) -> Iterator[tuple]:
    """Drive the tool loop, yielding UI frames and parking on approval requests."""
    model = get_model(context.model_key)

    while True:
        try:
            event = generator.send(decision)
        except StopIteration:
            return
        decision = None

        if isinstance(event, TextDelta):
            if context.answer_index is None:
                context.display.append({"role": "assistant", "content": ""})
                context.answer_index = len(context.display) - 1
            current = context.display[context.answer_index]
            context.display[context.answer_index] = {
                **current,
                "content": current["content"] + event.text,
            }

        elif isinstance(event, ToolStarted):
            context.display.append(_tool_bubble(event.name, event.arguments, "…"))
            context.tool_index, context.answer_index = len(context.display) - 1, None

        elif isinstance(event, ToolFinished):
            body = event.result.text if event.result.ok else f"⚠️ {event.result.text}"
            body = f"{body}\n\n`{event.elapsed_ms} ms`"
            if context.tool_index is None:  # arguments failed to parse: there was no start
                context.display.append(_tool_bubble(event.name, {}, body))
            else:
                context.display[context.tool_index] = {
                    **context.display[context.tool_index],
                    "content": body,
                }
            context.tool_index = None
            context.used_tools.append(event.name)
            image_path = event.result.payload.get("image_path")
            if image_path:
                context.photo = str(image_path)

        elif isinstance(event, ApprovalRequested):
            # Park here. The confirm/reject buttons resume this same generator.
            yield _frame(
                context,
                telemetry,
                "**Esperando tu confirmación**",
                parked=ParkedTurn(generator, context, event),
            )
            return

        elif isinstance(event, ToolRejected):
            context.display.append(
                _tool_bubble(event.name, event.arguments, "🚫 No autorizaste esta acción.")
            )
            context.answer_index = None

        elif isinstance(event, TurnFinished):
            elapsed = time.perf_counter() - context.started
            context.conversation = event.messages
            telemetry = [
                *telemetry,
                TurnRecord(
                    at=datetime.now().strftime("%H:%M:%S"),
                    model=model.label,
                    usage=event.usage,
                    seconds=elapsed,
                    rounds=event.rounds,
                    tools=tuple(context.used_tools),
                ),
            ]
            line = _status_line(event.usage, event.rounds, elapsed, model)
            # Show the text first; speech takes another second or two.
            yield _frame(context, telemetry, line)
            if voice and event.text.strip():
                try:
                    spoken = media.speak(event.text)
                except Exception as error:  # noqa: BLE001 - never let TTS break a turn
                    gr.Warning(f"No pude generar el audio: {error}")
                    spoken = None
                if spoken is not None:
                    yield _frame(context, telemetry, line, audio=str(spoken))
            return

        elif isinstance(event, LoopAborted):
            context.display.append({"role": "assistant", "content": f"⚠️ {event.reason}"})
            yield _frame(context, telemetry, "**Turno interrumpido**")
            return

        yield _frame(context, telemetry, "")


def respond(
    display: list[dict],
    conversation: list[dict],
    model_key: str,
    telemetry: list[TurnRecord],
    voice: bool,
    confirm_writes: bool,
) -> Iterator[tuple]:
    """Stream one assistant turn, updating the transcript as events arrive."""
    if not display or display[-1]["role"] != "user":
        yield display, conversation, telemetry, "", None, None, None, gr.update(visible=False), ""
        return

    if not conversation:
        conversation = [{"role": "system", "content": prompts.system_prompt(path=DB_PATH)}]

    context = TurnContext(
        model_key=model_key,
        display=list(display),
        conversation=[*conversation, {"role": "user", "content": display[-1]["content"]}],
        started=time.perf_counter(),
    )
    generator = run_turn(
        context.conversation,
        get_model(model_key),
        BACKEND,
        require_approval=confirm_writes,
        path=DB_PATH,
    )
    yield from _pump(generator, None, context, telemetry, voice)


def resume(
    parked: ParkedTurn | None, approved: bool, telemetry: list[TurnRecord], voice: bool
) -> Iterator[tuple]:
    """Answer the pending confirmation and let the turn finish."""
    if parked is None:
        return
    yield from _pump(parked.generator, approved, parked.context, telemetry, voice)


def resume_yes(parked: ParkedTurn | None, telemetry: list[TurnRecord], voice: bool) -> Iterator[tuple]:
    yield from resume(parked, True, telemetry, voice)


def resume_no(parked: ParkedTurn | None, telemetry: list[TurnRecord], voice: bool) -> Iterator[tuple]:
    yield from resume(parked, False, telemetry, voice)


MAX_ARENA_COLUMNS = 4


def _arena_column(slot: ArenaSlot) -> str:
    header = f"**{slot.model.label}**"
    if slot.error:
        return f"{header}\n\n⚠️ {slot.error}"
    if slot.first_token_seconds is not None:
        header += f"  \n<sub>1er token: {slot.first_token_seconds:.2f} s</sub>"
    body = slot.text or "_…_"
    return f"{header}\n\n{body}"


def compare(prompt: str, model_keys: list[str]) -> Iterator[tuple]:
    """Stream the same prompt through every selected model at once."""
    blanks = [""] * MAX_ARENA_COLUMNS
    if not prompt.strip():
        gr.Warning("Escribí un prompt para comparar.")
        yield (*blanks, [])
        return
    if not model_keys:
        gr.Warning("Elegí al menos un modelo.")
        yield (*blanks, [])
        return

    if len(model_keys) > MAX_ARENA_COLUMNS:
        gr.Warning(f"Comparo los primeros {MAX_ARENA_COLUMNS}; el resto queda afuera.")
    models = [get_model(key) for key in model_keys[:MAX_ARENA_COLUMNS]]

    for slots in run_arena(prompt, models, BACKEND, system=prompts.arena_prompt()):
        columns = [_arena_column(slot) for slot in slots]
        columns += [""] * (MAX_ARENA_COLUMNS - len(columns))
        yield (*columns, arena_table_rows(slots))


def reset() -> tuple[list[dict], list[dict], str, None, None, None, dict]:
    """Clear the conversation. Telemetry survives: it accounts for the session."""
    return [], [], "", None, None, None, gr.update(visible=False)


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
        parked = gr.State(None)  # a turn paused waiting for a confirmation

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
                    with gr.Row(visible=False) as confirm_row:
                        with gr.Column():
                            confirm_text = gr.Markdown()
                            with gr.Row():
                                approve_button = gr.Button("Confirmar", variant="primary")
                                reject_button = gr.Button("Rechazar", variant="stop")
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
                    confirm_writes = gr.Checkbox(
                        label="Confirmar antes de escribir",
                        value=True,
                        info="Reservar y cancelar te piden permiso antes de tocar la base.",
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

        with gr.Tab("Arena"):
            gr.Markdown(
                "El mismo prompt contra varios modelos **en paralelo**, sin herramientas: "
                "acá se compara el modelo crudo. La tabla ordena por tiempo hasta el primer token, "
                "que es lo que define qué tan rápido se *siente* una respuesta."
            )
            with gr.Row():
                arena_prompt = gr.Textbox(
                    placeholder="Ej: explicá qué es el prompt caching en dos oraciones",
                    show_label=False,
                    lines=1,
                    max_lines=3,
                    scale=8,
                )
                arena_go = gr.Button("Comparar", variant="primary", scale=1, min_width=110)
            arena_models = gr.CheckboxGroup(
                choices=[(model.label, model.key) for model in models],
                value=[model.key for model in models[:3]],
                label=f"Modelos (hasta {MAX_ARENA_COLUMNS})",
            )
            with gr.Row(equal_height=False):
                arena_columns = [gr.Markdown() for _ in range(MAX_ARENA_COLUMNS)]
            arena_table = gr.Dataframe(
                headers=list(ARENA_HEADERS), value=[], interactive=False, wrap=True
            )

        # events
        telemetry_outputs = [tel_summary, tel_models, tel_table, tel_plot, tel_media]
        arena_outputs = [*arena_columns, arena_table]
        for trigger in (arena_prompt.submit, arena_go.click):
            trigger(compare, [arena_prompt, arena_models], arena_outputs)
        model_picker.change(_model_note, inputs=model_picker, outputs=note)
        clear.click(
            reset, outputs=[chatbot, conversation, status, dish_photo, reply_audio, parked, confirm_row]
        )

        stream_inputs = [chatbot, conversation, model_picker, telemetry, voice, confirm_writes]
        stream_outputs = [
            chatbot,
            conversation,
            telemetry,
            status,
            dish_photo,
            reply_audio,
            parked,
            confirm_row,
            confirm_text,
        ]
        resume_inputs = [parked, telemetry, voice]
        for trigger in (message.submit, send.click):
            trigger(submit_message, [message, chatbot], [message, chatbot], queue=False).then(
                respond, stream_inputs, stream_outputs
            ).then(render_telemetry, telemetry, telemetry_outputs, queue=False)

        for button, handler in ((approve_button, resume_yes), (reject_button, resume_no)):
            button.click(handler, resume_inputs, stream_outputs).then(
                render_telemetry, telemetry, telemetry_outputs, queue=False
            )

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
