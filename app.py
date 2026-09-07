"""llm-tool-calling-lab -- the Gradio front end.

This module is deliberately thin: it turns the events coming out of
``tool_loop.run_turn`` into chat bubbles and status lines. No booking rule and
no provider detail lives here.

One thing worth knowing about the session state: the system prompt is built
*once* per conversation, not per turn. It contains the current time, and a
prefix that changes every minute can never be served from the provider's prompt
cache -- which is exactly the discount we want on turn two onwards.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from assistant import bench, db, media, prompts, routing
from assistant import telemetry as tel
from assistant.arena import HEADERS as ARENA_HEADERS
from assistant.arena import ArenaSlot, run_arena
from assistant.arena import table_rows as arena_table_rows
from assistant.config import (
    BENCH_ENABLED,
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
    Escalated,
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


def _escalation_bubble(event: Escalated) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": f"Sigo con **{event.to_model.label}**.",
        "metadata": {
            "title": f"⬆️ Cambio de modelo — {event.from_model.label} {event.reason}",
            "log": f"{event.from_model.key} → {event.to_model.key} ({event.trigger})",
        },
    }


def _status_line(
    usage: Usage,
    rounds: int,
    seconds: float,
    model: ModelSpec,
    route: tuple[str, ...] = (),
    first_token: float | None = None,
) -> str:
    cost = "sin precio" if usage.cost_usd is None else f"{usage.cost_usd * 100:.4f} ¢"
    cached = f" · {usage.cached_tokens} cacheados" if usage.cached_tokens else ""
    who = " → ".join(label.split(" · ")[0] for label in route) if len(route) > 1 else model.label
    opening = "" if first_token is None else f"{first_token:.2f} s al 1er token · "
    return (
        f"**{who}** · {usage.prompt_tokens} in / {usage.completion_tokens} out{cached}\n\n"
        f"{cost} · {opening}{seconds:.1f} s · {rounds} ronda(s) al modelo"
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
    #: Whether this turn was allowed to change models at all, and what made it.
    could_escalate: bool = False
    trigger: str | None = None
    #: Time to the first token the user got to keep -- reset when a turn changes
    #: hands, because the draft's text is taken back off the screen.
    first_token: float | None = None


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
            if context.first_token is None:
                context.first_token = time.perf_counter() - context.started
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

        elif isinstance(event, Escalated):
            # The round that triggered this was thrown away, so anything the
            # draft had already streamed has to come off the screen with it.
            if context.answer_index == len(context.display) - 1:
                context.display.pop()
            context.answer_index = None
            context.first_token = None
            context.model_key = event.to_model.key
            context.trigger = event.trigger
            context.display.append(_escalation_bubble(event))
            model = event.to_model

        elif isinstance(event, ToolRejected):
            context.display.append(
                _tool_bubble(event.name, event.arguments, "🚫 No autorizaste esta acción.")
            )
            context.answer_index = None

        elif isinstance(event, TurnFinished):
            elapsed = time.perf_counter() - context.started
            context.conversation = event.messages
            route = tuple(get_model(key).label for key in event.route)
            telemetry = [
                *telemetry,
                TurnRecord(
                    at=datetime.now().strftime("%H:%M:%S"),
                    model=model.label,
                    usage=event.usage,
                    seconds=elapsed,
                    rounds=event.rounds,
                    tools=tuple(context.used_tools),
                    first_token_seconds=context.first_token,
                    route=route,
                    trigger=context.trigger,
                    spend={
                        get_model(key).label: usage for key, usage in event.spend.items()
                    },
                    could_escalate=context.could_escalate,
                ),
            ]
            line = _status_line(
                event.usage, event.rounds, elapsed, model, route, context.first_token
            )
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
    escalate: bool,
    strong_key: str | None,
) -> Iterator[tuple]:
    """Stream one assistant turn, updating the transcript as events arrive."""
    if not display or display[-1]["role"] != "user":
        yield display, conversation, telemetry, "", None, None, None, gr.update(visible=False), ""
        return

    if not conversation:
        conversation = [{"role": "system", "content": prompts.system_prompt(path=DB_PATH)}]

    draft = get_model(model_key)
    target = get_model(strong_key) if (escalate and strong_key) else None
    # Picking the draft model as its own escalation target, or one that cannot
    # call tools, is a route that goes nowhere. Drop it rather than pretend.
    if not routing.can_escalate(draft, target):
        target = None

    context = TurnContext(
        model_key=model_key,
        display=list(display),
        conversation=[*conversation, {"role": "user", "content": display[-1]["content"]}],
        started=time.perf_counter(),
        could_escalate=target is not None,
    )
    generator = run_turn(
        context.conversation,
        draft,
        BACKEND,
        require_approval=confirm_writes,
        escalate_to=target,
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


def compare_policies(scenario_key: str, draft_key: str, strong_key: str) -> Iterator[tuple]:
    """Replay one scenario under the three routing policies, redrawing as it goes."""
    if not BENCH_ENABLED:
        gr.Warning("El banco está apagado en esta instalación (ARNIE_BENCH=off).")
        yield [], "", ""
        return

    draft, strong = get_model(draft_key), get_model(strong_key)
    if not routing.can_escalate(draft, strong):
        gr.Warning("Elegí un borrador y un modelo fuerte distintos, ambos con herramientas.")
        yield [], "", ""
        return

    scenario = bench.SCENARIOS_BY_KEY[scenario_key]
    policies = bench.policies_for(draft, strong)
    # A throwaway database per arm, never the app's: the bench books tables.
    root = Path(tempfile.mkdtemp(prefix="arnie-bench-"))

    results: list[bench.PolicyResult] = []
    for results in bench.stream_scenario(
        scenario, policies, BACKEND, lambda name: root / f"{name}.db"
    ):
        complete = bench.all_done(results, len(policies))
        yield (
            bench.table_rows(results),
            bench.verdict(results) if complete else "_Corriendo…_",
            bench.markdown_report(scenario, results) if complete else "",
        )


def _scenario_note(scenario_key: str) -> str:
    scenario = bench.SCENARIOS_BY_KEY[scenario_key]
    expect = scenario.expect
    wanted = f"{expect.confirmed} reserva(s) vigente(s)"
    if expect.cancelled:
        wanted += f" y {expect.cancelled} cancelada(s)"
    if expect.at_hour is not None:
        wanted += f", a las {expect.at_hour}:00"
    turns = len(scenario.messages)
    return (
        f"**{turns} turnos.** Al terminar la base tiene que quedar con {wanted} — "
        "una política que llega a otro estado perdió, cueste lo que cueste."
    )


def reset() -> tuple[list[dict], list[dict], str, None, None, None, dict]:
    """Clear the conversation. Telemetry survives: it accounts for the session."""
    return [], [], "", None, None, None, gr.update(visible=False)


EMPTY_PLOT = pd.DataFrame({"modelo": [], "tokens": []})


def render_telemetry(
    records: list[TurnRecord],
) -> tuple[str, str, list[list[str]], gr.BarPlot, str, str]:
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
        tel.routing_markdown(records),
    )


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    models = available_models()
    initial = default_model()
    targets = [model for model in models if model.supports_tools and model.trusted_for_writes]
    target = routing.default_target(models)
    if initial is None:
        raise SystemExit(
            "No hay ningún modelo disponible. Cargá al menos una API key en .env "
            "o levantá Ollama, y volvé a probar con `python -m assistant.config`."
        )

    with gr.Blocks(title=f"{BUSINESS.name} · llm-tool-calling-lab", fill_height=True) as ui:
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
                    escalate = gr.Checkbox(
                        label="Escalar cuando haga falta",
                        # Only pre-ticked when it can actually do something. A
                        # checked box over a route that goes nowhere lies.
                        value=routing.can_escalate(initial, target),
                        info=(
                            "El turno arranca en el modelo de arriba y cambia de manos si pide "
                            "escribir, manda argumentos inservibles, repite una llamada o se traba."
                        ),
                        visible=target is not None,
                    )
                    strong_picker = gr.Dropdown(
                        choices=[(model.label, model.key) for model in targets],
                        value=target.key if target else None,
                        label="Escalar a",
                        visible=target is not None,
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
                    tel_routing = gr.Markdown(tel.routing_markdown([]))
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

        bench_tab_visible = BENCH_ENABLED and target is not None
        with gr.Tab("Banco", visible=bench_tab_visible):
            gr.Markdown(
                "La misma conversación bajo **tres políticas**: siempre el modelo grande, "
                "siempre el chico, y ruteado. Se puntúa contra la **base de datos**, no "
                "contra lo que dijo el asistente: un modelo que contesta *«listo, ya te la "
                "cambié»* y deja las dos mesas tomadas falló, por linda que sea la oración.\n\n"
                "⚠️ Es el botón más caro de la app: tres conversaciones completas contra "
                "proveedores reales, unos centavos por corrida. Cada brazo usa su propia "
                "base descartable, así que no toca los datos del restaurante."
            )
            with gr.Row():
                bench_scenario = gr.Dropdown(
                    choices=[(scenario.title, scenario.key) for scenario in bench.SCENARIOS],
                    value="modification",
                    label="Escenario",
                    scale=3,
                )
                bench_draft = gr.Dropdown(
                    choices=[(model.label, model.key) for model in models if model.supports_tools],
                    # A model that is not the escalation target, so the button
                    # does something the first time it is pressed.
                    value=(routing.default_draft(models) or initial).key,
                    label="Borrador",
                    scale=2,
                )
                bench_strong = gr.Dropdown(
                    choices=[(model.label, model.key) for model in targets],
                    value=target.key if target else None,
                    label="Modelo fuerte",
                    scale=2,
                )
                bench_go = gr.Button("Correr", variant="primary", scale=1, min_width=110)
            bench_note = gr.Markdown(_scenario_note("modification"))
            bench_table = gr.Dataframe(
                headers=list(bench.HEADERS), value=[], interactive=False, wrap=True
            )
            bench_verdict = gr.Markdown()
            with gr.Accordion("Reporte para pegar en el README", open=False):
                bench_report = gr.Markdown()

        # events
        telemetry_outputs = [tel_summary, tel_models, tel_table, tel_plot, tel_media, tel_routing]
        bench_scenario.change(_scenario_note, inputs=bench_scenario, outputs=bench_note)
        bench_go.click(
            compare_policies,
            [bench_scenario, bench_draft, bench_strong],
            [bench_table, bench_verdict, bench_report],
        )
        arena_outputs = [*arena_columns, arena_table]
        for trigger in (arena_prompt.submit, arena_go.click):
            trigger(compare, [arena_prompt, arena_models], arena_outputs)
        model_picker.change(_model_note, inputs=model_picker, outputs=note)
        clear.click(
            reset, outputs=[chatbot, conversation, status, dish_photo, reply_audio, parked, confirm_row]
        )

        stream_inputs = [
            chatbot,
            conversation,
            model_picker,
            telemetry,
            voice,
            confirm_writes,
            escalate,
            strong_picker,
        ]
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
