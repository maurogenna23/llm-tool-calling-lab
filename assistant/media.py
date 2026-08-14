"""Images, speech and transcription.

These three go through the OpenAI SDK directly rather than LiteLLM: the chat
gateway earns its keep by normalising usage and cost across providers, and none
of that applies here.

Cost is the design constraint. Image generation is the only part of this app
that costs real money per call, so it is bounded three ways:

* only dishes that exist on the menu can be drawn -- an arbitrary prompt is
  refused, which caps the number of distinct images at the size of the menu;
* every image is cached on disk under its slug, so the second demo is free;
* ``ARNIE_IMAGES=off`` disables it outright, which is what a public deployment
  running on someone's personal API key wants.
"""

from __future__ import annotations

import base64
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from assistant import db
from assistant.config import (
    BUSINESS,
    IMAGE_CACHE_DIR,
    IMAGE_MODEL,
    IMAGES_ENABLED,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
    media_enabled,
)

#: Longest reply we will read out loud. Beyond this, speech is slow, expensive
#: and nobody listens to the end anyway.
MAX_SPEECH_CHARS = 600


@dataclass(frozen=True)
class MediaEvent:
    """One billable media call, for the telemetry tab."""

    kind: str  # image | speech | transcription
    detail: str
    cached: bool = False


#: Appended to as the session runs. Read by the telemetry tab.
EVENTS: list[MediaEvent] = []


def _client():  # noqa: ANN202 - the OpenAI client type is not worth importing eagerly
    from openai import OpenAI

    return OpenAI()


def slug(name: str) -> str:
    """A stable, filesystem-safe cache key: 'Sorrentinos de calabaza' -> sorrentinos-de-calabaza."""
    folded = unicodedata.normalize("NFKD", name.strip().lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "-", folded).strip("-")


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------


def dish_image(dish_name: str, path: Path | None = None) -> tuple[Path | None, str]:
    """Return ``(image_path, explanation)`` for a dish photo, generating it once.

    The explanation is what the model is told; the image path is for the UI.
    ``path`` is the database to look the dish up in.
    """
    if not IMAGES_ENABLED:
        return None, "La generación de imágenes está apagada en esta instalación."
    if not media_enabled():
        return None, "No hay credenciales de OpenAI para generar imágenes."

    dish = db.find_dish(dish_name, path=path)
    if dish is None:
        return None, f"'{dish_name}' no está en la carta, así que no puedo mostrar una foto."

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_CACHE_DIR / f"{slug(dish.name)}.png"
    if path.exists():
        EVENTS.append(MediaEvent("image", dish.name, cached=True))
        return path, f"Foto de {dish.name} (ya estaba generada)."

    prompt = (
        f"{dish.name}: {dish.description}. "
        f"Plato de un {BUSINESS.kind}. {BUSINESS.image_style}. Sin texto ni logos."
    )
    response = _client().images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        # 'low' is a tenth of the price and plenty for a chat thumbnail.
        quality="low",
        n=1,
    )
    path.write_bytes(base64.b64decode(response.data[0].b64_json))
    EVENTS.append(MediaEvent("image", dish.name))
    return path, f"Foto de {dish.name} generada."


# --------------------------------------------------------------------------
# speech
# --------------------------------------------------------------------------


def speak(text: str) -> Path | None:
    """Render a reply as audio. Returns ``None`` when speech is unavailable."""
    spoken = text.strip()
    if not spoken or not media_enabled():
        return None
    if len(spoken) > MAX_SPEECH_CHARS:
        spoken = spoken[:MAX_SPEECH_CHARS].rsplit(" ", 1)[0] + "…"

    response = _client().audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=spoken)
    handle = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    handle.write(response.content)
    handle.close()
    EVENTS.append(MediaEvent("speech", f"{len(spoken)} caracteres"))
    return Path(handle.name)


def transcribe(audio_path: str | Path | None) -> str:
    """Turn a recording into text. Returns an empty string when there is nothing to do."""
    if not audio_path or not media_enabled():
        return ""
    path = Path(audio_path)
    if not path.exists() or path.stat().st_size == 0:
        return ""

    with path.open("rb") as handle:
        response = _client().audio.transcriptions.create(model=STT_MODEL, file=handle)
    text = (response.text or "").strip()
    EVENTS.append(MediaEvent("transcription", f"{len(text)} caracteres"))
    return text
