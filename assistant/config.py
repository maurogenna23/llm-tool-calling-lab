"""Runtime configuration: the business profile and the model registry.

Two ideas live here:

1. ``BusinessProfile`` keeps every domain-specific string in one place, so the
   assistant can be re-pointed at a different kind of business by editing this
   file plus ``data/seed.sql`` -- never the tool loop or the UI.

2. ``MODELS`` declares what each model can actually *do*. Not every model
   supports tool calling (``deepseek-r1:1.5b`` does not), and local models are
   only reachable when Ollama is running. Declaring capabilities up front lets
   the UI degrade honestly instead of failing at request time.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_CACHE_DIR = DATA_DIR / "dish_images"

# Anchored to the package, not to the current directory: the app has to work
# when it is launched from somewhere else, and dotenv's search-upwards default
# breaks outright when there is no file to search from (piped scripts).
load_dotenv(ROOT / ".env", override=True)

def _resolve(value: str | None, default: Path) -> Path:
    """Resolve a configured path against the project, never the current directory.

    A relative default would put the database wherever the process happened to
    be started from -- which quietly creates a second, empty database the first
    time you launch from somewhere else.
    """
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


DB_PATH = _resolve(os.getenv("ARNIE_DB_PATH"), ROOT / "arnie.db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Media models. These are OpenAI-only, so the media features switch off
# together with OPENAI_API_KEY rather than per chat model.
IMAGE_MODEL = "gpt-image-1-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = os.getenv("ARNIE_TTS_VOICE", "onyx")
STT_MODEL = "whisper-1"


@dataclass(frozen=True)
class BusinessProfile:
    """Everything the assistant needs to know about the business it works for."""

    name: str
    kind: str
    city: str
    currency_symbol: str
    #: How long a table is held, in minutes. Drives the availability query.
    booking_window_minutes: int
    #: Party sizes outside this range must be handled by a human.
    min_party_size: int
    max_party_size: int
    #: Style hint passed to the image model so every dish photo looks consistent.
    image_style: str


BUSINESS = BusinessProfile(
    name="Bodegón Aurora",
    kind="restaurante de cocina de autor",
    city="Buenos Aires",
    currency_symbol="$",
    booking_window_minutes=90,
    min_party_size=1,
    max_party_size=8,
    image_style="food photography, natural light, shallow depth of field, on a rustic wooden table",
)


@dataclass(frozen=True)
class ModelSpec:
    """A chat model plus the capabilities the rest of the app may rely on."""

    key: str
    litellm_id: str
    label: str
    supports_tools: bool
    is_local: bool
    #: Environment variable that must be set for this model to be usable.
    requires_env: str | None = None
    #: Shown in the UI when the model is picked, e.g. to explain a limitation.
    note: str = ""

    @property
    def available(self) -> bool:
        if self.is_local:
            return ollama_is_running()
        return bool(os.getenv(self.requires_env or ""))


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="gpt-4.1-mini",
        litellm_id="openai/gpt-4.1-mini",
        label="GPT-4.1 mini · OpenAI",
        supports_tools=True,
        is_local=False,
        requires_env="OPENAI_API_KEY",
    ),
    ModelSpec(
        key="gpt-4.1-nano",
        litellm_id="openai/gpt-4.1-nano",
        label="GPT-4.1 nano · OpenAI",
        supports_tools=True,
        is_local=False,
        requires_env="OPENAI_API_KEY",
        note="Cheapest cloud option; occasionally skips a tool call it should make.",
    ),
    ModelSpec(
        key="gemini-flash-lite",
        # gemini-2.5-flash is closed to new API keys; 3.1-flash-lite is current,
        # supports tool calling and is ~20x cheaper than 3.5-flash.
        litellm_id="gemini/gemini-3.1-flash-lite",
        label="Gemini 3.1 Flash Lite · Google",
        supports_tools=True,
        is_local=False,
        requires_env="GOOGLE_API_KEY",
    ),
    ModelSpec(
        key="groq-oss",
        litellm_id="groq/openai/gpt-oss-120b",
        label="GPT-OSS 120B · Groq",
        supports_tools=True,
        is_local=False,
        requires_env="GROQ_API_KEY",
        note=(
            "The fastest of the bunch. Groq's free tier caps tokens per minute, "
            "so a long tool-heavy conversation will hit a rate limit."
        ),
    ),
    ModelSpec(
        key="llama3.2",
        litellm_id="ollama_chat/llama3.2",
        label="Llama 3.2 3B · local",
        supports_tools=True,
        is_local=True,
        note="Runs on your machine, costs nothing, and is noticeably less reliable at tool calling.",
    ),
    ModelSpec(
        key="deepseek-r1",
        litellm_id="ollama_chat/deepseek-r1:1.5b",
        label="DeepSeek-R1 1.5B · local",
        supports_tools=False,
        is_local=True,
        note="No tool calling: it can chat about the menu but cannot look anything up or book a table.",
    ),
)

MODELS_BY_KEY: dict[str, ModelSpec] = {model.key: model for model in MODELS}


@lru_cache(maxsize=1)
def ollama_is_running() -> bool:
    """Ping Ollama once per process. Hosted deployments have no local models."""
    try:
        with urllib.request.urlopen(OLLAMA_BASE_URL, timeout=0.7) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def available_models() -> list[ModelSpec]:
    return [model for model in MODELS if model.available]


def default_model() -> ModelSpec | None:
    """Prefer a cloud model that supports tools, then anything usable."""
    usable = available_models()
    for model in usable:
        if model.supports_tools and not model.is_local:
            return model
    return usable[0] if usable else None


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown model {key!r}. Known: {', '.join(MODELS_BY_KEY)}") from None


def media_enabled() -> bool:
    """Image generation, TTS and STT all run through OpenAI."""
    return bool(os.getenv("OPENAI_API_KEY"))


def describe_environment() -> str:
    """Human-readable startup report — also the body of ``python -m assistant.config``."""
    lines = [f"{BUSINESS.name} ({BUSINESS.kind})", ""]
    lines.append(f"Ollama at {OLLAMA_BASE_URL}: {'reachable' if ollama_is_running() else 'not reachable'}")
    lines.append(f"Media (image/TTS/STT): {'enabled' if media_enabled() else 'disabled, no OPENAI_API_KEY'}")
    lines.append("")
    lines.append("Models:")
    for model in MODELS:
        mark = "ok " if model.available else "-- "
        why = "" if model.available else f"  (needs {model.requires_env or 'Ollama running'})"
        tools = "tools" if model.supports_tools else "no tools"
        lines.append(f"  {mark}{model.label:<28} {tools}{why}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_environment())
