"""The model gateway.

Everything that talks to a provider goes through :class:`ChatBackend`. There
are two implementations: LiteLLM for real calls, and a scripted fake in the
test suite. The tool loop only ever sees the protocol, which is what makes the
loop testable without a network or an API key.

Why LiteLLM rather than the OpenAI client with a swapped ``base_url``: it
normalises ``usage`` across providers and knows the price of each model, which
is what the telemetry tab is built on. The cost is one more layer of
indirection.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from assistant.config import ModelSpec

Message = dict[str, object]


@dataclass(frozen=True)
class Usage:
    """Token and cost accounting for a single provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Prompt tokens served from the provider's cache. Free or heavily discounted.
    cached_tokens: int = 0
    #: ``None`` when the provider does not report a price we can trust.
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost_usd=cost,
        )


class ChatBackend(Protocol):
    """What the tool loop needs from a provider."""

    def stream(
        self, messages: Sequence[Message], model: ModelSpec, tools: list[dict[str, object]] | None
    ) -> Iterator[object]:
        """Yield streaming chunks shaped like an OpenAI chat completion delta."""

    def usage(self, chunks: Sequence[object], messages: Sequence[Message], model: ModelSpec) -> Usage:
        """Accounting for the chunks produced by one :meth:`stream` call."""


class LiteLLMBackend:
    """Real calls. ``litellm`` is imported lazily -- it is slow to import."""

    def __init__(self) -> None:
        self._configured = False

    def _litellm(self):  # noqa: ANN202 - third-party module
        import litellm

        if not self._configured:
            # Providers reject params they do not know (``stream_options`` on
            # Gemini, for instance). Dropping them beats branching per provider.
            litellm.drop_params = True
            litellm.suppress_debug_info = True
            self._configured = True
        return litellm

    def stream(
        self, messages: Sequence[Message], model: ModelSpec, tools: list[dict[str, object]] | None
    ) -> Iterator[object]:
        litellm = self._litellm()
        kwargs: dict[str, object] = {
            "model": model.litellm_id,
            "messages": list(messages),
            "stream": True,
            # Without this, a streamed response carries no usage at all and the
            # telemetry tab stays empty. It is the most commonly missed flag.
            "stream_options": {"include_usage": True},
        }
        if tools and model.supports_tools:
            kwargs["tools"] = tools
        return litellm.completion(**kwargs)

    def usage(self, chunks: Sequence[object], messages: Sequence[Message], model: ModelSpec) -> Usage:
        litellm = self._litellm()
        try:
            rebuilt = litellm.stream_chunk_builder(list(chunks), messages=list(messages))
            raw = rebuilt.usage
        except Exception:
            return Usage(cost_usd=0.0 if model.is_local else None)

        details = getattr(raw, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)

        if model.is_local:
            cost: float | None = 0.0
        else:
            try:
                cost = float(litellm.completion_cost(completion_response=rebuilt))
            except Exception:
                cost = None  # unknown model id in LiteLLM's price map

        return Usage(
            prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
            cached_tokens=cached,
            cost_usd=cost,
        )


def default_backend() -> ChatBackend:
    return LiteLLMBackend()


def describe_error(error: Exception) -> str:
    """Turn a provider exception into something worth showing a person.

    Every provider fails differently and LiteLLM surfaces the raw upstream body.
    Mapping it once here keeps that noise out of the UI.
    """
    name = type(error).__name__
    detail = str(error).replace("\n", " ")[:180]

    if "RateLimit" in name:
        return (
            "El proveedor cortó por límite de uso (tokens por minuto). "
            f"Esperá unos segundos o probá con otro modelo. — {detail}"
        )
    if "NotFound" in name:
        return f"El proveedor no reconoce este modelo. Puede haber quedado deprecado. — {detail}"
    if "Auth" in name or "PermissionDenied" in name:
        return f"Credenciales rechazadas por el proveedor. Revisá la API key en .env. — {detail}"
    if "Timeout" in name or "APIConnection" in name or "ServiceUnavailable" in name:
        return f"No pude llegar al proveedor. Puede ser la red o un corte del servicio. — {detail}"
    if "ContextWindow" in name or "BadRequest" in name:
        return f"El proveedor rechazó el pedido. — {detail}"
    return f"Falló la llamada al modelo ({name}). — {detail}"
