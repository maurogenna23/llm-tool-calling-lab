"""Media guardrails. The OpenAI client is stubbed: these tests never spend money."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant import media, tools

PIXEL = base64.b64encode(b"not-really-a-png").decode()


class FakeOpenAI:
    """Records what would have been billed."""

    def __init__(self) -> None:
        self.image_prompts: list[str] = []
        self.spoken: list[str] = []
        self.transcribed: list[str] = []
        outer = self

        class Images:
            def generate(self, **kwargs):  # noqa: ANN003, ANN201
                outer.image_prompts.append(kwargs["prompt"])
                return SimpleNamespace(data=[SimpleNamespace(b64_json=PIXEL)])

        class Speech:
            def create(self, **kwargs):  # noqa: ANN003, ANN201
                outer.spoken.append(kwargs["input"])
                return SimpleNamespace(content=b"mp3-bytes")

        class Transcriptions:
            def create(self, **kwargs):  # noqa: ANN003, ANN201
                outer.transcribed.append(kwargs["model"])
                return SimpleNamespace(text="  quiero una mesa para dos  ")

        self.images = Images()
        self.audio = SimpleNamespace(speech=Speech(), transcriptions=Transcriptions())


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeOpenAI:
    client = FakeOpenAI()
    monkeypatch.setattr(media, "_client", lambda: client)
    monkeypatch.setattr(media, "IMAGE_CACHE_DIR", tmp_path / "images")
    monkeypatch.setattr(media, "IMAGES_ENABLED", True)
    monkeypatch.setattr(media, "media_enabled", lambda: True)
    media.EVENTS.clear()
    return client


# --------------------------------------------------------------------------
# slugs
# --------------------------------------------------------------------------


def test_slug_is_filesystem_safe() -> None:
    assert media.slug("Sorrentinos de calabaza") == "sorrentinos-de-calabaza"
    assert media.slug("Rabas con alioli de limón") == "rabas-con-alioli-de-limon"
    assert media.slug("  Café  de   especialidad ") == "cafe-de-especialidad"


# --------------------------------------------------------------------------
# images -- the only part of the app that costs money per call
# --------------------------------------------------------------------------


def test_image_is_generated_once_and_then_cached(fake_openai: FakeOpenAI, db_path: Path) -> None:
    first, said = media.dish_image("Risotto de hongos", path=db_path)
    assert first is not None and first.exists()
    assert len(fake_openai.image_prompts) == 1
    assert "hongos" in fake_openai.image_prompts[0].lower()
    assert "generada" in said

    second, said_again = media.dish_image("Risotto de hongos", path=db_path)
    assert second == first
    assert len(fake_openai.image_prompts) == 1, "the cached image must not be regenerated"
    assert "ya estaba" in said_again
    assert [event.cached for event in media.EVENTS] == [False, True]


def test_unknown_dish_never_reaches_the_api(fake_openai: FakeOpenAI, db_path: Path) -> None:
    """Bounding images to the menu caps how many distinct ones can ever exist."""
    path, said = media.dish_image("pizza con ananá", path=db_path)
    assert path is None and "no está en la carta" in said
    assert fake_openai.image_prompts == []


def test_images_can_be_switched_off(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(media, "IMAGES_ENABLED", False)
    calls: list[str] = []
    monkeypatch.setattr(media, "_client", lambda: calls.append("boom"))

    path, said = media.dish_image("Risotto de hongos", path=db_path)
    assert path is None and "apagada" in said
    assert calls == []


def test_tool_exposes_the_image_path_to_the_ui(fake_openai: FakeOpenAI, db_path: Path) -> None:
    result = tools.execute("dish_image", {"dish_name": "Flan mixto"}, path=db_path)
    assert result.ok
    assert Path(str(result.payload["image_path"])).exists()


def test_tool_reports_an_unknown_dish_without_failing_the_turn(
    fake_openai: FakeOpenAI, db_path: Path
) -> None:
    result = tools.execute("dish_image", {"dish_name": "sushi"}, path=db_path)
    assert not result.ok and "carta" in result.text
    assert "image_path" not in result.payload


# --------------------------------------------------------------------------
# speech
# --------------------------------------------------------------------------


def test_long_replies_are_trimmed_before_being_read_out(fake_openai: FakeOpenAI) -> None:
    long_text = "palabra " * 400
    path = media.speak(long_text)
    assert path is not None and path.exists()
    spoken = fake_openai.spoken[0]
    assert len(spoken) <= media.MAX_SPEECH_CHARS + 1
    assert spoken.endswith("…")


def test_empty_reply_is_not_sent_to_tts(fake_openai: FakeOpenAI) -> None:
    assert media.speak("   ") is None
    assert fake_openai.spoken == []


def test_transcription_is_trimmed(fake_openai: FakeOpenAI, tmp_path: Path) -> None:
    recording = tmp_path / "clip.wav"
    recording.write_bytes(b"RIFF....")
    assert media.transcribe(recording) == "quiero una mesa para dos"


def test_missing_or_empty_recording_is_ignored(fake_openai: FakeOpenAI, tmp_path: Path) -> None:
    assert media.transcribe(None) == ""
    assert media.transcribe(tmp_path / "nope.wav") == ""

    empty = tmp_path / "empty.wav"
    empty.touch()
    assert media.transcribe(empty) == ""
    assert fake_openai.transcribed == []
