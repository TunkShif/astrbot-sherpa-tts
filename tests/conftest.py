"""Pytest bootstrap for offline AstrBot plugin tests.

The important detail: ``ASTRBOT_ROOT`` must be set *before* ``astrbot`` is
imported.  Importing ``astrbot.api`` executes ``astrbot/core/__init__.py``,
which creates ``<root>/data``, a SQLite database, default config and the t2i
template directory.  Pointing it at a throwaway directory keeps the developer
machine and CI runner clean.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="astrbot-tts-test-"))
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("ASTRBOT_TEST_MODE", "true")

import pytest
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.platform.astrbot_message import (
    AstrBotMessage,
    MessageMember,
)
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata

from astrbot_plugin_sherpa_tts.main import SherpaTtsPlugin
from astrbot_plugin_sherpa_tts.tts.audio import write_wav


class RecordingEvent(AstrMessageEvent):
    """A real AstrMessageEvent whose ``send`` records chains instead of I/O."""

    def __init__(self, message_str: str) -> None:
        message = AstrBotMessage()
        message.type = MessageType.GROUP_MESSAGE
        message.self_id = "10000"
        message.session_id = "20000"
        message.message_id = "msg-1"
        message.sender = MessageMember(user_id="12345", nickname="Tester")
        message.message = []
        message.message_str = message_str
        message.raw_message = {}
        super().__init__(
            message_str=message_str,
            message_obj=message,
            platform_meta=PlatformMetadata(
                name="aiocqhttp",
                description="test",
                id="test-aiocqhttp",
            ),
            session_id="20000",
        )
        self.sent: list[MessageChain] = []
        self.is_at_or_wake_command = True

    async def send(self, message: MessageChain) -> None:
        self.sent.append(message)


class FakeEngine:
    """Same async contract as SherpaTtsEngine, but no model/ONNX runtime."""

    model_dir = Path("/nonexistent")
    speed = 1.0
    precision = "auto"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def missing_files(self) -> list[str]:
        return []

    async def synthesize_async(self, text, out_path, *, speaker_id=None, speed=None):
        self.calls.append(
            {"text": text, "speed": speed, "speaker_id": speaker_id, "path": out_path},
        )
        write_wav(out_path, [0.0] * 2400, 24000)  # 0.1 s of silence
        return Path(out_path)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def plugin(engine: FakeEngine, tmp_path: Path, monkeypatch) -> SherpaTtsPlugin:
    instance = SherpaTtsPlugin(
        context=None,  # only used by LLM/tool registration, not by these paths
        config={
            "model_dir": str(tmp_path / "model"),
            "max_total_chars": 100,
            "max_sentences": 3,
            "cooldown_seconds": 0,
            "queue_timeout": 8,
            "max_queue": 2,
        },
    )
    instance.engine = engine
    return instance


@pytest.fixture
def event() -> RecordingEvent:
    return RecordingEvent("/tts 你好")


@pytest.fixture
def wav_factory(tmp_path: Path):
    def make(name: str = "sample.wav") -> Path:
        path = tmp_path / name
        write_wav(path, [0.0] * 1200, 24000)
        return path

    return make


def chain_types(chain: list[BaseMessageComponent]) -> list[str]:
    return [type(component).__name__ for component in chain]
