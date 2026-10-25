"""Standalone probe executed in a subprocess by ``test_plugin_load.py``.

Why a subprocess: AstrBot's plugin manager mutates process-global registries
(``star_map``, ``star_handlers_registry``, ``llm_tools``), creates a SQLite DB
and writes into ``ASTRBOT_ROOT``.  A subprocess gives each probe a genuinely
clean "bot process" without a server, WebSocket or message platform.

Prints one JSON line with everything the parent test asserts on.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(os.environ["PLUGIN_REPO_ROOT"]).resolve()
ROOT = Path(tempfile.mkdtemp(prefix="astrbot-load-probe-"))
os.environ["ASTRBOT_ROOT"] = str(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def write_tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 1200)


async def main() -> dict:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.core.config.astrbot_config import AstrBotConfig
    from astrbot.core.db.sqlite import SQLiteDatabase
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.provider.func_tool_manager import FunctionToolManager
    from astrbot.core.provider.register import llm_tools
    from astrbot.core.star.context import Context
    from astrbot.core.star.star import star_map
    from astrbot.core.star.star_handler import EventType, star_handlers_registry
    from astrbot.core.star.star_manager import PluginManager

    plugins = ROOT / "data" / "plugins"
    plugins.mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "astrbot_plugin_sherpa_tts",
        plugins / "astrbot_plugin_sherpa_tts",
    )

    config = AstrBotConfig()
    config._config_path = str(ROOT / "data" / "cmd_config.json")
    db = SQLiteDatabase(str(ROOT / "data" / "probe.db"))
    provider_manager = MagicMock()
    provider_manager.llm_tools = FunctionToolManager()
    context = Context(
        asyncio.Queue(),
        config,
        db,
        provider_manager,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(personas_v3=[]),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        None,
    )

    manager = PluginManager(context, config)
    ok, error = await manager.load(specified_dir_name="astrbot_plugin_sherpa_tts")
    metadata = next(
        meta
        for path, meta in star_map.items()
        if path.endswith(".main") and "sherpa" in path
    )

    class Event(AstrMessageEvent):
        def __init__(self, text: str):
            message = AstrBotMessage()
            message.type = MessageType.GROUP_MESSAGE
            message.self_id = "10000"
            message.session_id = "20000"
            message.message_id = "m1"
            message.sender = MessageMember(user_id="123", nickname="T")
            message.message = []
            message.message_str = text
            message.raw_message = {}
            super().__init__(
                text,
                message,
                PlatformMetadata(name="aiocqhttp", description="t", id="t"),
                "20000",
            )
            self.sent = []

        async def send(self, message):
            self.sent.append(message)

    class Engine:
        speed = 1.0

        async def synthesize_async(
            self, text, out_path, *, speaker_id=None, speed=None
        ):
            write_tiny_wav(Path(out_path))
            return out_path

    metadata.star_cls.engine = Engine()  # avoid loading the real 150 MB model
    metadata.star_cls.cooldown_seconds = 0  # command + tool run back-to-back

    handlers = [
        handler
        for handler in star_handlers_registry.get_handlers_by_event_type(
            EventType.AdapterMessageEvent,
        )
        if handler.handler_name == "tts_command"
    ]
    event = Event("/tts 你好，插件加载测试")
    results = [result async for result in handlers[0].handler(event)]

    tool = llm_tools.get_func("sherpa_tts")
    tool_event = Event("hi")
    tool_result = await tool.handler(tool_event, text="你好", speed=1.0)

    return {
        "load_ok": ok,
        "load_error": error,
        "metadata_name": metadata.name,
        "metadata_version": metadata.version,
        "plugin_class": type(metadata.star_cls).__name__,
        "command_handlers": len(handlers),
        "command_chain": [type(c).__name__ for c in results[0].chain],
        "audio_path": str(results[0].chain[0].path),
        "tool_name": tool.name,
        "tool_bound": tool.handler.func.__name__,
        "tool_result": tool_result,
        "tool_sent": [type(c).__name__ for c in tool_event.sent[0].chain],
    }


print(json.dumps(asyncio.run(main())))
