"""The exact OneBot payload AstrBot would hand to NapCat.

AstrBot's aiocqhttp adapter converts ``Record``/``Image`` components to
``base64://`` before calling ``send_group_msg``/``send_private_msg``.  This test
exercises that production conversion path with a real WAV, so it can assert the
final segment type and payload without a live NapCat/QQ connection.
"""

from __future__ import annotations

import base64

import pytest
from astrbot.api.message_components import Record
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)


@pytest.mark.asyncio
async def test_record_becomes_onebot_record_segment(wav_factory):
    record = Record.fromFileSystem(str(wav_factory()))

    segment = await AiocqhttpMessageEvent._from_segment_to_dict(record)

    assert segment["type"] == "record"
    data = segment["data"]["file"]
    assert data.startswith("base64://")

    audio = base64.b64decode(data.removeprefix("base64://"))
    assert audio[:4] == b"RIFF"
    assert audio[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_aiocqhttp_send_calls_send_group_msg_with_record(wav_factory):
    """Go one layer further: the exact kwargs passed to the OneBot action."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from astrbot.api.event import MessageChain
    from astrbot.core.platform.astrbot_message import (
        AstrBotMessage,
        Group,
        MessageMember,
    )
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata

    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "10000"
    message.session_id = "789"
    message.message_id = "m1"
    message.sender = MessageMember(user_id="123", nickname="T")
    message.message = []
    message.message_str = "tts 你好"
    message.raw_message = None
    message.group = Group(group_id="789")

    bot = SimpleNamespace(send_group_msg=AsyncMock())
    event = AiocqhttpMessageEvent(
        message_str="tts 你好",
        message_obj=message,
        platform_meta=PlatformMetadata(name="aiocqhttp", description="t", id="t"),
        session_id="789",
        bot=bot,
    )

    await event.send(MessageChain([Record.fromFileSystem(str(wav_factory()))]))

    bot.send_group_msg.assert_awaited_once()
    kwargs = bot.send_group_msg.await_args.kwargs
    assert kwargs["group_id"] == 789
    segment = kwargs["message"][0]
    assert segment["type"] == "record"
    assert segment["data"]["file"].startswith("base64://")
