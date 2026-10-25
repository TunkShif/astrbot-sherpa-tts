"""Command + LLM tool behaviour, without a running AstrBot service."""

from __future__ import annotations

from pathlib import Path

import pytest
from astrbot.api.message_components import Record

from astrbot_plugin_sherpa_tts.main import SherpaTtsPlugin

from .conftest import FakeEngine, RecordingEvent


@pytest.mark.asyncio
async def test_command_yields_record(plugin: SherpaTtsPlugin, event: RecordingEvent):
    results = [result async for result in plugin.tts_command(event)]

    assert len(results) == 1
    chain = results[0].chain
    assert len(chain) == 1
    record = chain[0]
    assert isinstance(record, Record)
    assert record.path and record.path.endswith(".wav")
    # 0.1 s fake audio was actually written and the engine got the text.
    assert plugin.engine.calls[0]["text"] == "你好"
    # The WAV is event-scoped: tracked now, deleted at event cleanup.
    assert record.path in event._temporary_local_files
    event.cleanup_temporary_local_files()
    assert not Path(record.path).exists()


@pytest.mark.asyncio
async def test_command_without_text_returns_usage(
    plugin: SherpaTtsPlugin,
):
    event = RecordingEvent("/tts")
    results = [result async for result in plugin.tts_command(event)]

    assert len(results) == 1
    assert results[0].chain[0].text.startswith("用法")


@pytest.mark.asyncio
async def test_command_rejects_overlong_text(plugin: SherpaTtsPlugin):
    event = RecordingEvent("/tts " + "字" * 101)
    results = [result async for result in plugin.tts_command(event)]

    assert "文本过长" in results[0].chain[0].text
    assert plugin.engine.calls == []


@pytest.mark.asyncio
async def test_llm_tool_sends_voice_and_returns_short_result(
    plugin: SherpaTtsPlugin,
    event: RecordingEvent,
):
    result = await plugin.tts_tool(event, text="你好，世界", speed=0.9)

    assert isinstance(result, str)
    assert "已发送" in result
    assert len(event.sent) == 1
    sent = event.sent[0].chain[0]
    assert isinstance(sent, Record)
    assert plugin.engine.calls[0]["speed"] == 0.9


@pytest.mark.asyncio
async def test_llm_tool_reports_engine_error_as_tool_result(
    plugin: SherpaTtsPlugin,
    event: RecordingEvent,
):
    class BrokenEngine(FakeEngine):
        async def synthesize_async(self, *args, **kwargs):
            from astrbot_plugin_sherpa_tts.tts.engine import TtsError

            raise TtsError("boom")

    plugin.engine = BrokenEngine()

    result = await plugin.tts_tool(event, text="你好")

    assert "失败" in result
    assert event.sent == []


@pytest.mark.asyncio
async def test_llm_tool_without_text_fails_gracefully(
    plugin: SherpaTtsPlugin,
    event: RecordingEvent,
):
    # AstrBot's docstring-based tool schema has no "required" list, so the
    # model may omit text entirely.  It must not raise TypeError.
    result = await plugin.tts_tool(event)

    assert "失败" in result
    assert event.sent == []


@pytest.mark.asyncio
async def test_llm_tool_rejects_non_numeric_speed(
    plugin: SherpaTtsPlugin,
    event: RecordingEvent,
):
    result = await plugin.tts_tool(event, text="你好", speed="fast")

    assert "失败" in result
    assert event.sent == []


@pytest.mark.asyncio
async def test_command_rejects_too_many_sentences(plugin: SherpaTtsPlugin):
    # max_sentences is 3 in the fixture.
    event = RecordingEvent("/tts 一。二。三。四。")
    results = [result async for result in plugin.tts_command(event)]

    assert "句" in results[0].chain[0].text
    assert plugin.engine.calls == []


def test_audio_directory_is_astrbot_temp(plugin: SherpaTtsPlugin):
    from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

    assert (
        plugin.audio_dir == Path(get_astrbot_temp_path()) / "astrbot_plugin_sherpa_tts"
    )
    assert plugin.audio_dir != plugin.data_dir / "audio"


def test_stale_audio_sweep_keeps_recent_files(plugin: SherpaTtsPlugin, tmp_path: Path):
    import os
    import time

    stale = plugin.audio_dir / "stale.wav"
    fresh = plugin.audio_dir / "fresh.wav"
    stale.write_bytes(b"RIFF")
    fresh.write_bytes(b"RIFF")
    old = time.time() - 3600
    os.utime(stale, (old, old))

    plugin._cleanup_stale_audio(older_than_seconds=600)

    assert not stale.exists()
    assert fresh.exists()


@pytest.mark.asyncio
async def test_cooldown_blocks_second_request(
    plugin: SherpaTtsPlugin, event: RecordingEvent
):
    from astrbot_plugin_sherpa_tts.tts.engine import TtsError

    plugin.cooldown_seconds = 5
    await plugin._make_record(event, "你好")

    with pytest.raises(TtsError, match="频繁"):
        await plugin._make_record(event, "再见")


@pytest.mark.asyncio
async def test_admin_bypasses_cooldown(plugin: SherpaTtsPlugin, event: RecordingEvent):
    plugin.cooldown_seconds = 5
    event.role = "admin"

    await plugin._make_record(event, "你好")
    await plugin._make_record(event, "再见")


@pytest.mark.asyncio
async def test_busy_queue_rejects_fast(plugin: SherpaTtsPlugin, event: RecordingEvent):
    from astrbot_plugin_sherpa_tts.tts.engine import TtsError

    plugin.max_queue = 0
    await plugin._infer_semaphore.acquire()
    try:
        with pytest.raises(TtsError, match="繁忙"):
            await plugin._make_record(event, "你好")
    finally:
        plugin._infer_semaphore.release()


@pytest.mark.asyncio
async def test_sequential_requests_release_the_slot(
    plugin: SherpaTtsPlugin, event: RecordingEvent
):
    first = await plugin._make_record(event, "你好")
    second = await plugin._make_record(event, "再见")

    assert first.path != second.path


@pytest.mark.asyncio
async def test_slow_speed_shrinks_the_20s_budget(
    plugin: SherpaTtsPlugin, event: RecordingEvent
):
    from astrbot_plugin_sherpa_tts.tts.engine import TtsError

    with pytest.raises(TtsError, match="最多 50 字"):
        await plugin._make_record(event, "字" * 51, speed=0.5)


@pytest.mark.asyncio
async def test_config_cap_cannot_exceed_20s_budget(
    plugin: SherpaTtsPlugin, event: RecordingEvent
):
    from astrbot_plugin_sherpa_tts.tts.engine import TtsError

    plugin.config["max_total_chars"] = 200
    with pytest.raises(TtsError, match="最多 100 字"):
        await plugin._make_record(event, "字" * 101, speed=1.0)
