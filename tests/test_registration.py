"""Verify what AstrBot actually registers for the LLM.

Importing the plugin module runs the ``@filter.llm_tool`` decorator, which
parses the docstring and pushes a ``FuncTool`` into AstrBot's global registry.
This test therefore catches malformed ``Args:`` sections, wrong type names and
duplicate tool names without starting the bot.
"""

from __future__ import annotations

from astrbot.core.provider.register import llm_tools

from .conftest import RecordingEvent


def test_tool_is_registered_with_expected_schema():
    tool = llm_tools.get_func("sherpa_tts")

    assert tool is not None
    assert "语音" in tool.description
    properties = tool.parameters["properties"]
    assert properties["text"]["type"] == "string"
    assert properties["text"]["description"]
    assert properties["speed"]["type"] == "number"


def test_tool_description_mentions_when_to_call():
    tool = llm_tools.get_func("sherpa_tts")
    description = tool.description

    # The model chooses tools from the description alone.
    for keyword in ("语音", "朗读"):
        assert keyword in description


def test_command_is_registered():
    from astrbot.core.star.star_handler import EventType, star_handlers_registry

    handlers = [
        handler
        for handler in star_handlers_registry.get_handlers_by_event_type(
            EventType.AdapterMessageEvent,
        )
        if handler.handler_name == "tts_command"
    ]
    assert handlers, "tts_command was not registered"
    command_filter = handlers[0].event_filters[0]
    assert command_filter.command_name == "tts"
    assert "语音" in command_filter.alias


def test_command_filter_matches_slash_command_and_alias():
    from astrbot.core.star.star_handler import EventType, star_handlers_registry

    handlers = [
        handler
        for handler in star_handlers_registry.get_handlers_by_event_type(
            EventType.AdapterMessageEvent,
        )
        if handler.handler_name == "tts_command"
    ]
    command_filter = handlers[0].event_filters[0]

    # AstrBot's WakingStage strips the "/" wake prefix before the command
    # filter runs, so message_str is "tts 你好" at this point.
    assert command_filter.filter(RecordingEvent("tts 你好"), None)
    assert command_filter.filter(RecordingEvent("语音 你好"), None)
    assert not command_filter.filter(RecordingEvent("ttsx 你好"), None)
    assert not command_filter.filter(RecordingEvent("你好"), None)
