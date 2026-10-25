"""Strongest offline check: load the plugin with AstrBot's real PluginManager.

This exercises metadata.yaml parsing, version-range validation,
``_conf_schema.json`` loading, module import, class discovery, instance
creation, ``initialize()``, decorator registration and handler binding — all
without starting the bot or connecting to NapCat/QQ.

It runs in a subprocess because the plugin manager owns process-global
registries and a SQLite database.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_real_plugin_manager_load_and_dispatch(tmp_path: Path):
    env = os.environ.copy()
    env["ASTRBOT_ROOT"] = str(tmp_path / "astrbot-root")
    env["PLUGIN_REPO_ROOT"] = str(REPO_ROOT)
    env.pop("TESTING", None)

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("plugin_load_probe.py"))],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["load_ok"] is True, payload["load_error"]
    assert payload["metadata_name"] == "astrbot_plugin_sherpa_tts"
    assert payload["metadata_version"] == "v0.1.0"
    assert payload["plugin_class"] == "SherpaTtsPlugin"
    assert payload["command_handlers"] == 1
    assert payload["command_chain"] == ["Record"]
    assert payload["audio_path"].endswith(".wav")
    assert payload["tool_name"] == "sherpa_tts"
    assert payload["tool_bound"] == "tts_tool"
    assert payload["tool_sent"] == ["Record"]
    assert "已发送" in payload["tool_result"]
