"""Optional end-to-end synthesis test with the real Kokoro model.

Skipped unless ``SHERPA_TTS_MODEL_DIR`` points at an extracted model, because
the archive is ~150-365 MB:

    SHERPA_TTS_MODEL_DIR=$PWD/.research/models/kokoro-multi-lang-v1_1 \
        pytest -m model

This is the layer that verifies native sherpa-onnx + ONNX Runtime + model
files + WAV writing.  It still does not need AstrBot's server.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from astrbot_plugin_sherpa_tts.tts.engine import SherpaTtsEngine

MODEL_DIR = os.environ.get("SHERPA_TTS_MODEL_DIR")

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not MODEL_DIR, reason="SHERPA_TTS_MODEL_DIR not set"),
]


@pytest.mark.asyncio
async def test_real_model_synthesizes_chinese_audio(tmp_path: Path):
    engine = SherpaTtsEngine(MODEL_DIR, speaker_id=3, num_threads=2)
    out = tmp_path / "real.wav"

    await engine.synthesize_async("你好，世界。这是一段真实的本地语音合成。", out)

    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        duration = wav.getnframes() / wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    assert duration > 0.5
    assert any(frames)
