"""Engine-level tests: text chunking, model wiring, WAV output.

These run without the 150 MB model and without the native ONNX runtime by
installing a fake ``sherpa_onnx`` module into ``sys.modules``.  The fake has
the same constructor shape as the real package, so a typo in the config
plumbing (``data_dir`` vs ``data-dir``, wrong class) still fails the test.
"""

from __future__ import annotations

import sys
import types
import wave
from pathlib import Path
from typing import ClassVar

import pytest

from astrbot_plugin_sherpa_tts.tts import engine as engine_module
from astrbot_plugin_sherpa_tts.tts.audio import write_wav
from astrbot_plugin_sherpa_tts.tts.engine import (
    ModelNotReadyError,
    SherpaTtsEngine,
    TtsError,
)
from astrbot_plugin_sherpa_tts.tts.text import (
    count_sentences,
    normalize_text,
    split_text,
)


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "kokoro"
    (model_dir / "espeak-ng-data").mkdir(parents=True)
    for name in (
        "model.int8.onnx",
        "voices.bin",
        "tokens.txt",
        "lexicon-us-en.txt",
        "lexicon-zh.txt",
    ):
        (model_dir / name).write_bytes(b"stub")
    return model_dir


class FakeOfflineTts:
    instances: ClassVar[list[FakeOfflineTts]] = []
    texts: ClassVar[list[str]] = []

    def __init__(self, config):
        self.config = config
        self.sample_rate = 24000
        self.num_speakers = 103
        FakeOfflineTts.instances.append(self)

    def generate(self, text, sid=0, speed=1.0, callback=None):
        FakeOfflineTts.texts.append(text)
        return types.SimpleNamespace(
            samples=[0.1] * 1200,
            sample_rate=self.sample_rate,
        )


@pytest.fixture
def fake_sherpa(monkeypatch):
    module = types.ModuleType("sherpa_onnx")

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs.get("model")

        def validate(self):
            return True

    class ModelConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.OfflineTtsConfig = Config
    module.OfflineTtsModelConfig = ModelConfig
    module.OfflineTtsKokoroModelConfig = ModelConfig
    module.OfflineTts = FakeOfflineTts
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    FakeOfflineTts.instances.clear()
    FakeOfflineTts.texts.clear()
    return module


def test_split_text_prefers_sentence_boundaries():
    chunks = split_text("第一句。第二句！Third sentence.", max_chars=20)
    assert chunks == ["第一句。", "第二句！", "Third sentence."]


def test_split_text_hard_wraps_long_text():
    chunks = split_text("甲" * 25, max_chars=10)
    assert chunks == ["甲" * 10, "甲" * 10, "甲" * 5]


def test_normalize_text_strips_markdown():
    assert normalize_text("**你好**\n`世界`") == "你好 世界"


def test_missing_model_is_reported_before_import(tmp_path: Path):
    engine = SherpaTtsEngine(tmp_path / "nope")
    with pytest.raises(ModelNotReadyError) as excinfo:
        engine.ensure_ready()
    assert "model.onnx" in str(excinfo.value)


def test_engine_wires_kokoro_config_and_writes_wav(
    tmp_path: Path, fake_sherpa, monkeypatch
):
    monkeypatch.setattr(engine_module, "_sherpa_major_version", lambda: 1)
    model_dir = make_model_dir(tmp_path)
    out = tmp_path / "out.wav"
    engine = SherpaTtsEngine(model_dir, speaker_id=58, num_threads=3)

    engine.synthesize("你好。世界。", out, speed=1.1)

    config = FakeOfflineTts.instances[0].config
    kokoro = config.model.kwargs["kokoro"].kwargs
    assert kokoro["model"].endswith("model.int8.onnx")
    assert kokoro["data_dir"].endswith("espeak-ng-data")
    assert kokoro["lexicon"].count(",") == 1
    assert config.model.kwargs["num_threads"] == 3
    assert FakeOfflineTts.texts == ["你好。", "世界。"]

    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() > 0


def test_engine_rejects_out_of_range_speed(tmp_path: Path, fake_sherpa):
    engine = SherpaTtsEngine(make_model_dir(tmp_path))
    with pytest.raises(TtsError, match="speech|语速"):
        engine.synthesize("你好", tmp_path / "out.wav", speed=3.0)


def test_engine_omits_espeak_data_dir_on_sherpa_v2(
    tmp_path: Path, fake_sherpa, monkeypatch
):
    monkeypatch.setattr(engine_module, "_sherpa_major_version", lambda: 2)
    engine = SherpaTtsEngine(make_model_dir(tmp_path))

    engine.ensure_ready()

    kokoro = FakeOfflineTts.instances[0].config.model.kwargs["kokoro"].kwargs
    assert "data_dir" not in kokoro


def test_engine_prefers_full_model_when_both_exist(tmp_path: Path):
    model_dir = make_model_dir(tmp_path)
    (model_dir / "model.onnx").write_bytes(b"stub")
    engine = SherpaTtsEngine(model_dir)

    assert engine.model_file.name == "model.onnx"


def test_write_wav_sanitizes_nan(tmp_path: Path):
    path = write_wav(tmp_path / "nan.wav", [float("nan"), 0.5, -0.5], 24000)
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())

    samples = [
        int.from_bytes(frames[i : i + 2], "little", signed=True)
        for i in range(0, len(frames), 2)
    ]
    assert samples[0] == 0
    assert samples[1] > 16000
    assert samples[2] < -16000


def test_engine_rejects_out_of_range_speaker(tmp_path: Path, fake_sherpa, monkeypatch):
    monkeypatch.setattr(engine_module, "_sherpa_major_version", lambda: 1)
    engine = SherpaTtsEngine(make_model_dir(tmp_path))
    engine.ensure_ready()

    with pytest.raises(TtsError, match="音色"):
        engine.synthesize("你好", tmp_path / "out.wav", speaker_id=200)


def test_count_sentences_handles_chinese_punctuation():
    assert count_sentences("你好。世界！") == 2
    assert count_sentences("没有标点") == 1
    assert count_sentences("版本 3.14 不是两句。") == 1
    assert count_sentences("Hello world. This is a test!") == 2
    assert count_sentences("Version 3.14.1 is one sentence.") == 1
    assert count_sentences("") == 0


def test_precision_selects_model_file(tmp_path: Path):
    model_dir = make_model_dir(tmp_path)
    (model_dir / "model.onnx").write_bytes(b"stub")

    assert (
        SherpaTtsEngine(model_dir, precision="int8").model_file.name
        == "model.int8.onnx"
    )
    assert SherpaTtsEngine(model_dir, precision="fp32").model_file.name == "model.onnx"
    assert SherpaTtsEngine(model_dir, precision="auto").model_file.name == "model.onnx"
