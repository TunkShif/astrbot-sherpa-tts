"""Small WAV writer used by the TTS engine.

``sherpa_onnx`` returns float32 samples in [-1, 1] at the model sample rate
(24000 Hz for Kokoro).  ``wave`` + :mod:`array` is enough, but numpy is
already a hard dependency of sherpa-onnx and makes the conversion much faster,
so use it when present.
"""

from __future__ import annotations

import wave
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _pcm16le(samples: Iterable[float]) -> bytes:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - sherpa-onnx always ships numpy
        pcm = array("h")
        for sample in samples:
            value = int(max(-1.0, min(1.0, float(sample))) * 32767)
            pcm.append(value)
        return pcm.tobytes()

    data = np.asarray(samples, dtype="float32")
    # Kokoro int8 + some OOV English words can emit NaN frames (unknown
    # phoneme).  Never let those reach the PCM cast: a NaN becomes INT16_MIN
    # and produces a loud click.
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype("<i2").tobytes()


def write_wav(path: str | Path, samples: Any, sample_rate: int) -> Path:
    """Write mono 16-bit PCM WAV and return the resolved path."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(_pcm16le(samples))
    return path.resolve()


def concat_silence(sample_rate: int, seconds: float = 0.15) -> bytes:
    """Return ``seconds`` of 16-bit mono silence (for chunk joins)."""
    return b"\x00\x00" * max(0, int(sample_rate * seconds))
