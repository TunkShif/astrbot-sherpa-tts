"""Thin, testable wrapper around ``sherpa_onnx.OfflineTts``.

The module deliberately imports :mod:`sherpa_onnx` lazily.  AstrBot users may
install the plugin before downloading the ~150 MB model, and the plugin's own
unit tests must be able to import the package on a machine without
sherpa-onnx or the model.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import threading
import time
import uuid
from pathlib import Path

from .audio import write_wav
from .text import split_text

LOG = logging.getLogger(__name__)


class TtsError(RuntimeError):
    """User-facing TTS failure (bad config, missing model, generation error)."""


class ModelNotReadyError(TtsError):
    """The model directory is incomplete or has not been downloaded yet."""


def _sherpa_major_version() -> int:
    """Return the installed sherpa-onnx major version (1 if unknown).

    v2.0.0 removed espeak-ng/piper-phonemize; ``kokoro.data_dir`` is ignored
    and Kokoro becomes lexicon-only.  v1.x still needs ``espeak-ng-data`` for
    OOV English words.
    """
    for package in ("sherpa-onnx", "sherpa-onnx-core"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        try:
            return int(version.split(".")[0])
        except ValueError:
            return 1
    return 1


class SherpaTtsEngine:
    """Lazy-loaded Kokoro (sherpa-onnx) TTS engine.

    Args:
        model_dir: Directory containing ``model.onnx`` / ``model.int8.onnx``,
            ``voices.bin``, ``tokens.txt``, ``espeak-ng-data`` and the lexicons.
        speaker_id: Default Kokoro speaker id (0-102 for v1.1). 3 = ``zf_001``.
        speed: Default speaking speed (1.0 = normal).
        num_threads: ONNX Runtime intra-op threads. CPU inference only.
        debug: Forwarded to sherpa-onnx (prints model metadata when true).
        max_chars: Maximum characters per synthesis chunk.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        speaker_id: int = 3,
        speed: float = 1.0,
        num_threads: int = 2,
        debug: bool = False,
        max_chars: int = 160,
        precision: str = "auto",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.speaker_id = int(speaker_id)
        self.speed = float(speed)
        self.num_threads = max(1, int(num_threads))
        self.debug = bool(debug)
        self.max_chars = int(max_chars)
        self.precision = precision if precision in {"auto", "fp32", "int8"} else "auto"

        self._tts = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------ files
    @property
    def model_file(self) -> Path:
        """Resolve the configured precision to an on-disk model file.

        ``auto`` prefers the full model: it is more robust for mixed Chinese /
        English and, on the tested Apple Silicon host, faster than int8.  x86
        hosts may prefer ``int8``; expose that as an explicit setting.
        """
        full = self.model_dir / "model.onnx"
        int8 = self.model_dir / "model.int8.onnx"
        if self.precision == "int8":
            return int8
        if self.precision == "fp32":
            return full
        return full if full.exists() else int8

    def missing_files(self) -> list[str]:
        """Return model files/directories that are absent."""
        required = [
            self.model_dir / "voices.bin",
            self.model_dir / "tokens.txt",
            self.model_dir / "lexicon-us-en.txt",
            self.model_dir / "lexicon-zh.txt",
        ]
        # sherpa-onnx >= 2.0 no longer uses espeak-ng; it is only needed by
        # v1.x for OOV English words.
        if _sherpa_major_version() < 2:
            required.append(self.model_dir / "espeak-ng-data")
        missing = [str(path.name) for path in required if not path.exists()]
        if not self.model_file.exists():
            label = (
                "model.onnx / model.int8.onnx"
                if self.precision == "auto"
                else self.model_file.name
            )
            missing.insert(0, label)
        return missing

    @property
    def ready(self) -> bool:
        return self._tts is not None

    # ------------------------------------------------------------------- load
    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        with self._load_lock:
            if self._tts is not None:
                return
            missing = self.missing_files()
            if missing:
                raise ModelNotReadyError(
                    f"模型文件不完整（缺少: {', '.join(missing)}），"
                    f"请检查 model_dir: {self.model_dir}",
                )

            try:
                import sherpa_onnx
            except ImportError as exc:  # pragma: no cover - depends on env
                raise TtsError(
                    "未安装 sherpa-onnx，请先安装插件依赖（pip install sherpa-onnx）",
                ) from exc

            lexicons = [
                str(self.model_dir / "lexicon-us-en.txt"),
                str(self.model_dir / "lexicon-zh.txt"),
            ]
            rule_fsts = [
                str(path)
                for path in (
                    self.model_dir / "date-zh.fst",
                    self.model_dir / "number-zh.fst",
                    self.model_dir / "phone-zh.fst",
                )
                if path.exists()
            ]

            major = _sherpa_major_version()
            kokoro_kwargs = {
                "model": str(self.model_file),
                "voices": str(self.model_dir / "voices.bin"),
                "tokens": str(self.model_dir / "tokens.txt"),
                "lexicon": ",".join(lexicons),
            }
            if major < 2:
                # v1.x uses espeak-ng/piper-phonemize for OOV English words.
                kokoro_kwargs["data_dir"] = str(self.model_dir / "espeak-ng-data")
            else:
                LOG.info(
                    "sherpa-onnx >= 2.0: espeak-ng is disabled; "
                    "English words outside the bundled lexicons are skipped.",
                )

            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(**kokoro_kwargs),
                    num_threads=self.num_threads,
                    debug=self.debug,
                    provider="cpu",
                ),
                rule_fsts=",".join(rule_fsts),
                max_num_sentences=1,
                silence_scale=0.2,
            )
            if not config.validate():
                raise TtsError(
                    f"sherpa-onnx 拒绝了模型配置，请检查文件是否完整: {self.model_dir}",
                )

            try:
                self._tts = sherpa_onnx.OfflineTts(config)
            except Exception as exc:
                raise TtsError(f"加载 TTS 模型失败: {exc}") from exc

    def ensure_ready(self) -> None:
        """Load the model if needed (blocking)."""
        self._ensure_loaded()

    def close(self) -> None:
        """Release the native model (called from ``Star.terminate``)."""
        self._tts = None

    # --------------------------------------------------------------- generate
    def _generate_samples(
        self,
        chunks: list[str],
        *,
        speaker_id: int,
        speed: float,
    ):
        import numpy as np

        pieces = []
        sample_rate = 0
        for index, chunk in enumerate(chunks):
            audio = self._tts.generate(chunk, sid=speaker_id, speed=speed)
            samples = np.asarray(audio.samples, dtype="float32")
            if samples.size == 0:
                LOG.warning("TTS produced no samples for chunk: %r", chunk[:40])
                continue
            if not np.isfinite(samples).all():
                # Known sherpa-onnx 1.x + int8 issue: an unknown phoneme (e.g.
                # ɚ from an OOV English word) yields NaN frames.  Zero them out
                # instead of writing INT16_MIN clicks; the rest stays audible.
                LOG.warning(
                    "TTS produced non-finite samples for chunk %r; sanitising",
                    chunk[:40],
                )
                samples = np.nan_to_num(samples, copy=False, nan=0.0)
                if not np.any(samples):
                    continue
            sample_rate = int(audio.sample_rate)
            pieces.append(samples)
            if index != len(chunks) - 1:
                # Short pause between sentences, consistent with
                # ``silence_scale=0.2``.
                pieces.append(np.zeros(int(sample_rate * 0.12), dtype="float32"))
        if not pieces:
            raise TtsError("模型没有生成任何音频，请检查文本内容")
        return np.concatenate(pieces), sample_rate

    def synthesize(
        self,
        text: str,
        out_path: str | Path,
        *,
        speaker_id: int | None = None,
        speed: float | None = None,
    ) -> Path:
        """Synthesise ``text`` into a 16-bit mono WAV (blocking)."""
        chunks = split_text(text, max_chars=self.max_chars)
        if not chunks:
            raise TtsError("文本为空，无法合成语音")

        self._ensure_loaded()
        speaker = self.speaker_id if speaker_id is None else int(speaker_id)
        num_speakers = int(getattr(self._tts, "num_speakers", 0))
        if num_speakers and not 0 <= speaker < num_speakers:
            raise TtsError(
                f"音色 ID {speaker} 超出范围（0-{num_speakers - 1}）",
            )
        rate = self.speed if speed is None else float(speed)
        if not 0.5 <= rate <= 2.0:
            raise TtsError("语速 speed 必须在 0.5 到 2.0 之间")

        with self._infer_lock:
            try:
                samples, sample_rate = self._generate_samples(
                    chunks,
                    speaker_id=speaker,
                    speed=rate,
                )
            except TtsError:
                raise
            except Exception as exc:
                raise TtsError(f"语音合成失败: {exc}") from exc

        return write_wav(out_path, samples, sample_rate)

    async def synthesize_async(
        self,
        text: str,
        out_path: str | Path,
        *,
        speaker_id: int | None = None,
        speed: float | None = None,
    ) -> Path:
        """Run :meth:`synthesize` in a worker thread (never block the bot)."""
        return await asyncio.to_thread(
            self.synthesize,
            text,
            out_path,
            speaker_id=speaker_id,
            speed=speed,
        )


def new_audio_path(directory: str | Path, prefix: str = "tts") -> Path:
    """Return a unique WAV path inside ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    return directory / f"{prefix}-{int(time.time() * 1000)}-{token}.wav"
