"""AstrBot entrypoint for the local sherpa-onnx TTS plugin.

Two ways to trigger speech:

* ``/tts <text>`` (aliases ``/语音``, ``/朗读``) — explicit slash command.
* the ``sherpa_tts`` LLM tool — the model decides to call it when the user
  asks for a voice reply.

Both paths produce a ``Record`` component.  AstrBot's ``aiocqhttp`` adapter
turns ``Record`` into a OneBot ``record`` segment (base64 WAV) and NapCat
transcodes it to QQ's silk format before uploading.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Record
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .model_download import download_model_async, spec_for_precision
from .tts.engine import ModelNotReadyError, SherpaTtsEngine, TtsError, new_audio_path
from .tts.text import count_sentences

PLUGIN_NAME = "astrbot_plugin_sherpa_tts"
# Hard latency budget: ~20s of generation on the reference AMD Ryzen 7 4800H.
# Kokoro speaks ~5.7 Chinese chars/second at speed=1.0.  With a conservative
# RTF of 1.0, 100 chars ≈ 17.5s of audio ≈ 17.5s of generation, leaving
# headroom for model warm-up and transport.
MAX_GENERATION_SECONDS = 20.0
HARD_CHAR_BUDGET = 100
MODEL_SUBDIRS = (
    Path("models") / "kokoro-multi-lang-v1_1",
    Path("models") / "kokoro-int8-multi-lang-v1_1",
)


class SherpaTtsPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        # Generated WAVs are event-scoped temp files, not plugin data.
        self.audio_dir = Path(get_astrbot_temp_path()) / PLUGIN_NAME
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        model_dir = self.config.get("model_dir") or str(self._default_model_dir())
        self._infer_semaphore = asyncio.Semaphore(1)
        self._waiting = 0
        self._last_request: dict[str, float] = {}
        self.queue_timeout = max(0.0, float(self.config.get("queue_timeout", 8)))
        self.max_queue = max(0, int(self.config.get("max_queue", 2)))
        self.cooldown_seconds = max(
            0.0,
            float(self.config.get("cooldown_seconds", 5)),
        )
        self.admin_bypass = bool(self.config.get("admin_bypass", True))

        self._auto_download = bool(self.config.get("auto_download_model", True))
        self._download_base_url = str(
            self.config.get("model_download_base_url", "") or "",
        )
        self._custom_model_dir = bool(self.config.get("model_dir"))
        self._download_task: asyncio.Task[None] | None = None
        self._download_progress = 0.0
        self._download_status = "idle"
        self._download_error = ""
        self._download_attempt_at = 0.0
        self._last_download_log_pct = -1

        self.engine = SherpaTtsEngine(
            model_dir,
            speaker_id=int(self.config.get("speaker_id", 3)),
            speed=float(self.config.get("speed", 1.0)),
            num_threads=int(self.config.get("num_threads", 2)),
            debug=bool(self.config.get("debug", False)),
            max_chars=int(self.config.get("max_chars", 160)),
            precision=str(self.config.get("model_precision", "auto")),
        )

    def _default_model_dir(self) -> Path:
        """Prefer the full model, fall back to an int8 extraction."""
        for subdir in MODEL_SUBDIRS:
            candidate = self.data_dir / subdir
            if candidate.exists():
                return candidate
        return self.data_dir / MODEL_SUBDIRS[0]

    # ------------------------------------------------------------- lifecycle
    async def initialize(self) -> None:
        """Optionally warm the model up in the background.

        Lazy loading is the default: AstrBot startup stays fast and users who
        never use TTS do not pay the ~300 MB model load cost.
        """
        self._cleanup_stale_audio()
        self._maybe_start_model_download()

        if not self.config.get("auto_load", False):
            logger.info("[sherpa-tts] 模型将按需加载: %s", self.engine.model_dir)
            return
        if self._download_task and not self._download_task.done():
            logger.info("[sherpa-tts] 模型下载完成后自动加载")
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.engine.ensure_ready),
                timeout=float(self.config.get("load_timeout", 180)),
            )
            logger.info("[sherpa-tts] 模型加载完成: %s", self.engine.model_dir)
        except (TimeoutError, TtsError) as exc:
            logger.warning("[sherpa-tts] 模型预加载失败（首次使用时重试）: %s", exc)

    async def terminate(self) -> None:
        task = self._download_task
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self.engine.close()

    # ------------------------------------------------------- model download
    def _maybe_start_model_download(self) -> None:
        """Start one background download if the default model is missing."""
        if not self._auto_download or self._custom_model_dir:
            return
        if self._download_task and not self._download_task.done():
            return
        if not self.engine.missing_files():
            return
        self._download_task = asyncio.create_task(self._download_model())

    def _on_download_progress(self, done: int, total: int) -> None:
        self._download_progress = done / total if total else 0.0
        percent = int(self._download_progress * 100)
        if percent >= self._last_download_log_pct + 10:
            logger.info(
                "[sherpa-tts] 模型下载 %d%%（%d/%d MiB）",
                percent,
                done // 2**20,
                total // 2**20,
            )
            self._last_download_log_pct = percent

    async def _download_model(self) -> None:
        spec = spec_for_precision(self.engine.precision)
        self._download_status = "downloading"
        self._download_error = ""
        self._download_progress = 0.0
        self._download_attempt_at = time.monotonic()
        self._last_download_log_pct = -1
        logger.info(
            "[sherpa-tts] 模型缺失，开始后台下载 %s（%d MiB）",
            spec.variant,
            spec.size // 2**20,
        )
        try:
            path = await download_model_async(
                self.data_dir / "models",
                spec.variant,
                base_url=self._download_base_url or None,
                progress=self._on_download_progress,
            )
            self.engine.model_dir = path
            self._download_status = "ready"
            self._download_progress = 1.0
            logger.info("[sherpa-tts] 模型下载完成: %s", path)
            if self.config.get("auto_load", False):
                try:
                    await asyncio.to_thread(self.engine.ensure_ready)
                except TtsError as exc:
                    # The download succeeded; a warm-up failure should not
                    # make the model look missing.  First TTS retries loading.
                    logger.warning("[sherpa-tts] 下载完成但预加载失败: %s", exc)
                else:
                    logger.info("[sherpa-tts] 模型加载完成: %s", path)
        except asyncio.CancelledError:
            self._download_status = "cancelled"
            logger.info("[sherpa-tts] 模型下载已取消，下次启动会续传")
            raise
        except Exception as exc:
            # aiohttp errors, disk errors, extraction errors: never let the
            # background task die silently or leave status stuck on downloading.
            self._download_status = "failed"
            self._download_error = str(exc)
            logger.error("[sherpa-tts] 模型下载失败: %s", exc)

    def _ensure_model_available(self) -> None:
        missing = self.engine.missing_files()
        if not missing:
            return
        if self._download_task and not self._download_task.done():
            raise TtsError(
                f"模型正在下载（{self._download_progress * 100:.0f}%），请稍后再试。",
            )
        if self._auto_download and not self._custom_model_dir:
            elapsed = time.monotonic() - self._download_attempt_at
            if self._download_status == "failed" and elapsed < 60:
                raise TtsError(
                    f"模型下载失败：{self._download_error}；"
                    f"约 {int(60 - elapsed) + 1} 秒后重试。",
                )
            self._maybe_start_model_download()
            raise TtsError("模型尚未就绪，已开始后台下载，请稍后再试。")
        raise TtsError(
            f"模型文件不完整：{', '.join(missing)}。请下载模型或检查 model_dir 设置。",
        )

    # ---------------------------------------------------------------- helpers
    def _command_text(self, event: AstrMessageEvent) -> str:
        """Extract everything after the command word.

        ``event.message_str`` still contains the command prefix, so a plain
        ``text: str`` handler parameter would only receive the first token.
        """
        parts = event.get_message_str().strip().split(maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""

    def _resolve_speed(self, speed: float | None) -> float:
        value = self.engine.speed if speed is None else float(speed)
        if not 0.5 <= value <= 2.0:
            raise TtsError("语速 speed 必须在 0.5 到 2.0 之间")
        return value

    def _char_limit(self, speed: float) -> int:
        """Chars allowed by the configured cap and the 20s hard budget.

        Slower speech produces longer audio, so the budget shrinks with speed;
        faster speech may allow more chars, but the user config can only lower
        the result, never exceed the 20s target.
        """
        configured = max(1, int(self.config.get("max_total_chars", HARD_CHAR_BUDGET)))
        target = int(HARD_CHAR_BUDGET * min(2.0, max(0.5, speed)))
        return max(1, min(configured, target))

    def _check_text(self, text: str, speed: float = 1.0) -> str:
        text = (text or "").strip()
        if not text:
            raise TtsError("文本为空")

        limit = self._char_limit(speed)
        if len(text) > limit:
            raise TtsError(
                f"文本过长（{len(text)} 字），当前语速下最多 {limit} 字"
                f"（目标 ≤{MAX_GENERATION_SECONDS:g} 秒语音）。",
            )

        max_sentences = max(1, int(self.config.get("max_sentences", 3)))
        sentences = count_sentences(text)
        if sentences > max_sentences:
            raise TtsError(
                f"文本包含 {sentences} 句，超过 {max_sentences} 句上限；请缩短后再试。",
            )
        return text

    def _check_cooldown(self, event: AstrMessageEvent) -> None:
        """Per-user cooldown; admins can bypass it."""
        if self.cooldown_seconds <= 0:
            return
        if self.admin_bypass and event.role == "admin":
            return
        key = event.get_sender_id()
        now = time.monotonic()
        last = self._last_request.get(key)
        if last is not None:
            remaining = self.cooldown_seconds - (now - last)
            if remaining > 0:
                raise TtsError(
                    f"请求太频繁，请 {int(remaining) + 1} 秒后再试。",
                )
        self._last_request[key] = now
        if len(self._last_request) > 1000:
            cutoff = now - 3600
            self._last_request = {
                user: timestamp
                for user, timestamp in self._last_request.items()
                if timestamp >= cutoff
            }

    async def _acquire_infer_slot(self) -> None:
        """Bounded wait for the single CPU inference slot.

        This is intentionally not a job system: TTS is a short local CPU job.
        Requests either get the slot quickly or fail fast with a readable
        "busy" message, instead of piling up and hitting the LLM tool timeout.
        """
        if self._infer_semaphore.locked() and (
            self.max_queue <= 0 or self._waiting >= self.max_queue
        ):
            raise TtsError("语音合成繁忙，请稍后再试。")
        self._waiting += 1
        try:
            if self.queue_timeout > 0:
                await asyncio.wait_for(
                    self._infer_semaphore.acquire(),
                    timeout=self.queue_timeout,
                )
            else:
                if self._infer_semaphore.locked():
                    raise TtsError("语音合成繁忙，请稍后再试。")
                await self._infer_semaphore.acquire()
        except TimeoutError as exc:
            raise TtsError(
                f"语音合成繁忙（等待超过 {self.queue_timeout:g} 秒），请稍后再试。",
            ) from exc
        finally:
            self._waiting -= 1

    def _cleanup_stale_audio(self, older_than_seconds: float = 600) -> None:
        """Remove WAVs left behind by a crash/reload.

        Normal cleanup is handled by AstrBot's event-scoped
        ``cleanup_temporary_local_files()``; this only sweeps stale files.
        """
        cutoff = time.time() - older_than_seconds
        for path in self.audio_dir.glob("*.wav"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    async def _make_record(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        speed: float | None = None,
    ) -> Record:
        try:
            speed = self._resolve_speed(speed)
        except (TypeError, ValueError) as exc:
            raise TtsError("语速 speed 必须是数字") from exc
        text = self._check_text(text, speed)
        self._ensure_model_available()
        self._check_cooldown(event)
        await self._acquire_infer_slot()
        try:
            out_path = new_audio_path(self.audio_dir)
            try:
                await self.engine.synthesize_async(text, out_path, speed=speed)
            except ModelNotReadyError as exc:
                out_path.unlink(missing_ok=True)
                raise TtsError(
                    f"{exc}\n提示：请把模型放到插件数据目录，或在配置中设置 model_dir。",
                ) from exc
            except BaseException:
                out_path.unlink(missing_ok=True)
                raise
            # AstrBot deletes tracked temp files when the event finishes, after
            # the platform adapter has read the WAV for base64 encoding.
            event.track_temporary_local_file(str(out_path))
            return Record.fromFileSystem(str(out_path))
        finally:
            self._infer_semaphore.release()

    # ---------------------------------------------------------------- command
    @filter.command("tts", alias={"语音", "朗读"})
    async def tts_command(self, event: AstrMessageEvent):
        """将文本合成为语音。用法：/tts <文本>"""
        text = self._command_text(event)
        if not text:
            yield event.plain_result(
                "用法：/tts <要朗读的文本>\n例如：/tts 你好，这是一段本地语音合成。",
            )
            return
        try:
            record = await self._make_record(event, text)
        except TtsError as exc:
            logger.warning("[sherpa-tts] 指令合成失败: %s", exc)
            yield event.plain_result(f"语音合成失败：{exc}")
            return
        yield event.chain_result([record])

    # --------------------------------------------------------------- LLM tool
    @filter.llm_tool(name="sherpa_tts")
    async def tts_tool(
        self,
        event: AstrMessageEvent,
        text: str = "",
        speed: float = 1.0,
    ):
        """将一段文本合成为语音并直接发送给用户。

        当用户明确要求“用语音说”“朗读”“发语音”“念出来”，或你判断语音比文字更合适时调用。
        只传入适合朗读的纯文本：不要包含 Markdown、代码块、链接或表情符号。
        最多 3 句、约 100 字；超过会被拒绝，请只朗读用户要求的那部分。

        Args:
            text(string): 要朗读的文本，最多 3 句、约 100 字。
            speed(number): 语速，0.5 到 2.0，默认 1.0。
        """
        try:
            record = await self._make_record(event, text, speed=speed)
        except TtsError as exc:
            logger.warning("[sherpa-tts] LLM 工具合成失败: %s", exc)
            return f"语音合成失败：{exc}"
        await event.send(event.chain_result([record]))
        return "语音已发送给用户。"
