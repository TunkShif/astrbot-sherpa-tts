"""Download and extract the Kokoro model on demand.

The plugin never bundles the model.  This module is shared by the plugin's
activation-time downloader and by ``scripts/download_model.py``; it has no
AstrBot imports so it can also be used from a maintenance script.

Only one model archive is downloaded at a time by the plugin.  Partial files
are resumable (``Range``), and extraction happens in a staging directory before
the final model directory is replaced, so a crash cannot leave a half-written
model in place.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import tarfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import aiohttp

DEFAULT_BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    archive: str
    top_dir: str
    model_file: str
    url: str
    sha256: str
    size: int


MODELS: dict[str, ModelSpec] = {
    "fp32": ModelSpec(
        variant="fp32",
        archive="kokoro-multi-lang-v1_1.tar.bz2",
        top_dir="kokoro-multi-lang-v1_1",
        model_file="model.onnx",
        url=f"{DEFAULT_BASE_URL}/kokoro-multi-lang-v1_1.tar.bz2",
        sha256="a3f4c73d043860e3fd2e5b06f36795eb81de0fc8e8de6df703245edddd87dbad",
        size=364_816_464,
    ),
    "int8": ModelSpec(
        variant="int8",
        archive="kokoro-int8-multi-lang-v1_1.tar.bz2",
        top_dir="kokoro-int8-multi-lang-v1_1",
        model_file="model.int8.onnx",
        url=f"{DEFAULT_BASE_URL}/kokoro-int8-multi-lang-v1_1.tar.bz2",
        sha256="a1e94694776049035c4f2c6529f003aaece993c76aae9a78995831c3c4dcafc6",
        size=147_031_220,
    ),
}

_REQUIRED_FILES = (
    "voices.bin",
    "tokens.txt",
    "lexicon-us-en.txt",
    "lexicon-zh.txt",
)


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be downloaded, verified or extracted."""


def spec_for_precision(precision: str) -> ModelSpec:
    """Map the plugin's precision setting to a downloadable model."""
    return MODELS["int8"] if precision == "int8" else MODELS["fp32"]


def is_model_complete(model_dir: str | Path, model_file: str) -> bool:
    """Cheap check used to decide whether a download is needed."""
    model_dir = Path(model_dir)
    if not (model_dir / model_file).is_file():
        return False
    return all((model_dir / name).is_file() for name in _REQUIRED_FILES)


def model_url(spec: ModelSpec, base_url: str | None = None) -> str:
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/{spec.archive}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if target != dest and dest not in target.parents:
                raise ModelDownloadError(f"unsafe archive member: {member.name}")
        tar.extractall(dest, filter="data")


async def _download_resumable(
    session: aiohttp.ClientSession,
    url: str,
    part: Path,
    expected_size: int,
    progress: Callable[[int, int], None] | None,
) -> None:
    offset = await asyncio.to_thread(
        lambda: part.stat().st_size if part.exists() else 0,
    )
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    async with session.get(url, headers=headers) as response:
        if response.status == 416:
            # The partial file is already at/over the end; let SHA-256 verify.
            return
        if response.status not in (200, 206):
            raise ModelDownloadError(f"HTTP {response.status} from {url}")

        if response.status == 200:
            offset = 0
            mode = "wb"
        else:
            mode = "ab"
        content_length = int(response.headers.get("Content-Length") or 0)
        total = offset + content_length if content_length else expected_size
        done = offset
        with part.open(mode) as handle:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, max(total, done))


async def download_model_async(
    dest_dir: str | Path,
    variant: str = "fp32",
    *,
    base_url: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    session: aiohttp.ClientSession | None = None,
) -> Path:
    """Download, verify and extract a model; return the extracted model dir."""
    try:
        spec = MODELS[variant]
    except KeyError as exc:
        raise ModelDownloadError(f"unknown model variant: {variant}") from exc

    dest = Path(dest_dir)
    await asyncio.to_thread(dest.mkdir, parents=True, exist_ok=True)
    final_dir = dest / spec.top_dir
    if is_model_complete(final_dir, spec.model_file):
        return final_dir

    part = dest / f".{spec.archive}.part"
    url = model_url(spec, base_url)
    own_session = session is None
    if own_session:
        timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=120)
        session = aiohttp.ClientSession(timeout=timeout)

    try:
        await _download_resumable(session, url, part, spec.size, progress)

        digest = await asyncio.to_thread(_sha256, part)
        if digest != spec.sha256:
            part.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"sha256 mismatch for {part.name}: expected {spec.sha256}, got {digest}",
            )

        staging = dest / f".extract-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            await asyncio.to_thread(_safe_extract, part, staging)
            source = staging / spec.top_dir
            if not source.is_dir():
                candidates = [item for item in staging.iterdir() if item.is_dir()]
                if len(candidates) != 1:
                    raise ModelDownloadError(
                        f"unexpected archive layout: no {spec.top_dir}/ directory",
                    )
                source = candidates[0]
            if final_dir.exists():
                await asyncio.to_thread(shutil.rmtree, final_dir)
            await asyncio.to_thread(shutil.move, str(source), str(final_dir))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        part.unlink(missing_ok=True)
        if not is_model_complete(final_dir, spec.model_file):
            raise ModelDownloadError(
                f"model files are incomplete after extraction: {final_dir}",
            )
        return final_dir
    finally:
        if own_session:
            await session.close()


def download_model_sync(
    dest_dir: str | Path,
    variant: str = "fp32",
    *,
    base_url: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Blocking wrapper for maintenance scripts."""
    return asyncio.run(
        download_model_async(
            dest_dir,
            variant,
            base_url=base_url,
            progress=progress,
        ),
    )
