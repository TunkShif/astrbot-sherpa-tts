"""Unit tests for model selection, resumable download and safe extraction."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
from astrbot_plugin_sherpa_tts import model_download
from astrbot_plugin_sherpa_tts.model_download import (
    MODELS,
    ModelDownloadError,
    ModelSpec,
    _safe_extract,
    download_model_async,
    is_model_complete,
    model_url,
    spec_for_precision,
)

REQUIRED = {
    "voices.bin": b"voices",
    "tokens.txt": b"tokens",
    "lexicon-us-en.txt": b"lexicon-en",
    "lexicon-zh.txt": b"lexicon-zh",
    "model.onnx": b"model",
}


def make_archive(tmp_path: Path, top_dir: str = "fake-model") -> bytes:
    archive = tmp_path / "fake.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        for name, data in REQUIRED.items():
            info = tarfile.TarInfo(f"{top_dir}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive.read_bytes()


def make_spec(data: bytes, *, sha: str | None = None) -> ModelSpec:
    return ModelSpec(
        variant="test",
        archive="fake.tar.bz2",
        top_dir="fake-model",
        model_file="model.onnx",
        url="https://example.invalid/fake.tar.bz2",
        sha256=sha or hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


class FakeContent:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def iter_chunked(self, _size: int):
        for index in range(0, len(self.data), 7):
            yield self.data[index : index + 7]


class FakeResponse:
    def __init__(self, data: bytes, status: int) -> None:
        self.status = status
        self.headers = {"Content-Length": str(len(data))}
        self.content = FakeContent(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeSession:
    def __init__(self, data: bytes, status: int = 200) -> None:
        self.data = data
        self.status = status
        self.headers: list[dict | None] = []

    def get(self, _url: str, headers: dict | None = None):
        self.headers.append(headers)
        return FakeResponse(self.data, self.status)


def test_precision_maps_to_download_variant():
    assert spec_for_precision("auto").variant == "fp32"
    assert spec_for_precision("fp32").variant == "fp32"
    assert spec_for_precision("int8").variant == "int8"


def test_model_specs_have_checksums_and_expected_files():
    assert MODELS["fp32"].model_file == "model.onnx"
    assert MODELS["int8"].model_file == "model.int8.onnx"
    assert len(MODELS["fp32"].sha256) == 64
    assert len(MODELS["int8"].sha256) == 64
    assert MODELS["int8"].size < MODELS["fp32"].size


def test_model_url_uses_mirror_base():
    spec = MODELS["int8"]
    assert model_url(spec, "https://mirror.example/tts/").endswith(
        "/kokoro-int8-multi-lang-v1_1.tar.bz2",
    )


def test_is_model_complete_requires_core_files(tmp_path: Path):
    model_dir = tmp_path / "kokoro"
    model_dir.mkdir()
    for name, data in REQUIRED.items():
        (model_dir / name).write_bytes(data)

    assert is_model_complete(model_dir, "model.onnx")
    (model_dir / "tokens.txt").unlink()
    assert not is_model_complete(model_dir, "model.onnx")


@pytest.mark.asyncio
async def test_download_verifies_extracts_and_reports_progress(
    tmp_path: Path, monkeypatch
):
    data = make_archive(tmp_path)
    spec = make_spec(data)
    monkeypatch.setitem(model_download.MODELS, "test", spec)
    session = FakeSession(data)
    progress: list[tuple[int, int]] = []

    result = await download_model_async(
        tmp_path / "dest",
        "test",
        session=session,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert result == tmp_path / "dest" / "fake-model"
    assert is_model_complete(result, "model.onnx")
    assert progress[-1] == (len(data), len(data))
    assert not (tmp_path / "dest" / ".fake.tar.bz2.part").exists()


@pytest.mark.asyncio
async def test_download_rejects_bad_sha(tmp_path: Path, monkeypatch):
    data = make_archive(tmp_path)
    spec = make_spec(data, sha="0" * 64)
    monkeypatch.setitem(model_download.MODELS, "test", spec)

    with pytest.raises(ModelDownloadError, match="sha256"):
        await download_model_async(
            tmp_path / "dest",
            "test",
            session=FakeSession(data),
        )

    assert not (tmp_path / "dest" / ".fake.tar.bz2.part").exists()


@pytest.mark.asyncio
async def test_download_resumes_with_range(tmp_path: Path, monkeypatch):
    data = make_archive(tmp_path)
    spec = make_spec(data)
    monkeypatch.setitem(model_download.MODELS, "test", spec)
    dest = tmp_path / "dest"
    dest.mkdir()
    split = len(data) // 2
    (dest / ".fake.tar.bz2.part").write_bytes(data[:split])
    session = FakeSession(data[split:], status=206)

    result = await download_model_async(dest, "test", session=session)

    assert is_model_complete(result, "model.onnx")
    assert session.headers[0] == {"Range": f"bytes={split}-"}


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "evil.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ModelDownloadError, match="unsafe"):
        _safe_extract(archive, tmp_path / "extract")
