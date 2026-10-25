#!/usr/bin/env python3
"""Download and extract a Kokoro v1.1 model for the AstrBot plugin.

The plugin repository intentionally does not bundle the model.  Run this on
the AstrBot host (or download manually and point ``model_dir`` at it):

    python scripts/download_model.py --dest /srv/astrbot/data/plugin_data/astrbot_plugin_sherpa_tts/models

The archive is downloaded from the official sherpa-onnx GitHub release and its
SHA-256 is verified before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

MODELS = {
    "fp32": {
        "archive": "kokoro-multi-lang-v1_1.tar.bz2",
        "top_dir": "kokoro-multi-lang-v1_1",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2",
        "sha256": "a3f4c73d043860e3fd2e5b06f36795eb81de0fc8e8de6df703245edddd87dbad",
    },
    "int8": {
        "archive": "kokoro-int8-multi-lang-v1_1.tar.bz2",
        "top_dir": "kokoro-int8-multi-lang-v1_1",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2",
        "sha256": "a1e94694776049035c4f2c6529f003aaece993c76aae9a78995831c3c4dcafc6",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "astrbot-sherpa-tts"})
    with urllib.request.urlopen(request) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            done += len(chunk)
            if total:
                percent = done * 100 // total
                print(
                    f"\r  {percent:3d}%  {done / 2**20:.1f}/{total / 2**20:.1f} MiB",
                    end="",
                )
    print()


def safe_extract(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if target != dest and dest not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(dest, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", required=True, type=Path, help="model download directory"
    )
    parser.add_argument(
        "--variant",
        choices=sorted(MODELS),
        default="fp32",
        help="fp32 = best quality (recommended), int8 = smaller",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the downloaded .tar.bz2 after extraction",
    )
    args = parser.parse_args()

    info = MODELS[args.variant]
    args.dest.mkdir(parents=True, exist_ok=True)
    model_dir = args.dest / info["top_dir"]
    if model_dir.exists() and not args.force:
        print(f"already present: {model_dir}")
        print(f"model_dir: {model_dir}")
        return 0

    archive = args.dest / info["archive"]
    if args.force or not archive.exists():
        print(f"downloading {args.variant}: {info['url']}")
        download(info["url"], archive)
    else:
        print(f"using cached archive: {archive}")

    print("verifying sha256 ...")
    actual = sha256_file(archive)
    if actual != info["sha256"]:
        raise SystemExit(
            f"sha256 mismatch for {archive}\nexpected {info['sha256']}\nactual   {actual}",
        )
    print("  ok")

    if model_dir.exists():
        shutil.rmtree(model_dir)
    print(f"extracting to {model_dir} ...")
    safe_extract(archive, args.dest)

    model_file = model_dir / (
        "model.onnx" if args.variant == "fp32" else "model.int8.onnx"
    )
    if not model_file.exists():
        raise SystemExit(f"model file missing after extraction: {model_file}")
    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    print()
    print("done. Set this in the plugin settings:")
    print(f"  model_dir = {model_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
