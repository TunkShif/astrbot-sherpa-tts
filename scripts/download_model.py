#!/usr/bin/env python3
"""Download and extract a Kokoro v1.1 model for the AstrBot plugin.

The plugin can also download automatically on activation; this script is for
manual installs, forced re-downloads, or custom mirrors.

Example:
    <AstrBot>/.venv/bin/python scripts/download_model.py \
        --dest <AstrBot>/data/plugin_data/astrbot_plugin_sherpa_tts/models \
        --variant fp32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astrbot_plugin_sherpa_tts.model_download import (  # noqa: E402
    MODELS,
    download_model_sync,
    is_model_complete,
    model_url,
)


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
        "--base-url",
        default=None,
        help="override the GitHub release base URL (for mirrors/proxies)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.variant]
    final_dir = args.dest / spec.top_dir

    if args.force and final_dir.exists():
        import shutil

        shutil.rmtree(final_dir)

    if is_model_complete(final_dir, spec.model_file):
        print(f"already present: {final_dir}")
        print(f"model_dir = {final_dir.resolve()}")
        return 0

    last_percent = -1

    def on_progress(done: int, total: int) -> None:
        nonlocal last_percent
        percent = int(done * 100 / total) if total else 0
        if percent >= last_percent + 5:
            print(
                f"\r  {percent:3d}%  {done / 2**20:.1f}/{total / 2**20:.1f} MiB", end=""
            )
            last_percent = percent

    print(f"downloading {args.variant}: {model_url(spec, args.base_url)}")
    try:
        path = download_model_sync(
            args.dest,
            args.variant,
            base_url=args.base_url,
            progress=on_progress,
        )
    except Exception as exc:
        raise SystemExit(f"download failed: {exc}") from exc

    print()
    print("done. Set this in the plugin settings:")
    print(f"  model_dir = {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
