#!/usr/bin/env python3
"""Measure Kokoro TTS latency on the machine that will run AstrBot.

This script does not import AstrBot.  Run it on the actual homelab server to
get the RTF (real-time factor) for the full and int8 models, then pick the
precision and thread count from real data instead of guesses.

Example:
    python scripts/bench_tts.py --model-dir /srv/models/kokoro-multi-lang-v1_1 \
        --precision fp32 --threads 4 --repeats 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.engine import SherpaTtsEngine
from tts.text import count_sentences

DEFAULT_TEXT = (
    "你好，这是本地语音合成测试。今天天气不错，我们来试试中英文混合。"
    "AstrBot 和 sherpa-onnx 都能正常工作。"
)


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "int8"],
        default="auto",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speaker", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    engine = SherpaTtsEngine(
        args.model_dir,
        speaker_id=args.speaker,
        speed=args.speed,
        num_threads=args.threads,
        precision=args.precision,
    )
    print(f"model dir : {args.model_dir}")
    print(f"model file: {engine.model_file.name}")
    print(f"threads   : {args.threads}")
    print(
        f"text      : {len(args.text)} chars / {count_sentences(args.text)} sentences"
    )
    print()

    started = time.perf_counter()
    engine.ensure_ready()
    load_seconds = time.perf_counter() - started
    print(f"model load: {load_seconds:.2f}s")
    print()

    rows: list[tuple[int, float, float, float]] = []
    with tempfile.TemporaryDirectory(prefix="tts-bench-") as tmp:
        for index in range(1, args.repeats + 1):
            out = Path(tmp) / f"run-{index}.wav"
            started = time.perf_counter()
            engine.synthesize(args.text, out)
            generate_seconds = time.perf_counter() - started
            duration = audio_duration(out)
            rtf = generate_seconds / duration if duration else float("inf")
            rows.append((index, generate_seconds, duration, rtf))
            print(
                f"run {index}: generate={generate_seconds:.2f}s "
                f"audio={duration:.2f}s rtf={rtf:.3f} "
                f"size={out.stat().st_size / 1024:.1f}KiB",
            )

    avg_gen = statistics.mean(row[1] for row in rows)
    avg_audio = statistics.mean(row[2] for row in rows)
    median_rtf = statistics.median(row[3] for row in rows)
    print()
    print(
        f"average: generate={avg_gen:.2f}s audio={avg_audio:.2f}s "
        f"median_rtf={median_rtf:.3f}",
    )
    print(
        "rule of thumb: generation time ~= audio duration * RTF; "
        "a 10s sentence costs about 10 * RTF seconds here.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
