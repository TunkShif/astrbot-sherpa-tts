# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-25

### Added

- Initial release of the local sherpa-onnx / Kokoro TTS plugin for AstrBot.
- `/tts` slash command with `/语音` and `/朗读` aliases.
- `sherpa_tts` LLM tool for automatic voice replies.
- `/tts`-independent model management: fp32 and int8 Kokoro v1.1 downloads
  with SHA-256 verification.
- WebUI settings for model directory/precision, speaker, speed, CPU threads,
  text limits, cooldown and queueing.
- Event-scoped temporary WAVs under AstrBot's temp directory.
- Bounded single-flight inference queue and per-user cooldown to protect the
  CPU on small homelab servers.
- Offline AstrBot test suite, optional real-model test, model downloader and
  CPU benchmark script.

[0.1.0]: https://github.com/tunkshif/astrbot_plugin_sherpa_tts/releases/tag/v0.1.0
