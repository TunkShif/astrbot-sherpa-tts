# 测试与验证

本项目把“能离线验证的”尽量放到 pytest 里，把必须真机的部分明确列为发布前 checklist。

## 1. 快速运行

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "astrbot==4.29.3" "sherpa-onnx>=1.13,<2" numpy pytest pytest-asyncio pytest-cov ruff

ASTRBOT_ROOT=$(mktemp -d) .venv/bin/python -m pytest -q
```

当前结果：

```text
48 passed, 1 skipped
coverage: 83%
```

跳过的测试是 `tests/test_real_model.py`，它需要真实模型。

## 2. 分层测试矩阵

| 层 | 依赖 | 验证内容 | 对应文件 |
| --- | --- | --- | --- |
| L0 纯逻辑 | 无 | 文本清理、分句、句数、动态字数上限、WAV、NaN 清洗 | `tests/test_engine.py` |
| L1 引擎适配 | 无（fake `sherpa_onnx`） | 配置字段、1.x/2.x 分支、精度选择、WAV 输出 | `tests/test_engine.py` |
| L2 Handler | `astrbot` wheel | 指令/工具产生 `Record`、错误、限额、冷却、队列、临时文件、自动下载状态 | `tests/test_handlers.py` |
| L3 注册契约 | `astrbot` | 指令名/别名、LLM 工具 schema、注册表 | `tests/test_registration.py` |
| L3.5 模型下载 | 无（fake HTTP） | 精度选择、SHA、断点续传、安全解压 | `tests/test_model_download.py` |
| L4 真实加载 | `astrbot` | metadata、schema、版本、import、实例化、`initialize`、handler 绑定 | `tests/test_plugin_load.py`（子进程） |
| L5 平台序列化 | `astrbot` | `Record` → OneBot `record` → `send_group_msg` kwargs | `tests/test_onebot_payload.py` |
| L6 真实模型 | 模型 + 原生库 | 真实 ONNX 推理、24kHz WAV、时长/非空 | `tests/test_real_model.py` |
| L7 真实投递 | AstrBot + NapCat + QQ | 语音真的可播放、silk 转码、客户端兼容 | 发布前真机 checklist |
| L8 LLM 决策 | AstrBot + LLM provider | 模型是否在合适时机调用工具 | 可选真实 provider 测试 |

## 3. 关键测试技巧

### 3.1 `ASTRBOT_ROOT` 必须早于 import

`import astrbot.api` 会执行 `astrbot/core/__init__.py`，创建 `<root>/data`、SQLite、默认配置、t2i 模板。`tests/conftest.py` 在导入 AstrBot 之前设置：

```python
os.environ["ASTRBOT_ROOT"] = tempfile.mkdtemp(prefix="astrbot-tts-test-")
```

这会让测试完全不污染开发机/CI。`ruff` 对 conftest 的 `E402` 做了 per-file ignore，这是必要的导入顺序。

### 3.2 真实事件对象，而不是 MagicMock

`tests/conftest.py` 的 `RecordingEvent` 继承真实 `AstrMessageEvent`，只替换 `send()` 为记录列表。因此 `chain_result()`、`get_message_str()`、`unified_msg_origin`、`track_temporary_local_file()` 都是真实行为。

### 3.3 真实 `PluginManager` 子进程

`tests/test_plugin_load.py` 启动一个子进程，把插件复制到临时 `data/plugins/`，用真实 `PluginManager.load()` 加载。它覆盖：

- `metadata.yaml` 解析、`astrbot_version` 校验；
- `_conf_schema.json` 加载；
- 模块 import、Star 子类发现、实例化、`initialize()`；
- `@filter.command` / `@filter.llm_tool` 注册与绑定；
- 指令和工具各产生一次 `Record`。

子进程隔离是必要的，因为插件管理器会修改 `star_map`、`star_handlers_registry`、`llm_tools` 等进程级全局状态，并创建 SQLite 数据库。

### 3.4 假引擎测限额与队列

`tests/conftest.py` 的 `FakeEngine` 实现与 `SherpaTtsEngine` 相同的 `synthesize_async()`，写入 0.1s 静音 WAV，不加载模型。这样可以快速测：

- 3 句 / 100 字 / 动态语速上限；
- 每用户冷却、管理员绕过；
- 队列满时快速返回“繁忙”；
- 成功/失败后释放推理槽位；
- 临时 WAV 注册与清理。

### 3.5 真实 aiocqhttp 发送路径

`tests/test_onebot_payload.py`：

- 直接调用 `AiocqhttpMessageEvent._from_segment_to_dict()`，断言 `{"type":"record","data":{"file":"base64://..."}}`；
- 再进一步，用真实 `AiocqhttpMessageEvent.send()` + fake `send_group_msg`，断言 `group_id`、`message` 结构、base64 解码后的 `RIFF/WAVE` 头。

这覆盖“插件产出”到“即将写 WebSocket”之间的全部 AstrBot 代码，不需要启动 NapCat。

## 4. 真实模型测试

```bash
SHERPA_TTS_MODEL_DIR=<...>/kokoro-multi-lang-v1_1 \
  .venv/bin/python -m pytest -q -m model
```

或在 CI nightly / 本地用：

```bash
SHERPA_TTS_MODEL_DIR=<...> .venv/bin/python -m pytest -m model
```

建议同时测 fp32 与 int8，并记录：

- 模型加载时间；
- 100 字 / 3 句的生成时间；
- 音频时长、文件大小；
- 是否出现 NaN 警告或 OOV 跳过日志。

`scripts/bench_tts.py` 是更完整的基准工具，输出 RTF 和多次运行的 median。

## 5. 离线无法证明的部分

| 项目 | 原因 | 发布前怎么做 |
| --- | --- | --- |
| QQ 真的收到并播放 | 需要 NapCat + QQ 账号 | 真机 `/tts 你好`，看语音气泡、时长、波形 |
| NapCat silk/ffmpeg | NapCat 内部转码 | 在目标 NapCat 版本跑一次；检查日志 |
| 真实 LLM 选择工具 | 取决于模型/prompt | 用实际 provider 说“用语音说…”，看工具调用 |
| 音质/发音 | 主观 + 模型/音色 | 人工试听；ASR 回环可选 |
| 长稳/并发/内存 | 单测不长时间跑 | 4–8 并发压测；观察内存、队列、温度 |
| 4800H 真实 RTF | 目标机才有 | 跑 `scripts/bench_tts.py` |
| AstrBot 版本漂移 | 核心 API 会变 | CI 矩阵 4.25.0 / 4.29.3 |

## 6. CI

`.github/workflows/ci.yml`：

- Python 3.12；
- AstrBot 4.25.0（最低支持）与 4.29.3（当前验证）；
- `ruff check` + `ruff format --check`；
- `pytest -q -m "not model"`；
- 不下载 147–365MB 模型，真实模型测试留给 nightly/本地。

## 7. 常见测试问题

- **`Event loop is closed` / `audioop` 警告**：AstrBot 的 SQLite/音频依赖在测试结束时的清理警告，不影响结果；不要把它当成插件失败。
- **没有 `ASTRBOT_ROOT`**：conftest 默认会创建临时目录；手动跑单个测试时也建议设置。
- **`star_map` 里出现旧插件**：`test_plugin_load` 用子进程隔离；如果自己写测试，避免在同一进程重复加载同一模块。
- **真实模型测试 404/缺文件**：检查 `SHERPA_TTS_MODEL_DIR` 是否指向解压后的顶层目录，而不是 `models/`。
