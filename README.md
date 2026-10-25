# astrbot_plugin_sherpa_tts

基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 与 Kokoro 多语言模型的 AstrBot **本地语音合成**插件。不调用云端 TTS API，不上传文本，适合部署在家庭服务器、NAS 或旧笔记本上。

支持 `/tts` 斜杠指令和 `sherpa_tts` LLM 工具：用户可以主动要求“发语音”，也可以让模型在合适的时候自动把回复读出来。生成结果会作为 QQ 语音消息发送（OneBot v11 / NapCat）。

![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.25-0ea5e9)
![Python](https://img.shields.io/badge/Python-%3E%3D3.12-3776ab)
![License](https://img.shields.io/badge/License-MIT-green)

## 特性

- **完全本地**：sherpa-onnx + Kokoro ONNX，CPU 推理，无需 GPU，不产生 API 费用。
- **中文 / 英文**：Kokoro v1.1 支持 103 个音色，默认 `zf_001` 中文女声。
- **两种触发方式**：`/tts <文本>` 指令，或 LLM 工具 `sherpa_tts`。
- **适合小机器**：单并发推理、有界排队、每用户冷却、3 句 / 100 字 / 20 秒预算，防止刷 CPU。
- **临时音频**：WAV 写在 AstrBot 临时目录，消息发送后自动删除，不落插件数据目录。
- **模型分离 + 激活自动下载**：插件仓库不包含模型；首次启用时在后台自动下载（可关闭），也提供 SHA-256 校验的脚本，支持 fp32 / int8。
- **可配置**：模型目录/精度、音色、语速、线程数、文本限制、冷却与队列全部可在 WebUI 配置。
- **可测试**：AstrBot 真实 `PluginManager` 离线加载、指令/LLM 工具注册、OneBot payload、真实模型可选测试。

## 运行要求

| 项目 | 要求 |
| --- | --- |
| AstrBot | `>=4.25,<5`（依赖事件级临时文件 API） |
| Python | `>=3.12` |
| 平台 | `aiocqhttp`（NapCat / OneBot v11） |
| 依赖 | `sherpa-onnx>=1.13,<2`、`numpy>=1.24`、`aiohttp>=3.9` |
| 模型 | Kokoro `kokoro-multi-lang-v1_1`（单独下载） |

### 性能参考

- 测试机（Apple M4，4 threads，full 模型）：61 字 / 3 句，音频 10.7s，生成 3.2s（RTF ≈ 0.30）。
- 目标机（AMD Ryzen 7 4800H，8C/16T，16GB）：预计 RTF ≈ 0.5–0.9；3 句约 5–11s 生成，100 字硬上限约 9–18s 生成。
- 以上是估算，不是 4800H 实测。请运行 [`scripts/bench_tts.py`](scripts/bench_tts.py) 获取目标机真实 RTF。
- 16GB 内存不是瓶颈；模型 + ONNX Runtime + AstrBot 通常 <3GB。

## 安装

### 方式一：插件市场（推荐）

在 AstrBot WebUI 的 **插件 → 插件市场** 搜索 `astrbot_plugin_sherpa_tts` 或“本地语音合成”，安装后重载插件。

### 方式二：手动安装

```bash
cd <AstrBot>/data/plugins
git clone https://github.com/tunkshif/astrbot_plugin_sherpa_tts.git
```

然后在 WebUI 插件页点击 **重载插件**。AstrBot 会读取 `requirements.txt` 并安装依赖；如果自动安装失败，可手动执行：

```bash
cd <AstrBot>
.venv/bin/python -m pip install -r data/plugins/astrbot_plugin_sherpa_tts/requirements.txt
```

> 插件依赖固定为 `sherpa-onnx>=1.13,<2`。sherpa-onnx 2.0 移除了 espeak-ng/piper-phonemize，Kokoro 变成纯 lexicon，未收录的英文词会被跳过；v1.13 + full 模型是当前验证过的质量基线。

### 激活时会发生什么

AstrBot 加载/启用插件时会调用插件的 `initialize()`。插件在这个生命周期里检查默认模型目录：

1. 模型完整：直接可用，不做任何下载。
2. 模型缺失且 `auto_download_model=true`（默认）：启动一个**后台下载任务**，`initialize()` 立即返回，不阻塞 AstrBot 启动。
3. 下载进度写入日志（每 10%）；下载完成后自动切换到新模型目录。
4. 下载期间 `/tts` 会回复“模型正在下载（xx%）”；下载失败后首次 `/tts` 会提示错误，并在约 60 秒后允许重试。
5. 插件被禁用/卸载时，`terminate()` 会取消下载任务；未完成的 `.part` 文件保留，下次启用会断点续传。

> 只有默认模型目录会自动下载。若手动配置了 `model_dir`，插件认为该目录由用户管理，不会覆盖或写入它。

### Docker / 无 CLI 依赖

下载和解压**不需要 `curl`、`wget`、`tar`、`unzip` 或 `bzip2` 命令**：

| 操作 | 实现 |
| --- | --- |
| HTTP 下载 | `aiohttp`（AstrBot 已有依赖，插件 requirements 也显式声明） |
| 断点续传 | HTTP `Range` |
| 解压 `.tar.bz2` | Python 标准库 `tarfile` + `bz2` |
| 校验 | 标准库 `hashlib`（SHA-256） |
| 目录替换 | 标准库 `shutil` / `pathlib` |

Docker 部署注意事项：

- 确认 Python 带 `bz2` 模块（官方 `python:3.12-slim` 通常自带）：

  ```bash
  docker exec -it <astrbot-container> python -c "import bz2, tarfile; print('ok')"
  ```

  如果缺少，会得到明确的“Python 缺少 bz2 模块”错误；换一个带 bz2 的 Python 镜像即可，不需要安装 Linux 命令。
- `data/plugin_data` 和 `data/temp` 必须可写。建议把 AstrBot 的 `data/` 挂载为持久卷，否则容器重建后模型会重新下载。
- 磁盘峰值（下载 + 解压）：fp32 约 **800MB**，int8 约 **360MB**；完成后压缩包/临时文件会删除，只保留模型目录。
- AstrBot 容器本身不需要 ffmpeg；QQ 语音的 silk 转码由 NapCat 负责。若 NapCat 在另一个精简容器里，请确保 NapCat 自带 ffmpeg/silk 支持。
- Docker 内访问 GitHub 慢时，设置 `model_download_base_url` 指向可访问的镜像。

## 模型下载（自动/手动）

插件**不包含模型**。默认模型放在插件数据目录：

```text
<AstrBot>/data/plugin_data/astrbot_plugin_sherpa_tts/models/
├── kokoro-multi-lang-v1_1/          # fp32，推荐
│   ├── model.onnx
│   ├── voices.bin
│   ├── tokens.txt
│   ├── lexicon-us-en.txt
│   ├── lexicon-zh.txt
│   ├── espeak-ng-data/
│   └── ...
└── kokoro-int8-multi-lang-v1_1/     # int8，可选
```

### 自动下载（默认）

安装并启用插件后，若默认目录没有模型，会自动下载 `fp32`（约 365MB）。想用更小的 `int8`（约 147MB），先把 `model_precision` 设为 `int8` 再重载插件；它会在激活时下载 int8。

如果 GitHub 访问慢，可在配置中设置 `model_download_base_url` 为镜像/代理的 release 前缀，例如：

```text
https://your-mirror.example/sherpa-onnx/tts-models
```

### 使用脚本（手动/指定版本）

```bash
cd <AstrBot>/data/plugins/astrbot_plugin_sherpa_tts

# fp32：质量最好，约 365MB 下载 / 311MB 模型
<AstrBot>/.venv/bin/python scripts/download_model.py \
  --dest <AstrBot>/data/plugin_data/astrbot_plugin_sherpa_tts/models \
  --variant fp32

# 或者 int8：约 147MB 下载 / 114MB 模型
<AstrBot>/.venv/bin/python scripts/download_model.py \
  --dest <AstrBot>/data/plugin_data/astrbot_plugin_sherpa_tts/models \
  --variant int8
```

脚本会校验官方 release 的 SHA-256、安全解压，并打印应填入的 `model_dir`。

### 手动下载（使用 curl/tar；不想用 CLI 请用自动下载或脚本）

```bash
mkdir -p <AstrBot>/data/plugin_data/astrbot_plugin_sherpa_tts/models
cd <AstrBot>/data/plugin_data/astrbot_plugin_sherpa_tts/models

curl -L -o kokoro.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2
tar -xjf kokoro.tar.bz2
```

下载地址（[官方模型页](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_1.html)）：

- fp32：`kokoro-multi-lang-v1_1.tar.bz2`
- int8：`kokoro-int8-multi-lang-v1_1.tar.bz2`

## 配置

安装后在 **插件 → 本地语音合成** 中配置，保存后重载插件。

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `model_dir` | 空 | 留空自动查找 `models/kokoro-multi-lang-v1_1`，再找 `kokoro-int8-multi-lang-v1_1` |
| `model_precision` | `auto` | `auto` / `fp32` / `int8`；auto 优先 full，缺失时用 int8；也决定自动下载的版本（auto/fp32→fp32） |
| `auto_download_model` | `true` | 插件激活时若默认模型目录缺失，后台自动下载；关闭则需手动下载 |
| `model_download_base_url` | 空 | 模型 release 的镜像/代理前缀，末尾不要加 `/` |
| `auto_load` | `false` | 启动时预加载模型；关闭则首次合成稍慢 |
| `load_timeout` | `180` | 预加载超时（秒） |
| `speaker_id` | `3` | 0–102；3=`zf_001`，58=`zm_009`（第一个中文男声） |
| `speed` | `1.0` | 0.5–2.0；语速越慢，允许的字数上限会自动收紧 |
| `num_threads` | `2` | ONNX Runtime CPU 线程；4800H 建议 `4` |
| `max_chars` | `160` | 单段合成长度，一般无需修改 |
| `max_sentences` | `3` | 单次最多句数 |
| `max_total_chars` | `100` | 配置上限；运行时还受 20 秒预算约束 |
| `cooldown_seconds` | `5` | 每用户冷却；0 表示关闭 |
| `queue_timeout` | `15` | 排队等待上限，超时返回“繁忙” |
| `max_queue` | `2` | 最多等待请求数，再多直接拒绝 |
| `admin_bypass` | `true` | 管理员绕过冷却（不绕过队列） |
| `debug` | `false` | sherpa-onnx 调试日志 |

## 使用

### 指令

```text
/tts 你好，这是本地语音合成。
/语音 今天天气不错。
/朗读 AstrBot 和 sherpa-onnx 都能正常工作。
```

- 指令别名：`/语音`、`/朗读`。
- 群聊中需要 @ 机器人或带唤醒前缀，具体取决于 AstrBot 的唤醒配置。
- 超过 3 句或当前语速下的字数上限会直接提示错误，不会截断后静默发送。

### LLM 工具

直接对机器人说：

```text
用语音说一下今天的学习计划。
把这句话读出来：晚安，做个好梦。
```

模型会在合适的时候调用 `sherpa_tts`，生成语音并发送。工具只向模型返回“语音已发送”或错误原因，不会把服务器文件路径暴露给模型或用户。

### 限额与并发

- 默认 **3 句 / 100 字**，目标单次生成 **≤20 秒**。
- 100 字在语速 1.0 下约 17.5 秒音频；按保守 RTF=1.0 估算约 17.5 秒生成，留出预热/传输余量。
- 语速调慢时字数上限自动按比例收紧（例如 0.5 倍速最多 50 字），避免音频翻倍导致生成超时。
- 同一时间只跑一个 ONNX 推理；最多 2 个请求等待，等待超过 15 秒返回“繁忙”。
- 每用户 5 秒冷却，防止连续刷 `/tts`；管理员可绕过冷却。
- AstrBot 自身的 `platform_settings.rate_limit` 可作为全局消息限流兜底。

### 临时文件与数据目录

| 内容 | 位置 | 生命周期 |
| --- | --- | --- |
| 模型 | `data/plugin_data/astrbot_plugin_sherpa_tts/models/` | 持久保存 |
| 生成 WAV | `data/temp/astrbot_plugin_sherpa_tts/` | 事件结束自动删除 |
| 插件配置 | `data/config/astrbot_plugin_sherpa_tts_config.json` | 持久保存 |

生成音频通过 `event.track_temporary_local_file()` 注册，AstrBot 在消息发送完成后调用 `cleanup_temporary_local_files()` 删除。插件启动时还会清理超过 10 分钟的崩溃残留 WAV。

## 平台与消息链路

当前只声明 `aiocqhttp`（NapCat / OneBot v11）：

```text
sherpa-onnx -> 16-bit 24kHz WAV -> Record
  -> AstrBot aiocqhttp: {"type":"record","data":{"file":"base64://..."}}
  -> NapCat: 内置 silk/ffmpeg 转码 -> QQ 语音
```

插件不需要安装 pysilk/ffmpeg，也不需要自己生成 silk。QQ 官方机器人（`qq_official`）使用另一条 `tencent_silk` 链路，本版本未适配。

## 故障排查

### 模型正在下载 / 下载失败

- 首次启用时后台下载 147–365MB，期间 `/tts` 会提示百分比；等待完成即可。
- 下载失败会写日志（`[sherpa-tts] 模型下载失败`），首次 `/tts` 会显示原因；约 60 秒后可重试，或重启插件/手动运行下载脚本。
- 国内网络可在配置中设置 `model_download_base_url`。
- 检查 `data/plugin_data/astrbot_plugin_sherpa_tts/models/` 下是否有 `.part` 文件；保留它可断点续传。

### `模型文件不完整` / `ModelNotReadyError`

检查 `model_dir` 是否指向包含 `model.onnx`（或 `model.int8.onnx`）、`voices.bin`、`tokens.txt`、`lexicon-us-en.txt`、`lexicon-zh.txt` 的目录。使用下载脚本后会打印正确路径。

### `ModuleNotFoundError: sherpa_onnx`

依赖未安装到 AstrBot 使用的 Python 环境：

```bash
<AstrBot>/.venv/bin/python -m pip install -r \
  <AstrBot>/data/plugins/astrbot_plugin_sherpa_tts/requirements.txt
```

### 没有声音 / 只有文字

- 确认平台是 `aiocqhttp`，且 NapCat 正常连接。
- 查看 AstrBot 日志中是否有 `record` 段发送记录。
- NapCat 需要能调用内置 ffmpeg/silk 转码；部分精简版 NapCat 需要额外安装 ffmpeg。
- 先用很短的 `/tts 你好` 测试，排除文本超限或模型问题。

### 提示“语音合成繁忙”或“请求太频繁”

- 前者表示 CPU 队列已满/等待超时；稍后再试，或降低 `max_queue`、提高 `queue_timeout`。
- 后者是 `cooldown_seconds`；管理员可绕过，或把冷却调小。
- 4800H 上建议 `num_threads=4`，不要占满 16 线程。

### 英文单词被跳过 / 发音奇怪

- 使用 `sherpa-onnx>=1.13,<2` + fp32 模型（默认推荐）以获得 espeak-ng OOV 回退。
- 如果手动升级到 sherpa-onnx 2.x，Kokoro 变成纯 lexicon，未收录英文词会被跳过。
- 文本尽量使用中文；品牌名可加入自定义 lexicon（后续版本支持）。

### 首次响应很慢

首次调用需要加载 ONNX 模型（约 1–3 秒，视磁盘而定）。可以开启 `auto_load` 在插件加载时预热，但会拖慢 AstrBot 启动。

## 开发与测试

```bash
git clone https://github.com/tunkshif/astrbot_plugin_sherpa_tts.git
cd astrbot_plugin_sherpa_tts

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "astrbot==4.29.3" "sherpa-onnx>=1.13,<2" "aiohttp>=3.9" numpy \
  pytest pytest-asyncio pytest-cov ruff

ASTRBOT_ROOT=$(mktemp -d) .venv/bin/python -m pytest -q
```

离线测试覆盖：指令/LLM 工具、注册表、真实 `PluginManager` 加载、OneBot payload、限额/冷却/队列、临时文件清理。真实模型测试默认跳过：

```bash
SHERPA_TTS_MODEL_DIR=<...>/kokoro-multi-lang-v1_1 \
  .venv/bin/python -m pytest -q -m model
```

更多设计说明见 [docs/design.md](docs/design.md)，测试矩阵见 [docs/testing.md](docs/testing.md)。

## 项目结构

```text
astrbot_plugin_sherpa_tts/
├── astrbot_plugin_sherpa_tts/
│   ├── main.py                 # 指令、LLM 工具、限额、队列、临时文件
│   ├── metadata.yaml
│   ├── _conf_schema.json
│   ├── requirements.txt
│   ├── model_download.py      # 激活自动下载 / SHA-256 / 断点续传
│   └── tts/
│       ├── engine.py           # sherpa-onnx OfflineTts 封装
│       ├── text.py             # 分句与 20 秒字数预算
│       └── audio.py            # WAV 编码与 NaN 清理
├── scripts/
│   ├── download_model.py       # SHA-256 模型下载/解压
│   └── bench_tts.py            # 目标机 RTF 基准
├── tests/                      # 离线测试 + 可选真实模型测试
├── docs/
│   ├── design.md
│   └── testing.md
├── pyproject.toml
├── LICENSE
└── README.md
```

## 已知限制

- 仅支持 `aiocqhttp` / OneBot v11；QQ 官方机器人未适配。
- 音色暂用数字 `speaker_id`，尚未提供音色名称列表与 `/tts voice`。
- 长文本不会自动分段成多条语音；超过 3 句 / 100 字会直接拒绝。
- 没有持久化任务队列；服务重启会丢失排队中的请求（短任务可接受）。
- 自动下载依赖 GitHub Release；国内网络可能需要 `model_download_base_url` 镜像。

## 许可证

- 本插件代码：[MIT](LICENSE)。
- Kokoro 模型：[Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)。
- sherpa-onnx：[Apache-2.0](https://github.com/k2-fsa/sherpa-onnx)。
- 模型压缩包内的 `espeak-ng-data` 等资源遵循其上游许可证；本仓库不重新分发模型。

## 致谢

- [AstrBot](https://github.com/AstrBotDevs/AstrBot) 插件框架。
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 与 [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)。
- [astrbot_plugin_image_generation](https://github.com/Railgun19457/astrbot_plugin_image_generation) 与 [astrbot_plugin_GPT_SoVITS](https://github.com/Zhalslar/astrbot_plugin_GPT_SoVITS) 的 LLM 工具/语音发送实现参考。

## 免责声明

本项目为个人自托管插件，与 AstrBot、sherpa-onnx、Kokoro、腾讯或 NapCat 官方无隶属关系。语音内容由用户输入和本地模型生成，请遵守所在平台的服务条款与当地法律法规。
