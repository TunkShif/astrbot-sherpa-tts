# 插件设计

> 状态：v0.1.0（2026-10-25）  
> 目标运行环境：AstrBot >= 4.25、Python >= 3.12、AMD Ryzen 7 4800H / 16GB 级别的 CPU-only 家庭服务器。

## 1. 目标与非目标

### 目标

- 用本地 sherpa-onnx + Kokoro 模型完成中英文 TTS，不调用云端 API。
- 同时提供 `/tts` 斜杠指令和 `sherpa_tts` LLM 工具。
- 在 QQ（OneBot v11 / NapCat）上以真正的语音消息发送，而不是发文件或链接。
- 在旧笔记本/小机器上保护 CPU：单并发、有限排队、每用户冷却、文本硬上限。
- 模型与插件分离；模型可校验下载，插件包保持远小于市场 16MB 限制。
- 生成的 WAV 是临时文件，消息发送后自动清理。

### 非目标（v0.1）

- 不支持 `qq_official` 的 `tencent_silk` 链路。
- 不做 image-generation 风格的后台任务系统、进度页、取消/恢复。
- 不内置模型，不自动在首次 `/tts` 时隐式下载。
- 不做音色名称表、`/tts voice`、i18n、Web 管理页（列入路线图）。
- 不做长文自动拆成多条语音。

## 2. 架构

```text
                      AstrBot event
                           |
              +------------+-------------+
              |                          |
        /tts command                 sherpa_tts LLM tool
              |                          |
              +------------+-------------+
                           |
                    SherpaTtsPlugin
                           |
        +------------------+------------------+
        |                  |                  |
   text budget        cooldown/queue      temp file
   (3 句/100 字/20s)  (1 active, 2 wait)  data/temp/...
        |                  |                  |
        +------------------+------------------+
                           |
                 SherpaTtsEngine (thread)
                           |
                 sherpa-onnx OfflineTts
                           |
                 16-bit 24kHz mono WAV
                           |
                  Record.fromFileSystem()
                           |
             event.chain_result / event.send
                           |
               aiocqhttp: base64 record
                           |
                 NapCat: silk transcode -> QQ
```

### 模块职责

| 模块 | 职责 |
| --- | --- |
| `main.py` | AstrBot 生命周期、指令/工具、限额、冷却、队列、临时文件 |
| `tts/engine.py` | 懒加载 `OfflineTts`、精度选择、版本兼容、线程锁、WAV 输出 |
| `tts/text.py` | 朗读清理、分句、句数统计、20 秒字数预算 |
| `tts/audio.py` | float32 -> PCM16 WAV、NaN/Inf 清洗 |
| `scripts/download_model.py` | 官方模型下载、SHA-256、安全解压 |
| `scripts/bench_tts.py` | 目标机 RTF/延迟基准 |

## 3. 请求流程

### `/tts <文本>`

1. `@filter.command("tts", alias={"语音", "朗读"})` 接收事件。
2. 从 `event.message_str` 截取命令后的完整文本（AstrBot 已去掉 `/` 唤醒前缀）。
3. 校验：非空、句数、动态字数、语速范围。
4. 校验每用户冷却。
5. 获取 CPU 推理槽位（有界等待）。
6. 在 `asyncio.to_thread` 中合成 WAV。
7. `event.track_temporary_local_file(path)` 注册临时文件。
8. `yield event.chain_result([Record.fromFileSystem(path)])`。
9. Pipeline 发送消息；事件结束时 AstrBot 删除临时 WAV。

### 插件生命周期与下载时机

AstrBot 在插件成功 import/实例化之后、触发 `OnPluginLoadedEvent` 之前，会 **await** 插件的 `initialize()`。这就是“安装/启用后”的可靠时机，不需要等第一次 TTS：

```text
plugin import -> instantiate -> bind handlers -> initialize() -> OnPluginLoadedEvent
```

- **`initialize()`**：启动后台模型下载任务后立即返回；不能在这里 `await` 365MB 下载，否则插件加载会卡住。
- **`OnPluginLoadedEvent`**（`@filter.on_plugin_loaded`）：在 `initialize()` 之后触发，且会对所有插件加载触发；可以用来做额外通知，但启动下载用 `initialize()` 更直接。
- **`OnAstrBotLoadedEvent`**（`@filter.on_astrbot_loaded`）：只在 AstrBot 整体启动完成时触发；之后从插件市场安装/热重载插件不会触发，不能作为下载时机。
- **`terminate()`**：插件禁用/卸载/重载时取消后台下载任务；已下载的 `.part` 保留，下次启用断点续传。
- 没有独立的“安装完成”事件；AstrBot 的安装流程最终会加载并启用插件，因此 `initialize()` 就是实际可用的激活钩子。

### `sherpa_tts` LLM 工具

与指令共用 3–7 步，然后：

```python
await event.send(event.chain_result([record]))
return "语音已发送给用户。"
```

工具返回短字符串给模型，不返回路径；失败时返回错误文本，让模型决定是否重试或改用文字。

## 4. 20 秒生成预算

### 公式

```text
audio_seconds ≈ chars / 5.7 / speed
generation_seconds ≈ audio_seconds × RTF
```

- `5.7`：中文在 speed=1.0 时的近似字/秒（实测 61 字 ≈ 10.7s 音频）。
- `RTF`：目标机实测值；设计预算按保守的 **1.0** 计算，避免依赖乐观估计。
- `MAX_GENERATION_SECONDS = 20`。
- 因此硬预算为 `100` 字（100 / 5.7 ≈ 17.5s 音频），留出约 2.5s 余量给预热、WAV 编码和传输。

### 动态字数上限

```python
target = int(100 * clamp(speed, 0.5, 2.0))
limit = min(config.max_total_chars, target)
```

| 语速 | 动态上限 | 音频长度（约） | RTF=1.0 生成（约） |
| ---: | ---: | ---: | ---: |
| 0.5 | 50 字 | 17.5s | 17.5s |
| 0.8 | 80 字 | 17.5s | 17.5s |
| 1.0 | 100 字 | 17.5s | 17.5s |
| 1.5 | 150 字（若配置允许） | 17.5s | 17.5s |
| 2.0 | 200 字（若配置允许） | 17.5s | 17.5s |

- `max_total_chars` 默认 100，只能进一步收紧，不能突破 20 秒预算。
- `max_sentences=3` 同时生效；短句通常先触发句数限制，长句先触发字数限制。
- 目标机 RTF 更低时（例如 0.7）实际生成约 12s，预算是上限而非预期值。

### 为什么是 100 而不是 114

`20 × 5.7 = 114`，但 114 字没有任何余量；实际还存在标点停顿、语速误差、CPU 降频、WAV/base64/上传时间。100 字提供约 12% 的余量，并让“3 句”通常落在 2–3 句的正常回复范围内。

## 5. 并发与滥用控制

### AstrBot 已有能力

- 核心有 per-session 固定窗口限流（`platform_settings.rate_limit`，默认 30 条/60 秒，`stall`/`discard`）。它限制消息，不限制 CPU，本插件不能只依赖它。
- 核心没有插件级 CPU 队列/任务系统。
- `event.role == "admin"` 可用于管理员绕过冷却。

### 本插件机制

| 机制 | 默认 | 作用 |
| --- | ---: | --- |
| 单并发 `asyncio.Semaphore(1)` | 1 | `OfflineTts` 不并行推理 |
| `max_queue` | 2 | 最多两个等待请求，再多立即拒绝 |
| `queue_timeout` | 15s | 等待超时返回“繁忙”，避免堆积到工具超时 |
| `cooldown_seconds` | 5 | 同一发送者两次请求的最小间隔 |
| `admin_bypass` | true | 管理员绕过冷却 |
| 文本硬上限 | 3 句 / 100 字 | 限制单次 CPU 工作量 |
| AstrBot 全局限流 | 建议 10/60s | 兜底所有消息 |

失败快速返回，不建立持久任务：

- 指令：`语音合成失败：语音合成繁忙，请稍后再试。`
- 工具：同样的错误字符串返回给 LLM。

### 为什么不使用 job system

image-generation 插件需要 job system，因为：外部 API 可能几十秒到几分钟、需要轮询/重试、需要进度/取消、需要重启恢复、需要结果唤醒。TTS 是本地短任务，通常 5–20s，失败可直接重试；引入任务 ID、持久化、后台 worker、结果路由只会增加复杂度和故障面。

只有在以下情况才考虑 job system：

- 单次经常 >30–60s；
- 需要进度、取消、重启恢复；
- 多引擎/多 worker/多 GPU；
- 公开 bot 的大规模并发与公平调度。

## 6. 数据与临时文件

| 内容 | 路径 | 生命周期 |
| --- | --- | --- |
| 模型 | `data/plugin_data/astrbot_plugin_sherpa_tts/models/` | 持久 |
| 下载中的 `.part` | `.../models/.<archive>.part` | 成功后删除；失败/取消保留以续传 |
| 生成 WAV | `data/temp/astrbot_plugin_sherpa_tts/` | 事件结束删除 |
| 插件配置 | `data/config/astrbot_plugin_sherpa_tts_config.json` | 持久 |

- 持久目录：`StarTools.get_data_dir(PLUGIN_NAME)`。
- 临时目录：`get_astrbot_temp_path()`。
- 注册：`event.track_temporary_local_file(path)`，AstrBot 在事件结束调用 `cleanup_temporary_local_files()`。
- 启动时清理超过 10 分钟的遗留 WAV，覆盖崩溃/重载场景。
- 不把音频写进 `plugin_data`，避免磁盘累积。

`track_temporary_local_file` 从 AstrBot 4.25 开始提供；因此 `metadata.yaml` 声明 `astrbot_version: ">=4.25,<5"`。

## 7. 模型管理

- 插件仓库不包含模型，`requirements.txt` 只包含库依赖。
- **激活自动下载**：`initialize()` 检测默认模型目录缺失后，创建后台 `asyncio.Task` 调用 `model_download.download_model_async()`；`initialize()` 不等待下载。默认 `auto_download_model=true`，`model_dir` 为空时生效；手动设置 `model_dir` 视为用户管理，不自动写入。
- **下载状态机**：`idle -> downloading -> ready | failed | cancelled`；下载中 `/tts` 返回百分比，失败后首次 `/tts` 返回错误，约 60 秒后再次触发重试。
- **断点续传**：下载写入 `.<archive>.part`，HTTP `Range` 续传；SHA-256 校验通过后才解压到 staging 目录，再原子替换最终模型目录，避免半个模型。
- **镜像**：`model_download_base_url` 可替换 release 前缀；脚本 `--base-url` 同样支持。
- `model_precision = auto | fp32 | int8`：
  - `auto`：优先 `model.onnx`，缺失时回退 `model.int8.onnx`；
  - `fp32`：311MB，质量最好，中英混合更稳；
  - `int8`：114MB，适合磁盘/下载受限或实测更快的机器。
- `scripts/download_model.py` 从官方 GitHub release 下载，校验 SHA-256，安全解压，默认删除压缩包（`--keep-archive` 可保留）。
- 依赖固定 `sherpa-onnx>=1.13,<2`：
  - 1.x 保留 espeak-ng/piper-phonemize，对 OOV 英文有回退；
  - 2.0 移除了 espeak-ng/piper-phonemize，Kokoro 变成纯 lexicon，未收录英文词会被跳过；
  - 实测 1.13.8 + int8 对某些 OOV 英文可出现 NaN；full 未出现；代码仍有 `nan_to_num` 兜底。
- 首次使用模型缺失时返回可操作的错误，不在请求路径里隐式下载。

## 8. 配置

完整表格见 [README 配置](../README.md#配置)。设计上分四组：

1. **模型**：`model_dir`、`model_precision`、`auto_download_model`、`model_download_base_url`、`auto_load`、`load_timeout`、`debug`。
2. **声音**：`speaker_id`、`speed`、`num_threads`。
3. **限额**：`max_sentences`、`max_total_chars`、`max_chars`。
4. **保护**：`cooldown_seconds`、`queue_timeout`、`max_queue`、`admin_bypass`。

`_conf_schema.json` 的 `slider`/`options` 让 WebUI 直接渲染滑杆和下拉框。

## 9. 错误处理、日志与隐私

- 指令错误：`plain_result` 返回原因。
- 工具错误：返回 `"语音合成失败：..."`，不抛出未捕获异常。
- 模型错误：区分“缺文件”“未安装 sherpa-onnx”“配置校验失败”“生成失败”。
- 日志记录：字符数、句数、队列等待、合成耗时、音频大小；不记录完整文本，只记录前 N 字/摘要。
- 不向 LLM 或用户返回服务器文件路径。
- `terminate()` 释放原生模型引用。

## 10. 兼容性与平台

- AstrBot 4.25+（临时文件 API），Python 3.12+。
- `aiocqhttp`：`Record` -> base64 WAV -> NapCat silk。
- `qq_official`：未适配；它使用 `tencent_silk`，需要单独实现。
- 无 ffmpeg/pysilk 依赖；NapCat 负责最终 silk 转码。
- 模型依赖 sherpa-onnx 的平台 wheel（Linux/macOS/Windows）。

## 11. 决策记录

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 任务模型 | 单并发 + 有界队列，不做 job system | TTS 短、本地、同步；任务系统过度设计 |
| 音频生命周期 | `data/temp` + event tracking | AstrBot 原生清理，不落插件数据 |
| LLM 工具注册 | `@filter.llm_tool`，所有参数给默认值 | 参数少、代码少；装饰器 schema 无 `required`，需自校验 |
| 模型精度默认 | `auto`（优先 fp32） | 质量/稳定优先；int8 可显式选择 |
| 模型下载时机 | `initialize()` 后台任务 | 激活即下载；不阻塞启动；首 TTS 作为失败回退 |
| sherpa-onnx 版本 | `<2` | 保留 OOV 英文回退；2.0 行为变化 |
| 许可证 | MIT | 宽松、适合社区插件 |

## 12. 路线图

- 音色名称映射 + `/tts voice`。
- `/tts status`、`/tts download`（管理员）。
- 自定义 lexicon（品牌名/专有名词）。
- i18n + 插件配置页。
- 长文本后台任务（真正的 job system 场景）。
- `qq_official` 适配。
- ASR 回环测试，检查可懂度。

## 13. 参考

- [AstrBot 插件开发](https://docs.astrbot.app/dev/star/plugin-new.html)
- [AstrBot 发送消息 / Record](https://docs.astrbot.app/dev/star/guides/send-message.html)
- [AstrBot 插件配置](https://docs.astrbot.app/dev/star/guides/plugin-config.html)
- [AstrBot RateLimitStage](https://github.com/AstrBotDevs/AstrBot/blob/v4.29.3/astrbot/core/pipeline/rate_limit_check/stage.py)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Kokoro v1.1 模型页](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_1.html)
- [NapCat 文件/音频处理](https://napneko.pages.dev/develop/file)
