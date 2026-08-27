# 视频理解到 retrieval_text Demo

这是一个独立的 Python 3.11 + FastAPI 视频理解 Demo。目标是对独立视频执行音频、画面和多模态理解，输出带证据引用的 `VIDEO_SEGMENT`、`VIDEO_SUMMARY`、`retrieval_text` 和 `retrieval_hash`。

当前状态：Demo 主链已完成。上传、可靠 Worker、ffprobe/FFmpeg、字幕优先、Silero VAD、云端 Whisper、镜头/关键帧、章节多图 VLM、结构化知识文档、证据分页和关键帧查询均已接通。生产链只使用云 ASR、文本 LLM 和章节视觉模型三套配置，结果 Schema 为 3.0。

提交时必须准确区分两类结论：

- Demo 产品链与机器一致性 smoke：已通过。详见[双视频产品链报告](./.codex/reports/two-video-demo-chain-20260821.md)和[双视频质量 smoke](./.codex/reports/two-video-quality-smoke-20260821.md)。
- 正式五语云端 ASR 质量和 M1 两段耐久：仍为 `NOT_RUN` 或外部前置条件不足，不能用中文 Demo 代替。Qwen 公网完整视频视觉理解已真实通过；同一个完整公网 URL 不保证按 `start_ms/end_ms` 截取片段，因此不能冒充精确片段输入。综合验证见[报告](./.codex/reports/qwen-visual-understanding-20260821.md)，公网 URL 对账见[对账报告](./.codex/reports/qwen-remote-url-reconciliation-20260821.md)。

历史实现记录：`示例视频.mp4` 的 921,484 毫秒全片曾完成一次本地 ASR 与增强链验收，产生 601 条 ASR、5,743 条对齐词、329 个关键帧、329 条 OCR 和 128 个场景证据。该数据只描述迁移前实现，当前产品不再生成对齐词，也不能用于证明云端 ASR 已通过。后续严格 Qwen 与融合验收复用了该次运行的代理和已落库证据；报告见[全片严格验收](./.codex/reports/qwen-production-full-video-validation-20260821.json)。

Qwen 的视觉职责与证据边界：完整视频视觉报告可识别人物、场景、账号页、软件界面、关键事件和画面文字；模型观察可能误读账号名、数字或字幕。原始可定位画面文字仍以百度 OCR 证据为权威，语音以字幕或云端 ASR 的 `SpeechSegment` 为权威。需要精确片段视觉理解时，必须传实际派生短片（本地 clip/Data URI 或该短片自己的公网 URL），不得只给完整视频 URL 再依赖时间字段让供应商自动 seek。

## 视频转文本策略

视频文本解析按“内嵌文本字幕优先、云端 ASR 兜底”执行：`ffprobe` 先发现容器内字幕流，系统只把 `subrip`、`ass`、`ssa`、`webvtt` 和 `mov_text` 当作可解析文本字幕。候选字幕经 UTF-8 解析、大小和 cue 数量限制、时间轴及启发式完整性检查后，以独立的 `SUBTITLE_CUE` 证据输出；命中时不生成 `audio.wav`，也不启动 VAD 或调用云端 ASR。

PGS、DVD Subtitle 等位图字幕和直接烧录在画面里的字幕当前不做 OCR，存在音轨时自动提取 WAV 并进入 ASR，不能把“探测到字幕轨”误解为“字幕文本已识别”。字幕完整性门槛只用于决定是否启用 ASR，是工程启发式，不是字幕准确率或完整性的认证；字幕不合格、解码失败或缺失时会自动兜底，不需要重新创建 Run。

ASR 是自动语音识别，即把音频中的语音转成文本，不是人声分离，也不负责区分谁在说话。无合格字幕时，Worker 提取单声道 16kHz PCM WAV，在一次性 ASR 子进程内执行 Silero VAD，然后通过 OpenAI 兼容接口严格串行上传本地派生 WAV。普通 VAD 区间各自成为独立窗口；只有单个连续语音区间超过 10 分钟时，才按 1 秒重叠均衡拆分。当前只输出段级 `SpeechSegment`，不提供词级对齐、说话人分析或音频事件。

创建 Run 时可提供热词和核心上下文。系统按“核心上下文在前、空格连接的热词在后”确定性合并为云端 Whisper `prompt`，只影响 ASR 兜底，不会改写已提取字幕。两者都是识别偏置而不是强制替换规则，错误提示可能降低准确率；prompt、API Key 和请求头不会写入日志、快照或评测报告。

ASR 阶段仍在受监督的一次性子进程中执行，以隔离 FFmpeg、VAD 和网络调用故障。每个成功窗口会立即发布独立 JSON 缓存；同一 Run 重试时只补传失败窗口，全部成功后再发布整段 ASR 快照。云端最终失败会使整个 ASR 阶段失败，不会发布空转写或部分成功结果。上传 WAV 是可再生临时产物，窗口完成、失败、超时或取消后都会清理。

云端 ASR 只接受本地文件的 `multipart/form-data` 上传，当前实现不接受 OSS 音频 URL。一个 ASR 阶段内窗口请求严格串行；单 Worker 部署下不会并发上传窗口，多 Worker 部署则不提供跨进程全局限流。

质量评测会单独生成提示效果伴随报告，只比较同一授权媒体的 `NONE/CORRECT` 成对 ASR 结果，并报告术语召回率及 CER/WER 差值。它不进入现有发布硬门槛；失败预测、空术语、任一端不是 ASR 或没有合格配对时均为 `NOT_RUN`。字幕命中不能证明热词或核心上下文有效。

## 知识文档生产流程

Worker 在本地完成媒体探测、音频转写、镜头与关键帧提取；章节视觉模型接收授权章节的多张 JPEG，文本模型根据转写与视觉观察生成带证据引用的知识文档。生产结果只写入 Schema 3.0；读取旧 Run 时返回 `RESULT_SCHEMA_UNSUPPORTED`，需要重新创建 Run。


## 基础开发环境

```bash
export UV_CACHE_DIR="$PWD/.codex/uv-cache"
uv sync --extra dev
.venv/bin/pytest
```

语音与视觉可选依赖必须按实际组件显式安装；语音组只保留 Silero VAD、Torch 和 Torchaudio，不下载本地 Whisper 权重：

```bash
uv sync --extra dev --extra speech --extra vision --extra evaluation
```

生产 API、Worker、生产流水线和诊断入口启动时都要求完整的 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `OPENAI_MODEL`。Base URL 必须是 HTTPS 的 OpenAI 兼容 `/v1` 根路径，客户端会自行追加 `/audio/transcriptions`；不要把完整转写端点填入 Base URL。真实 API Key 只写入被 Git 忽略的本地 `.env`，不得提交到源码、示例、测试或报告。

任何缺少模型、外部凭据或授权评测素材的质量项都必须报告 `NOT_RUN`，不能用 mock 结果冒充通过。

## 启动 API 与 Worker

API 和 Worker 必须同时运行；API 只创建持久任务，Worker 才负责领取和处理：

```bash
.venv/bin/uvicorn video_demo.main:app --host 127.0.0.1 --port 8000
.venv/bin/video-demo-worker
```

启动时 API 与 Worker 会使用 `.database-migration.lock` 串行升级运行时 SQLite。该机制仅支持
macOS/Linux 的本地文件系统；运行目录和数据库不得放在 NFS 等网络文件系统上，也不要依赖
网络文件锁提供等价安全性。

两个进程启动后，浏览器访问 `http://127.0.0.1:8000/`，选择一个本地视频并点击
“开始处理”。页面会自动完成上传、创建任务、查询处理状态和展示结果，内部固定使用
`tenant-demo / app-demo / kb-demo` 作为本地演示作用域；现有 API 的调用方式不变。

如果页面长时间停留在等待状态，请先确认 Worker 进程仍在运行。模型或外部服务凭据不完整时，
Worker 会按现有错误语义结束任务，页面只展示后端返回的结果，不会用前端模拟处理成功。

排查任务领取时可运行一次领取尝试：

```bash
.venv/bin/video-demo-worker --once --worker-id local-debug-worker
```

缺少 ffmpeg/ffprobe、Silero 运行依赖或外部服务凭据时，Worker 会以稳定错误码失败关闭，真实媒体、五语质量与性能验收保持 `NOT_RUN`。当前工作区已包含 FFmpeg/ffprobe 6.0，无需重复下载。

## 唯一评测 CLI

所有评测入口统一由 `video_demo.evaluation.cli` 提供。Secret 只能通过环境变量注入，命令行不接受 Token、API Key、凭据或任意文件路径参数：

```bash
EVALUATION_RUN_ID=eval_20260820_001
.venv/bin/python -m video_demo.evaluation.cli preflight --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli media --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli live --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli quality predict --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli quality score --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli durability --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli final --evaluation-run-id "$EVALUATION_RUN_ID"
```

退出码固定为：`0=全部 PASS`、`1=至少一项 FAIL`、`2=没有 FAIL 但存在 NOT_RUN`、`3=验收器配置或证据损坏`。stdout 只包含状态、工作区相对报告路径和稳定原因。

质量评测分两阶段：`quality predict` 通过真实产品 API、Worker 和查询接口生成预测；人工审阅者在 `.codex/video-rag-demo/eval/judgments/<evaluation_run_id>/` 放置完整判断后，`quality score` 才会从已绑定预测和人工判断重算指标，评分阶段不调用模型。数据、授权和目录格式见 [.codex/video-rag-demo/eval/README.md](.codex/video-rag-demo/eval/README.md)。

清理命令是显式破坏性操作：

```bash
.venv/bin/python -m video_demo.evaluation.cli cleanup --evaluation-run-id "$EVALUATION_RUN_ID"
```

它会先写 `.codex/video-rag-demo/eval/cleanup/<evaluation_run_id>.json`，再删除该评测运行、确定性派生阶段以及经严格证据和数据库绑定的产品 run；不会删除普通对象、其他评测运行或其他作用域。证据损坏、符号链接或活跃标记存在时拒绝清理。
