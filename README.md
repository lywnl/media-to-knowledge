# 视频知识文档 Demo

这是一个独立的 Python 3.11 + FastAPI 视频理解 Demo。目标是对独立视频执行音频和多模态理解，输出 Schema `4.2.0` 结构化知识文档和确定性 Markdown。

当前状态：Demo 主链已完成。上传、可靠 Worker、ffprobe/FFmpeg、字幕优先、Silero VAD、云端 Whisper、章节时间点抽帧、章节多图 VLM 和结构化 Markdown 文档均已接通。FFmpeg 是唯一的视频抽帧组件，关键帧只作为 VLM 的内部输入，前端仅展示文本；项目不生成 RAG 检索文本，也不建立向量索引。生产链只使用云 ASR、文本 LLM 和章节视觉模型三套配置，结果 Schema 为 4.2.0。

提交或验收时必须准确区分两类结论：

- Demo 产品链与机器一致性 smoke：已通过。正式五语云端 ASR、代表性视觉质量、人工文档 rubric 和 M1 两段耐久是否通过，必须以对应授权素材和评测报告为准；缺少真实前置条件时明确报告 `NOT_RUN/INCONCLUSIVE`，不能用 Demo 样本或配置单测替代。

章节视觉模型只对授权章节的本地 JPEG 做观察，结果可能误读画面文字、数字或界面状态；画面文字质量由独立的代表性视觉评测量化，不能把模型返回文本或 Schema 合法性当作准确率。语音以字幕或云端 ASR 的 `SpeechSegment` 为准，视觉观察通过证据 ID 与章节正文绑定。

## 视频转文本策略

视频文本解析按“内嵌文本字幕优先、云端 ASR 兜底”执行：`ffprobe` 先发现容器内字幕流，系统只把 `subrip`、`ass`、`ssa`、`webvtt` 和 `mov_text` 当作可解析文本字幕。候选字幕经 UTF-8 解析、大小和 cue 数量限制、时间轴及启发式完整性检查后，以独立的 `SUBTITLE_CUE` 证据输出；命中时不生成 `audio.wav`，也不启动 VAD 或调用云端 ASR。

PGS、DVD Subtitle 等位图字幕和直接烧录在画面里的字幕当前不作为文本字幕解析；存在音轨时自动提取 WAV 并进入 ASR，不能把“探测到字幕轨”误解为“字幕文本已识别”。字幕完整性门槛只用于决定是否启用 ASR，是工程启发式，不是字幕准确率或完整性的认证；字幕不合格、解码失败或缺失时会自动兜底，不需要重新创建 Run。

ASR 是自动语音识别，即把音频中的语音转成文本，不是人声分离，也不负责区分谁在说话。无合格字幕时，Worker 提取单声道 16kHz PCM WAV，在一次性 ASR 子进程内执行 Silero VAD，然后通过 OpenAI 兼容接口严格串行上传本地派生 WAV。普通 VAD 区间各自成为独立窗口；只有单个连续语音区间超过 10 分钟时，才按 1 秒重叠均衡拆分。当前只输出段级 `SpeechSegment`，不提供词级对齐、说话人分析或音频事件。

创建 Run 时可提供热词和核心上下文。系统按“核心上下文在前、空格连接的热词在后”确定性合并为云端 Whisper `prompt`，只影响 ASR 兜底，不会改写已提取字幕。两者都是识别偏置而不是强制替换规则，错误提示可能降低准确率；prompt、API Key 和请求头不会写入日志、快照或评测报告。

ASR 阶段仍在受监督的一次性子进程中执行，以隔离 FFmpeg、VAD 和网络调用故障。每个成功窗口会立即发布独立 JSON 缓存；同一 Run 重试时只补传失败窗口，全部成功后再发布整段 ASR 快照。云端最终失败会使整个 ASR 阶段失败，不会发布空转写或部分成功结果。上传 WAV 是可再生临时产物，窗口完成、失败、超时或取消后都会清理。

云端 ASR 只接受本地文件的 `multipart/form-data` 上传，不接受远程对象 URL。一个 ASR 阶段内窗口请求严格串行；单 Worker 部署下不会并发上传窗口，多 Worker 部署则不提供跨进程全局限流。

质量评测会单独生成提示效果伴随报告，只比较同一授权媒体的 `NONE/CORRECT` 成对 ASR 结果，并报告术语召回率及 CER/WER 差值。它不进入现有发布硬门槛；失败预测、空术语、任一端不是 ASR 或没有合格配对时均为 `NOT_RUN`。字幕命中不能证明热词或核心上下文有效。

## 知识文档生产流程

Worker 在本地完成媒体探测、音频转写和基础片段准备；章节时间点由 ASR 语义锚点与章节中点确定，FFmpeg 抽取 JPEG 后交给章节视觉模型，文本模型根据转写与视觉观察生成 Markdown 知识文档。关键帧和证据用于内部可追溯校验，不作为前端展示内容。生产结果只写入 Schema 4.2；读取旧 Run 时返回 `RESULT_SCHEMA_UNSUPPORTED`，需要重新创建 Run。

正式生产使用三套相互独立的模型配置：云端 ASR（`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`）、文本 LLM（`VIDEO_DEMO_TEXT_LLM_*`）和章节视觉 VLM（`VIDEO_DEMO_VLM_*`）。章节 VLM 只接收本地授权章节的 2～4 张图片，最终文本制品和结果均为 Schema `4.2.0`。用户通过 `/document` 获取唯一 Markdown 文本制品；`/result`、`/evidence` 与关键帧内容接口仅供内部校验和诊断使用，前端不会调用后两类接口。

质量评测通过统一 CLI 执行：

```bash
.venv/bin/python -m video_demo.evaluation.cli quality visual --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli quality visual-resolution --evaluation-run-id "$EVALUATION_RUN_ID"
.venv/bin/python -m video_demo.evaluation.cli quality score --evaluation-run-id "$EVALUATION_RUN_ID"
```

`quality visual` 生成代表性视觉文字事实，`quality visual-resolution` 只给出 1280/1920 的证据对照建议；两者不会自动修改生产默认分辨率。质量报告同时区分视觉文字准确率、关键字段召回率，以及文档自动指标 `chapter_time_coverage`、`claim_evidence_rate`、`markdown_json_consistency`。标题/摘要相关性、视觉观察是否补充转写、重复和冲突判断必须来自绑定样本与制品摘要的人工 rubric；没有完整人工判断时显示 `NOT_RUN`，不会用 LLM 自评或 Schema 合法性冒充通过。视觉集缺少授权样本、视觉质量报告或参考分母为零时同样保留 `NOT_RUN/INCONCLUSIVE`，真实调用失败则保留 `FAIL`。

耐久门禁默认验证已授权的两段 30 分钟、至少 1920×1080 的真实素材；生产配置默认允许的最大视频时长为 `7_200_000ms`（恰好两小时），超过该值在执行前拒绝。没有真实授权两小时素材时，系统只报告 `NOT_RUN`，不会循环短视频、mock 或把配置单测写成 PASS。

Task 11B 已移除历史对象存储、传统文字识别、全片视频推理和 2.0 生产隔离链。升级后旧 Run 不兼容，必须重新执行 `quality predict`、视觉质量和文档质量评测；未实际运行的代表性视觉集、人工 rubric 和两小时耐久均不得写成 PASS。


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

## 启动 API

视频任务由 FastAPI 进程内的总调度器自动领取和处理；音频、图片任务仍可使用各自独立 Worker：

```bash
.venv/bin/video-demo-api
```

API 默认监听 `127.0.0.1:7999`；如需直接使用 Uvicorn，等价命令为
`.venv/bin/uvicorn video_demo.main:app --host 127.0.0.1 --port 7999`。

启动时 API 会使用 `.database-migration.lock` 串行升级运行时 SQLite。该机制仅支持
macOS/Linux 的本地文件系统；运行目录和数据库不得放在 NFS 等网络文件系统上，也不要依赖
网络文件锁提供等价安全性。

启动 API 后，浏览器访问 `http://127.0.0.1:7999/`，选择一个本地视频并点击
“开始处理”。页面会自动完成上传、创建任务、查询处理状态和展示结果，内部固定使用
`tenant-demo / app-demo / kb-demo` 作为本地演示作用域；现有 API 的调用方式不变。

如果页面长时间停留在等待状态，请先查看 API 日志中的调度器阶段状态。模型或外部服务凭据不完整时，
调度器会按现有错误语义结束任务，页面只展示后端返回的结果，不会用前端模拟处理成功。

缺少 ffmpeg/ffprobe、Silero 运行依赖或外部服务凭据时，视频任务会以稳定错误码失败关闭，真实媒体、五语质量与性能验收保持 `NOT_RUN`。当前工作区已包含 FFmpeg/ffprobe 6.0，无需重复下载。

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
