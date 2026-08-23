# 视频理解到 retrieval_text Demo

这是一个独立的 Python 3.11 + FastAPI 视频理解 Demo。目标是对独立视频执行音频、画面和多模态理解，输出带证据引用的 `VIDEO_SEGMENT`、`VIDEO_SUMMARY`、`retrieval_text` 和 `retrieval_hash`。

当前状态：Demo 主链已完成。上传、可靠 Worker、ffprobe/FFmpeg、字幕优先与 ASR 兜底、可选的 WhisperX 中文对齐、镜头/关键帧、百度 OCR、确定性融合、结构化结果、`retrieval_text`、证据分页和关键帧查询均已接通。生产 Qwen 链路使用私有 OSS 完整视频签名 URL：完整代理视频只上传一次，`qwen3-vl-flash` 只发起一次全片请求；模型返回按时间排序的粗粒度视觉语义和全片摘要，程序再映射到本地冻结的精确窗口和证据。实施账本见[计划](./.codex/plans/2026-08-17-video-understanding-retrieval-text-implementation.md)。

提交时必须准确区分两类结论：

- Demo 产品链与机器一致性 smoke：已通过。详见[双视频产品链报告](./.codex/reports/two-video-demo-chain-20260821.md)和[双视频质量 smoke](./.codex/reports/two-video-quality-smoke-20260821.md)。
- 正式五语质量、pyannote 说话人分析和 M1 两段耐久：仍为 `NOT_RUN` 或外部前置条件不足，不能用中文 Demo 代替。Qwen 公网完整视频视觉理解已真实通过；同一个完整公网 URL 不保证按 `start_ms/end_ms` 截取片段，因此不能冒充精确片段输入。综合验证见[报告](./.codex/reports/qwen-visual-understanding-20260821.md)，公网 URL 对账见[对账报告](./.codex/reports/qwen-remote-url-reconciliation-20260821.md)。

`示例视频.mp4` 的 921,484 毫秒全片已完成一次真实本地产品链，产生 601 条 ASR、5,743 条对齐词、329 个关键帧、329 条 OCR 和 128 个场景证据；首次运行因旧 Qwen 细窗口协议降级。修复后的严格 Qwen 与融合验收复用了该次运行的 47,881,184 字节完整代理和已落库证据，没有重新执行约 53 分钟的本地 ASR/OCR。最新严格验收用唯一对象键完成一次 OSS PUT 上传、一次 Qwen 全片请求、32 个细窗口合法映射和 10 个融合检索片段；每个窗口都绑定该窗口的全部本地证据，首段 `retrieval_text` 为 975 字符，全部检索哈希重算一致。融合片段数会随 Qwen 返回的粗语义分组变化，不是固定协议。当前尚未再次从上传接口启动并持久化一条全新 Run。报告见[全片严格验收](./.codex/reports/qwen-production-full-video-validation-20260821.json)。

Qwen 的视觉职责与证据边界：完整视频视觉报告可识别人物、场景、账号页、软件界面、关键事件和画面文字；模型观察可能误读账号名、数字或字幕。原始可定位画面文字仍以百度 OCR 证据为权威，语音以 ASR/WhisperX 为权威。需要精确片段视觉理解时，必须传实际派生短片（本地 clip/Data URI 或该短片自己的公网 URL），不得只给完整视频 URL 再依赖时间字段让供应商自动 seek。

## 视频转文本策略

视频文本解析按“内嵌文本字幕优先、ASR 兜底”执行：`ffprobe` 先发现容器内字幕流，系统只把 `subrip`、`ass`、`ssa`、`webvtt` 和 `mov_text` 当作可解析文本字幕。候选字幕经 UTF-8 解析、大小和 cue 数量限制、时间轴及启发式完整性检查后，以独立的 `SUBTITLE_CUE` 证据输出；命中时不生成 `audio.wav`，也不启动 VAD、faster-whisper、WhisperX、pyannote 或 YAMNet。

PGS、DVD Subtitle 等位图字幕和直接烧录在画面里的字幕当前不做 OCR，存在音轨时自动提取 WAV 并进入 ASR，不能把“探测到字幕轨”误解为“字幕文本已识别”。字幕完整性门槛只用于决定是否启用 ASR，是工程启发式，不是字幕准确率或完整性的认证；字幕不合格、解码失败或缺失时会自动兜底，不需要重新创建 Run。

ASR 是自动语音识别，即把音频中的语音转成文本，不是人声分离，也不负责区分谁在说话。创建 Run 时 `speech_enrichment_mode` 默认是 `text`：无合格字幕时只执行 VAD、LID 和 faster-whisper，输出可用于检索的 ASR 文本，不执行增强模型。只有显式传入 `speech_enrichment_mode="full"`，才会在复用 ASR 快照后继续运行 WhisperX 词级时间对齐、pyannote 说话人分析和 YAMNet 音频事件识别。省略该字段的旧客户端仍能创建 Run，但新版本不再默认产生词级、说话人和音频事件证据；依赖这些字段的调用方必须显式迁移到 `full`。字幕路径不会运行 ASR 或增强模型，因此对应质量指标记为“不适用（`NOT_RUN`）”；这不代表模型零错误，也不能把字幕 cue 伪装成 ASR 词时间或置信度。

创建 Run 时可提供热词和核心上下文。热词用于提高人名、术语和产品名的识别概率，核心上下文用于提供视频主题先验；它们分别映射到 faster-whisper 的 `hotwords` 和 `initial_prompt`，只影响 ASR 兜底，不会改写已提取字幕，也不会传给语言探测。两者都是识别偏置而不是强制替换规则，错误提示可能降低准确率。

`text` 或 `full` 的 ASR 阶段都在受监督的一次性子进程中加载重语音模型，以隔离原生崩溃和内存故障；每次真正执行 ASR 都有模型冷启动成本。字幕命中、完整语音快照命中或 ASR 快照命中会跳过相应加载；快照只服务同一 Run 的失败重试，保证已成功的 ASR 不被无意义重复执行，不是跨视频或跨 Run 的全局缓存。`full` 在 ASR 快照命中但增强快照未命中时只运行增强阶段，不重复转写。

质量评测会单独生成提示效果伴随报告，只比较同一授权媒体的 `NONE/CORRECT` 成对 ASR 结果，并报告术语召回率及 CER/WER 差值。它不进入现有发布硬门槛；失败预测、空术语、任一端不是 ASR 或没有合格配对时均为 `NOT_RUN`。字幕命中不能证明热词或核心上下文有效。

## Qwen 的私有 OSS 全片中转

生产 Worker 现在支持同一条产品链完成“本地解析 + 单次全片视觉理解”：源视频仍由现有上传接口进入本地对象存储；Worker 在本地执行 FFmpeg、ASR、关键帧、场景检测和百度 OCR，并生成连续且每段不超过 30 秒的冻结理解窗口；完整 MP4 代理视频只上传到私有阿里云 OSS 一次，Qwen 只接收这一条全片的一小时签名 URL，并在一个结构化逻辑请求中返回按时间排序的粗粒度语义数组和全片摘要。程序按模型实际返回的组数等分全片，将粗语义映射回全部本地细窗口；精确时间和 ASR/OCR/关键帧/场景引用始终由本地程序绑定，再生成 `retrieval_text`。公共上传、创建 Run 和结果查询接口没有变化。

Worker 需要以下环境变量：

```bash
VIDEO_DEMO_OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
VIDEO_DEMO_OSS_BUCKET=your-private-bucket
VIDEO_DEMO_OSS_ACCESS_KEY_ID=your-rotated-access-key-id
VIDEO_DEMO_OSS_ACCESS_KEY_SECRET=your-rotated-access-key-secret
VIDEO_DEMO_OSS_PREFIX=video-demo/qwen-clips
VIDEO_DEMO_OSS_SIGNED_URL_TTL_SECONDS=3600
```

部署前必须在 Bucket 上完成两项设置：Bucket 禁止公共读；为 `video-demo/qwen-clips/` 前缀配置对象创建后一天自动删除的生命周期规则。Qwen 调用成功或失败后，应用会按本次发布返回的精确对象键立即发送 DELETE；Worker 崩溃、断电或 DELETE 失败时由生命周期规则兜底。AccessKey、Secret 和签名 URL 不写入结果、证据或报告；建议使用仅具备目标前缀读写权限的 RAM 子账号，并定期轮换凭据。

配置 Qwen 时必须同时完整配置 OSS；严格生产模式拒绝使用本地 Base64/Data URI 发送视频。Qwen 收到的是 FFmpeg 生成的完整 MP4 代理视频，而且生产 Run 不执行能力探测、逐窗口 `understand_segment()` 或第二次摘要请求。窗口时间只来自本地冻结边界，不依赖“完整视频 URL + `start_ms/end_ms`”让供应商自动 seek；Qwen 响应不是合法 JSON、顶层结构不符、粗语义为空或数量超过本地窗口数时，整次理解失败，不发布部分模型结果。Qwen 不生成时间和证据 ID，因此无法把模型引用误绑定到窗口外。显式 Demo 降级只能使用本地证据生成确定性语义，并产生 `DEMO_DEGRADED_QWEN`。

## 基础开发环境

```bash
export UV_CACHE_DIR="$PWD/.codex/uv-cache"
uv sync --extra dev
.venv/bin/pytest
```

重模型依赖必须按实际组件显式安装，基础 API 环境不会自动下载模型：

```bash
uv sync --extra dev --extra speech --extra vision --extra audio-events --extra evaluation
```

`faster-whisper` 首次真实调用会将 `large-v3` 的五个必需文件平铺到
`.codex/video-rag-demo/models/faster-whisper/`，下载缓存独立放在
`.codex/video-rag-demo/cache/huggingface/`。模型完整后只从本地目录加载，不再依赖网络；
`config.json`、`model.bin`、`preprocessor_config.json`、`tokenizer.json` 或
`vocabulary.json` 任一缺失、为空或为符号链接时，预检都会报告模型不可用。

`speech` 依赖组显式约束 `torch/torchaudio 2.8.x`、`pyannote.audio 4.x`、
`WhisperX 3.4.2` 和 `huggingface-hub 0.x`：该组合匹配
`speaker-diarization-community-1` 的 PLDA 配置，并保留当前 WhisperX 对齐 API。
macOS 上 pyannote 通过内存 PCM 波形执行，不依赖 TorchCodec 动态链接系统 FFmpeg。
WhisperX alignment 模型只缓存到
`.codex/video-rag-demo/models/whisperx/<language>/`；上游无法对齐且未返回
`start/end/score`，或返回的完整词时间不属于任何原 ASR segment 时，该词项会被跳过并产生
稳定警告，不会伪造或裁剪词时间；字段不完整或数值非法仍以稳定模型错误失败关闭。

`audio-events` 依赖组使用支持 NumPy 2.x 和 Apple Silicon 的 TensorFlow 2.20，
并将 Setuptools 约束为 `>=80.9,<82`：`tensorflow-hub 0.16.1` 仍导入
`pkg_resources`，而 Setuptools 82 已移除该 API。
YAMNet SavedModel 和 521 项官方类别表固定在
`.codex/video-rag-demo/models/yamnet/`，来源与摘要见该目录的 `SOURCE.md`。
生产代码只在 TensorFlow Hub 导入边界屏蔽这一条已知上游弃用警告，不吞其他依赖警告。

任何缺少模型、外部凭据或授权评测素材的质量项都必须报告 `NOT_RUN`，不能用 mock 结果冒充通过。

## 启动 API 与 Worker

API 和 Worker 必须同时运行；API 只创建持久任务，Worker 才负责领取和处理：

```bash
.venv/bin/uvicorn video_demo.main:app --host 127.0.0.1 --port 8000
.venv/bin/video-demo-worker
```

两个进程启动后，浏览器访问 `http://127.0.0.1:8000/`，选择一个本地视频并点击
“开始处理”。页面会自动完成上传、创建任务、查询处理状态和展示结果，内部固定使用
`tenant-demo / app-demo / kb-demo` 作为本地演示作用域；现有 API 的调用方式不变。

如果页面长时间停留在等待状态，请先确认 Worker 进程仍在运行。模型或外部服务凭据不完整时，
Worker 会按现有错误语义结束任务，页面只展示后端返回的结果，不会用前端模拟处理成功。

排查任务领取时可运行一次领取尝试：

```bash
.venv/bin/video-demo-worker --once --worker-id local-debug-worker
```

缺少 ffmpeg/ffprobe、重模型或外部服务凭据时，Worker 会以稳定错误码失败关闭，真实媒体、五语质量与性能验收保持 `NOT_RUN`。当前工作区已包含 FFmpeg/ffprobe 6.0，无需重复下载。

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
