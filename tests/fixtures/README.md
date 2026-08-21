# 测试素材规则

本目录只保存无版权、已授权或已脱敏的最小测试 fixture。录制的外部服务响应必须删除正文、凭据、Token、请求签名和可识别个人信息；fixture 只能证明协议解析，不能作为真实质量通过证据。

真实媒体与评测素材不放在 `tests/fixtures/`。工作区生成媒体仅允许写入 `.codex/video-rag-demo/eval/generated/<运行 ID>/`，它只能证明真实 codec、进程、OpenCV 和场景检测链，绝不是授权数据集、质量、服务或耐久门禁的证据。生成媒体分为四类：`normal_audio/`（正常音频）、`no_audio/`（无音轨）、`rotation/`（旋转元数据）和 `vfr/`（可变帧率）。每类目录只能属于同一个运行 ID；运行 ID 必须是单一路径组件，禁止绝对路径、`..`、符号链接和目录逃逸。

授权五语素材和两段 30 分钟耐久样本放入 `.codex/video-rag-demo/eval/`，并由 Manifest 记录媒体与标注 SHA-256。授权 Manifest 不得引用 `generated/` 下的任何媒体，即使文件摘要、授权记录和样本数量都看似合法。

缺少工作区 FFmpeg/ffprobe、模型、凭据或授权素材时，相应集成、质量和性能结论必须为 `NOT_RUN`，不得用 mock、fake、合成文本或 pytest skip 代替真实通过。
