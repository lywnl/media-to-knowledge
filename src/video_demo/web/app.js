"use strict";

(() => {
  const SCOPE = Object.freeze({
    tenantId: "tenant-demo",
    applicationId: "app-demo",
    knowledgeBaseId: "kb-demo",
  });
  const POLL_INTERVAL_MS = 2000;
  const MAX_CONSECUTIVE_POLL_FAILURES = 5;
  const WORKER_HINT_AFTER_MS = 30000;
  const SUCCESS_STATUSES = new Set(["SUCCEEDED", "PARTIAL_SUCCEEDED"]);
  const FAILURE_STATUSES = new Set(["FAILED", "CANCELLED"]);
  const RETRYABLE_HTTP_STATUSES = new Set([408, 429, 502, 503, 504]);
  const MIME_BY_EXTENSION = Object.freeze({
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
  });
  const STAGE_LABELS = Object.freeze({
    REGISTER: "登记任务",
    PROBE: "读取媒体信息",
    TRANSCODE: "转换媒体",
    EVIDENCE_PREP: "并行准备语音与镜头证据",
    CHAPTER_PLAN: "规划知识章节",
    FRAME_SEARCH: "搜索章节关键帧",
    VISUAL_EVIDENCE: "理解章节画面",
    CHAPTER_WRITE: "撰写章节",
    DOCUMENT_ASSEMBLY: "组装知识文档",
    RESULT: "整理结果",
  });

  const form = document.querySelector("#upload-form");
  const fileInput = document.querySelector("#video-file");
  const fileDetail = document.querySelector("#file-detail");
  const submitButton = document.querySelector("#submit-button");
  const submitButtonLabel = submitButton.querySelector("span");
  const statusPanel = document.querySelector("#status-panel");
  const statusMessage = document.querySelector("#status-message");
  const statusDetail = document.querySelector("#status-detail");
  const trackSegments = [...document.querySelectorAll(".process-track span")];
  const errorMessage = document.querySelector("#error-message");
  const resultPanel = document.querySelector("#result-panel");
  const resultContent = document.querySelector("#result-content");
  const historyStatus = document.querySelector("#history-status");
  const historyList = document.querySelector("#history-list");
  const refreshHistoryButton = document.querySelector("#refresh-history");
  const documentOverview = document.querySelector("#document-overview");
  const documentKeyPoints = document.querySelector("#document-key-points");
  const documentToc = document.querySelector("#document-toc");
  const documentChapters = document.querySelector("#document-chapters");
  const downloadStatus = document.querySelector("#download-status");
  let activeController = null;
  let activeKeyframeUrls = [];
  let keyframeController = null;
  let documentRenderVersion = 0;

  class RequestError extends Error {
    constructor(message, code = null, httpStatus = null) {
      super(message);
      this.name = "RequestError";
      this.code = code;
      this.httpStatus = httpStatus;
    }
  }

  class TerminalRunError extends Error {
    constructor(message) {
      super(message);
      this.name = "TerminalRunError";
    }
  }

  fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    fileDetail.textContent = file
      ? `${file.name} · ${formatBytes(file.size)}`
      : "尚未选择文件";
    clearError();
  });

  refreshHistoryButton.addEventListener("click", () => loadHistory());
  loadHistory();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const [file] = fileInput.files;
    if (!file) {
      renderError(new RequestError("请先选择一个本地视频文件"));
      fileInput.focus();
      return;
    }

    activeController?.abort();
    const controller = new AbortController();
    activeController = controller;
    resetOutput();
    setProcessing(true);

    try {
      updateStatus("正在上传本地视频", "文件将上传到本机后端", 0);
      const videoObject = await uploadVideo(file, controller.signal);
      updateStatus("视频已上传，正在创建任务", "对象已登记", 1);
      const createdRun = await createRun(videoObject.object_ref, controller.signal);
      const completedRun = await waitForCompletion(createdRun.run_id, controller.signal);
      updateStatus("处理完成，正在读取结果", "正在读取结构化知识文档", 3);
      const result = await fetchResult(completedRun.run_id, controller.signal);
      await renderResult(result, { ...completedRun, original_filename: videoObject.original_filename });
      await loadHistory();
      updateStatus("处理完成", "知识文档已准备就绪", 4);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      updateStatus("处理未完成", "请根据下方提示检查后重新处理", 0);
      renderError(error);
    } finally {
      if (activeController === controller) {
        activeController = null;
        setProcessing(false);
      }
    }
  });

  window.addEventListener("beforeunload", () => {
    activeController?.abort();
    keyframeController?.abort();
    revokeKeyframeUrls();
  });

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {
        "X-Tenant-Id": SCOPE.tenantId,
        "X-Application-Id": SCOPE.applicationId,
        ...options.headers,
      },
    });
    const responseText = await response.text();
    let payload = null;
    if (responseText) {
      try {
        payload = JSON.parse(responseText);
      } catch {
        payload = null;
      }
    }
    if (!response.ok) {
      const code = payload?.error?.code ?? null;
      const message = payload?.error?.message ?? `请求失败（HTTP ${response.status}）`;
      throw new RequestError(message, code, response.status);
    }
    if (payload === null) {
      throw new RequestError("后端返回了无法解析的内容", null, response.status);
    }
    return payload;
  }

  async function uploadVideo(file, signal) {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    const declaredMime = MIME_BY_EXTENSION[extension];
    if (!declaredMime) {
      throw new RequestError("请选择 MP4、MOV、MKV 或 WebM 视频文件");
    }
    const body = new FormData();
    const uploadFile = file.type === declaredMime
      ? file
      : new File([file], file.name, { type: declaredMime });
    body.append("file", uploadFile, uploadFile.name);
    return requestJson(
      `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-objects`,
      { method: "POST", body, signal },
    );
  }

  async function createRun(objectRef, signal) {
    return requestJson(
      `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          object_ref: objectRef,
          idempotency_key: `web-${crypto.randomUUID()}`,
          language_hints: [],
        }),
        signal,
      },
    );
  }

  async function waitForCompletion(runId, signal) {
    const startedAt = Date.now();
    let consecutiveFailures = 0;
    const url = `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs/${encodeURIComponent(runId)}`;

    while (true) {
      try {
        const run = await requestJson(url, { signal });
        consecutiveFailures = 0;
        const elapsedMs = Date.now() - startedAt;
        updateRunStatus(run, elapsedMs);
        if (SUCCESS_STATUSES.has(run.status)) {
          return run;
        }
        if (FAILURE_STATUSES.has(run.status)) {
          const code = run.error_code ? `：${run.error_code}` : "";
          throw new TerminalRunError(
            `后端任务已${run.status === "CANCELLED" ? "取消" : "失败"}${code}`,
          );
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }
        if (error instanceof TerminalRunError || !isRetryablePollingError(error)) {
          throw error;
        }
        consecutiveFailures += 1;
        if (consecutiveFailures >= MAX_CONSECUTIVE_POLL_FAILURES) {
          throw new RequestError("连续多次无法查询处理状态，请检查 API 后重试");
        }
        updateStatus(
          "暂时无法连接后端，正在重试",
          `第 ${consecutiveFailures}/${MAX_CONSECUTIVE_POLL_FAILURES} 次查询失败`,
          1,
        );
      }
      await delay(POLL_INTERVAL_MS, signal);
    }
  }

  async function fetchResult(runId, signal) {
    return requestJson(
      `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs/${encodeURIComponent(runId)}/result`,
      { signal },
    );
  }

  async function loadHistory() {
    historyStatus.textContent = "正在读取历史记录";
    try {
      const payload = await requestJson(
        `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs`,
      );
      renderHistory(payload.items ?? []);
      historyStatus.textContent = payload.items?.length ? "" : "还没有历史解析记录";
    } catch (error) {
      historyList.replaceChildren();
      historyStatus.textContent = error instanceof Error
        ? `历史记录读取失败：${error.message}`
        : "历史记录读取失败";
    }
  }

  function renderHistory(items) {
    const fragment = document.createDocumentFragment();
    items.forEach((item) => {
      const button = element("button", "history-item");
      button.type = "button";
      button.append(
        element("span", "history-item-name", item.original_filename),
        element("span", "history-item-meta", `${formatHistoryStatus(item)} · ${formatDate(item.created_at)}`),
      );
      button.addEventListener("click", () => openHistoryItem(item));
      fragment.append(button);
    });
    historyList.replaceChildren(fragment);
  }

  async function openHistoryItem(item) {
    clearError();
    if (!SUCCESS_STATUSES.has(item.status)) {
      updateStatus("该任务尚未完成", `${item.original_filename} · ${formatHistoryStatus(item)}`, 1);
      return;
    }
    try {
      updateStatus("正在读取历史解析文本", item.original_filename, 3);
      const result = await fetchResult(item.run_id);
      await renderResult(result, item);
    } catch (error) {
      renderError(error);
    }
  }

  function formatHistoryStatus(item) {
    if (item.status === "SUCCEEDED") return "已完成";
    if (item.status === "PARTIAL_SUCCEEDED") return "部分完成";
    if (item.status === "FAILED") return "失败";
    if (item.status === "CANCELLED") return "已取消";
    return item.status === "RUNNING" ? `处理中：${STAGE_LABELS[item.current_stage] ?? item.current_stage}` : "排队中";
  }

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "时间未知" : date.toLocaleString("zh-CN", { hour12: false });
  }

  function isRetryablePollingError(error) {
    if (error instanceof TypeError) {
      return true;
    }
    return error instanceof RequestError
      && error.httpStatus !== null
      && RETRYABLE_HTTP_STATUSES.has(error.httpStatus);
  }

  function updateRunStatus(run, elapsedMs) {
    const isPending = run.status === "PENDING";
    const stage = STAGE_LABELS[run.current_stage] ?? run.current_stage;
    const workerHint = elapsedMs >= WORKER_HINT_AFTER_MS && isPending
      ? " · 若长时间无进展，请确认 Worker 已启动"
      : "";
    updateStatus(
      isPending ? "任务正在等待 Worker" : `正在${stage}`,
      `${run.status} · ${stage} · 已等待 ${formatDuration(elapsedMs)}${workerHint}`,
      isPending ? 1 : 2,
    );
  }

  function updateStatus(message, detail, completedSegments) {
    statusPanel.hidden = false;
    statusMessage.textContent = message;
    statusDetail.textContent = detail;
    trackSegments.forEach((segment, index) => {
      segment.classList.toggle("is-complete", index < completedSegments);
      segment.classList.toggle("is-active", index === completedSegments && completedSegments < 4);
    });
  }

  async function renderResult(result, run) {
    const version = ++documentRenderVersion;
    revokeKeyframeUrls();
    keyframeController?.abort();
    keyframeController = new AbortController();
    downloadStatus.textContent = "";
    const evidenceById = await fetchEvidence(run.run_id, result, keyframeController.signal);
    if (version !== documentRenderVersion) return;

    renderDocumentOverview(result.summary, run);
    renderDocumentToc(result.sections ?? [], result.chapters ?? []);
    const chapterList = element("ol", "segment-list");
    const keyframePromises = new Map();
    result.chapters.forEach((chapter, chapterIndex) => {
      const item = element("li", "segment-card");
      item.id = `chapter-${chapterIndex + 1}`;
      item.append(
        element("span", "timecode", timeRange(chapter.start_ms, chapter.end_ms)),
        element("h4", null, chapter.title),
        element("p", "segment-summary", chapter.summary_zh),
      );
      const renderedFrames = new Set();
      chapter.body_blocks.forEach((block) => {
        item.append(renderBodyBlock(block, evidenceById, chapter, run.run_id, keyframePromises, renderedFrames));
      });
      appendUncertainties(item, chapter, evidenceById);
      [...new Set(chapter.selected_keyframe_refs)].forEach((evidenceRef) => {
        const keyframeId = keyframeIdForEvidenceRef(evidenceRef, evidenceById);
        if (!keyframeId || renderedFrames.has(keyframeId)) return;
        renderedFrames.add(keyframeId);
        item.append(renderKeyframeFigure(chapter.title, run.run_id, keyframeId, keyframePromises));
      });
      chapterList.append(item);
    });
    documentChapters.replaceChildren(chapterList);
    if (run.status === "PARTIAL_SUCCEEDED" && run.warning_codes?.length > 0) {
      documentChapters.prepend(element("p", "warning-message", `部分处理完成：${run.warning_codes.join("、")}`));
    }
    resultContent.replaceChildren(
      documentOverview,
      documentKeyPoints,
      documentToc,
      createDownloadLink(run.run_id),
      downloadStatus,
      documentChapters,
    );
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ block: "start" });
  }

  function renderDocumentOverview(summary, run) {
    const card = element("article", "result-summary");
    card.append(
      element("p", "result-source", run.original_filename ? `视频文件：${run.original_filename}` : "知识文档"),
      element("h3", null, summary.title),
      element("p", null, summary.overview_zh),
    );
    documentOverview.replaceChildren(card);
    const points = element("section", "key-points");
    points.append(element("h3", "result-section-title", "关键结论"));
    const list = element("ul");
    (summary.key_points ?? []).forEach((point) => list.append(element("li", null, point.text)));
    points.append(list);
    documentKeyPoints.replaceChildren(points);
  }

  function renderDocumentToc(sections, chapters) {
    documentToc.replaceChildren(element("h3", "result-section-title", "目录"));
    const list = element("ol");
    chapters.forEach((chapter, index) => {
      const link = element("a", null, `${index + 1}. ${chapter.title}`);
      link.href = `#chapter-${index + 1}`;
      const item = element("li");
      item.append(link, element("span", "toc-time", timeRange(chapter.start_ms, chapter.end_ms)));
      list.append(item);
    });
    if (sections.length > 0) {
      documentToc.append(element("p", "toc-sections", sections.map((section) => section.title).join(" · ")));
    }
    documentToc.append(list);
  }

  function createDownloadLink(runId) {
    const link = element("a", "document-download", "下载 Markdown");
    link.href = `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs/${encodeURIComponent(runId)}/document`;
    link.addEventListener("click", (event) => downloadMarkdown(event, link.href));
    return link;
  }

  async function downloadMarkdown(event, url) {
    event.preventDefault();
    downloadStatus.textContent = "正在准备 Markdown 下载";
    try {
      const response = await fetch(url, { headers: scopeHeaders() });
      if (!response.ok) throw new RequestError(`下载失败（HTTP ${response.status}）`, null, response.status);
      const blobUrl = URL.createObjectURL(await response.blob());
      const link = element("a");
      link.href = blobUrl;
      link.download = "knowledge-note.md";
      link.click();
      URL.revokeObjectURL(blobUrl);
      downloadStatus.textContent = "Markdown 已准备下载";
    } catch (error) {
      downloadStatus.textContent = `下载 Markdown失败：${error instanceof Error ? error.message : "请求失败"}`;
    }
  }

  function renderBodyBlock(block, evidenceById, chapter, runId, keyframePromises, renderedFrames) {
    switch (block.block_type) {
      case "PARAGRAPH": return element("p", "chapter-body", block.text);
      case "BULLET_LIST": {
        const list = element("ul", "chapter-body chapter-body--list");
        block.items.forEach((item) => list.append(element("li", null, item)));
        return list;
      }
      case "QUOTE": return element("blockquote", "chapter-body chapter-body--quote", block.text);
      case "CODE": {
        const pre = element("pre", "chapter-body chapter-body--code");
        pre.append(element("code", null, block.code));
        return pre;
      }
      case "TABLE": return renderTableBlock(block);
      case "FORMULA": {
        const formula = element("div", "chapter-body chapter-body--formula");
        formula.append(element("code", null, block.latex), element("p", null, block.explanation));
        return formula;
      }
      case "VISUAL": {
        const visual = element("div", "chapter-body chapter-body--visual");
        visual.append(element("p", null, block.caption || "视觉补充"));
        const observation = evidenceById.get(block.visual_observation_ref);
        const frameRefs = visualFrameRefs(block, observation, evidenceById);
        frameRefs.forEach((keyframeId) => {
          if (renderedFrames.has(keyframeId)) return;
          renderedFrames.add(keyframeId);
          visual.append(renderKeyframeFigure(chapter.title, runId, keyframeId, keyframePromises));
        });
        return visual;
      }
      default: return element("p", "chapter-body", "未识别的正文块");
    }
  }

  function renderTableBlock(block) {
    const table = element("table", "chapter-body chapter-body--table");
    const head = element("thead");
    const headRow = element("tr");
    block.columns.forEach((column) => headRow.append(element("th", null, column)));
    head.append(headRow);
    const body = element("tbody");
    block.rows.forEach((row) => {
      const rowNode = element("tr");
      row.forEach((cell) => rowNode.append(element("td", null, cell)));
      body.append(rowNode);
    });
    table.append(head, body);
    return table;
  }

  function visualFrameRefs(block, observation, evidenceById) {
    if (!observation) return [];
    const byContent = new Map();
    [...(observation.content_blocks ?? []), ...(observation.visual_facts ?? [])].forEach((content) => {
      const id = content.visual_content_id ?? content.visual_fact_id;
      if (id) byContent.set(id, content.source_keyframe_refs ?? []);
    });
    const mapped = [...new Set(block.visual_content_refs.flatMap((ref) => byContent.get(ref) ?? []))];
    const refs = mapped.length > 0 ? mapped : [...new Set(observation.keyframe_refs ?? [])];
    return refs.map((ref) => keyframeIdForEvidenceRef(ref, evidenceById)).filter(Boolean);
  }

  function keyframeIdForEvidenceRef(evidenceRef, evidenceById) {
    const evidence = evidenceById.get(evidenceRef);
    return evidence?.evidence_type === "KEYFRAME" ? evidence.keyframe_id : null;
  }

  function appendUncertainties(parent, chapter, evidenceById) {
    const uncertainties = [];
    chapter.body_blocks.forEach((block) => {
      if (block.block_type !== "VISUAL") return;
      const observation = evidenceById.get(block.visual_observation_ref);
      (observation?.uncertainties ?? []).forEach((value) => uncertainties.push(value));
    });
    if (uncertainties.length === 0) return;
    const note = element("aside", "uncertainty-note");
    note.append(element("strong", null, "不确定性"));
    const list = element("ul");
    [...new Set(uncertainties)].forEach((value) => list.append(element("li", null, value)));
    note.append(list);
    parent.append(note);
  }

  async function fetchEvidence(runId, result, signal) {
    const hasVisual = result.chapters.some((chapter) => chapter.body_blocks.some((block) => block.block_type === "VISUAL"));
    if (!hasVisual) return new Map();
    const items = [];
    let cursor = null;
    try {
      do {
        const suffix = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
        const page = await requestJson(`/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs/${encodeURIComponent(runId)}/evidence?limit=100${suffix}`, { signal });
        items.push(...(page.items ?? []));
        cursor = page.next_cursor;
      } while (cursor);
    } catch {
      return new Map();
    }
    return new Map(items.map((item) => [item.evidence_id, item]));
  }

  function renderKeyframeFigure(chapterTitle, runId, keyframeId, keyframePromises) {
    const figure = element("figure", "chapter-keyframe");
    const image = element("img");
    image.alt = `${chapterTitle}关键画面`;
    const caption = element("figcaption", null, "正在加载关键画面");
    figure.append(image, caption);
    if (!keyframePromises.has(keyframeId)) {
      keyframePromises.set(keyframeId, loadKeyframe(image, caption, runId, keyframeId));
    } else {
      keyframePromises.get(keyframeId).then((url) => { if (url) image.src = url; });
    }
    return figure;
  }

  async function loadKeyframe(image, caption, runId, keyframeId) {
    try {
      const response = await fetch(
        `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs/${encodeURIComponent(runId)}/keyframes/${encodeURIComponent(keyframeId)}/content`,
        { headers: scopeHeaders(), signal: keyframeController?.signal },
      );
      if (!response.ok) throw new RequestError(`关键画面加载失败（HTTP ${response.status}）`, null, response.status);
      const url = URL.createObjectURL(await response.blob());
      activeKeyframeUrls.push(url);
      image.src = url;
      caption.textContent = "关键画面";
      return url;
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return null;
      image.hidden = true;
      caption.textContent = "关键画面暂时无法加载，正文仍可阅读";
      image.closest(".chapter-keyframe")?.classList.add("is-failed");
      return null;
    }
  }

  function scopeHeaders() {
    return { "X-Tenant-Id": SCOPE.tenantId, "X-Application-Id": SCOPE.applicationId };
  }

  function revokeKeyframeUrls() {
    activeKeyframeUrls.forEach((url) => URL.revokeObjectURL(url));
    activeKeyframeUrls = [];
  }

  function renderTags(values) {
    const group = element("div", "tag-group");
    [...new Set(values)].forEach((value) => group.append(element("span", "tag", value)));
    return group;
  }

  function element(tagName, className = null, text = null) {
    const node = document.createElement(tagName);
    if (className) {
      node.className = className;
    }
    if (text !== null && text !== undefined) {
      node.textContent = String(text);
    }
    return node;
  }

  function renderError(error) {
    const message = error instanceof Error ? error.message : "处理过程中发生未知错误";
    const code = error instanceof RequestError && error.code ? `（${error.code}）` : "";
    errorMessage.textContent = `${message}${code}`;
    errorMessage.hidden = false;
  }

  function clearError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
  }

  function resetOutput() {
    clearError();
    resultContent.replaceChildren();
    resultPanel.hidden = true;
    statusPanel.hidden = true;
  }

  function setProcessing(isProcessing) {
    fileInput.disabled = isProcessing;
    submitButton.disabled = isProcessing;
    submitButtonLabel.textContent = isProcessing ? "正在处理" : "开始处理";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    const units = ["KB", "MB", "GB", "TB"];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
  }

  function formatTime(milliseconds) {
    const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return hours > 0
      ? [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":")
      : [minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
  }

  function formatDuration(milliseconds) {
    return `${Math.max(0, Math.floor(milliseconds / 1000))} 秒`;
  }

  function timeRange(startMs, endMs) {
    return `${formatTime(startMs)} — ${formatTime(endMs)}`;
  }

  function delay(milliseconds, signal) {
    return new Promise((resolve, reject) => {
      const rejectOnAbort = () => {
        window.clearTimeout(timeoutId);
        reject(new DOMException("请求已取消", "AbortError"));
      };
      const timeoutId = window.setTimeout(() => {
        signal.removeEventListener("abort", rejectOnAbort);
        resolve();
      }, milliseconds);
      signal.addEventListener("abort", rejectOnAbort, { once: true });
    });
  }
})();
