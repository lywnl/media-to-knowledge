"use strict";

(() => {
  const SCOPE = Object.freeze({
    tenantId: "tenant-demo",
    applicationId: "app-demo",
    knowledgeBaseId: "kb-demo",
  });
  const POLL_INTERVAL_MS = 2000;
  const MAX_CONSECUTIVE_POLL_FAILURES = 5;
  const SUCCESS_STATUSES = new Set(["SUCCEEDED", "PARTIAL_SUCCEEDED"]);
  const FAILURE_STATUSES = new Set(["FAILED", "CANCELLED"]);
  const RETRYABLE_HTTP_STATUSES = new Set([408, 429, 502, 503, 504]);
  const MEDIA_CONFIG = Object.freeze({
    VIDEO: Object.freeze({
      kind: "VIDEO",
      label: "视频",
      description: "视频会进入视频理解流水线，输出章节化 Markdown。",
      accept: ".mp4,.mov,.mkv,.webm",
      uploadPath: `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-objects`,
      runPath: `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs`,
      mimeByExtension: Object.freeze({
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
      }),
    }),
    AUDIO: Object.freeze({
      kind: "AUDIO",
      label: "音频",
      description: "音频会执行转写与章节写作，不包含画面补充。",
      accept: ".mp3,.wav,.m4a,.aac,.ogg,.flac,.webm",
      uploadPath: `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/audio-objects`,
      runPath: `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/audio-understanding-runs`,
      mimeByExtension: Object.freeze({
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
      }),
    }),
    IMAGE: Object.freeze({
      kind: "IMAGE",
      label: "图片",
      description: "图片会执行单图视觉理解，输出图片 Markdown。",
      accept: ".jpg,.jpeg,.png,.webp",
      uploadPath: `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/image-objects`,
      runPath: `/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/image-understanding-runs`,
      mimeByExtension: Object.freeze({
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
      }),
    }),
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
    SPEECH: "转写音频内容",
    VLM: "理解图片内容",
  });

  const form = document.querySelector("#upload-form");
  const fileInput = document.querySelector("#video-file");
  const fileDetail = document.querySelector("#file-detail");
  const mediaKindButtons = [...document.querySelectorAll("[data-media-kind]")];
  const uploadTitle = document.querySelector("#upload-title");
  const uploadDescription = document.querySelector("#upload-description");
  const fileActionLabel = document.querySelector("#file-action-label");
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
  const documentToc = document.querySelector("#document-toc");
  const documentChapters = document.querySelector("#document-chapters");
  const downloadStatus = document.querySelector("#download-status");
  let currentMediaKind = "VIDEO";
  let activeController = null;

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

  function currentMediaConfig() {
    return MEDIA_CONFIG[currentMediaKind] ?? MEDIA_CONFIG.VIDEO;
  }

  function setMediaKind(kind) {
    if (!MEDIA_CONFIG[kind]) return;
    currentMediaKind = kind;
    const media = currentMediaConfig();
    mediaKindButtons.forEach((button) => {
      const isActive = button.dataset.mediaKind === media.kind;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
    uploadTitle.textContent = `选择${media.label}`;
    uploadDescription.textContent = media.description;
    fileActionLabel.textContent = `选择本地${media.label}`;
    fileInput.accept = media.accept;
    fileInput.value = "";
    fileDetail.textContent = "尚未选择文件";
    resetOutput();
    clearError();
  }

  function mediaResultPath(media, runId, suffix) {
    return `${media.runPath}/${encodeURIComponent(runId)}${suffix}`;
  }

  mediaKindButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (activeController) return;
      setMediaKind(button.dataset.mediaKind);
      loadHistory();
    });
  });

  setMediaKind(currentMediaKind);

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
    const media = currentMediaConfig();
    const [file] = fileInput.files;
    if (!file) {
      renderError(new RequestError(`请先选择一个本地${media.label}文件`));
      fileInput.focus();
      return;
    }

    activeController?.abort();
    const controller = new AbortController();
    activeController = controller;
    resetOutput();
    setProcessing(true);

    try {
      updateStatus(`正在上传本地${media.label}`, "文件将上传到本机后端", 0);
      const mediaObject = await uploadMedia(file, media, controller.signal);
      updateStatus(`${media.label}已上传，正在创建任务`, "对象已登记", 1);
      const createdRun = await createRun(mediaObject.object_ref, controller.signal, media);
      const completedRun = await waitForCompletion(createdRun.run_id, controller.signal, media);
      updateStatus("处理完成，正在读取结果", "正在读取结构化知识文档", 3);
      const result = await fetchResult(completedRun.run_id, controller.signal, media);
      await renderResult(result, { ...completedRun, original_filename: mediaObject.original_filename }, media);
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

  async function uploadMedia(file, media, signal) {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    const declaredMime = media.mimeByExtension[extension];
    if (!declaredMime) {
      throw new RequestError(`请选择${media.label}支持的文件格式（${media.accept}）`);
    }
    const body = new FormData();
    const uploadFile = file.type === declaredMime
      ? file
      : new File([file], file.name, { type: declaredMime });
    body.append("file", uploadFile, uploadFile.name);
    return requestJson(media.uploadPath, { method: "POST", body, signal });
  }

  async function createRun(objectRef, signal, media = currentMediaConfig()) {
    return requestJson(media.runPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        object_ref: objectRef,
        idempotency_key: `web-${crypto.randomUUID()}`,
        language_hints: [],
      }),
      signal,
    });
  }

  async function waitForCompletion(runId, signal, media = currentMediaConfig()) {
    const startedAt = Date.now();
    let consecutiveFailures = 0;
    const url = `${media.runPath}/${encodeURIComponent(runId)}`;

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

  async function fetchResult(runId, signal, media = currentMediaConfig()) {
    return requestJson(mediaResultPath(media, runId, "/result"), { signal });
  }

  async function loadHistory() {
    const media = currentMediaConfig();
    historyStatus.textContent = "正在读取历史记录";
    try {
      const payload = await requestJson(media.runPath);
      renderHistory(payload.items ?? [], media);
      historyStatus.textContent = payload.items?.length ? "" : "还没有历史解析记录";
    } catch (error) {
      historyList.replaceChildren();
      historyStatus.textContent = error instanceof Error
        ? `历史记录读取失败：${error.message}`
        : "历史记录读取失败";
    }
  }

  function renderHistory(items, media = currentMediaConfig()) {
    const fragment = document.createDocumentFragment();
    items.forEach((item) => {
      const button = element("button", "history-item");
      button.type = "button";
      button.append(
        element("span", "history-item-name", item.original_filename),
        element("span", "history-item-meta", `${formatHistoryStatus(item)} · ${formatDate(item.created_at)}`),
      );
      button.addEventListener("click", () => openHistoryItem(item, media));
      fragment.append(button);
    });
    historyList.replaceChildren(fragment);
  }

  async function openHistoryItem(item, media = currentMediaConfig()) {
    clearError();
    if (!SUCCESS_STATUSES.has(item.status)) {
      updateStatus("该任务尚未完成", `${item.original_filename} · ${formatHistoryStatus(item)}`, 1);
      return;
    }
    try {
      updateStatus("正在读取历史解析文本", item.original_filename, 3);
      const result = await fetchResult(item.run_id, undefined, media);
      await renderResult(result, item, media);
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
    const schedulerHint = isPending
      ? " · 视频任务由 API 内置调度器处理"
      : "";
    updateStatus(
      isPending ? "任务正在等待 API 调度" : `正在${stage}`,
      `${run.status} · ${stage} · 已等待 ${formatDuration(elapsedMs)}${schedulerHint}`,
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

  function renderResult(result, run, media = currentMediaConfig()) {
    downloadStatus.textContent = "";

    if (media.kind === "IMAGE") {
      renderImageResult(result, run);
    } else if (media.kind === "AUDIO") {
      renderAudioResult(result, run);
    } else {
      renderVideoResult(result, run);
    }
    if (run.status === "PARTIAL_SUCCEEDED" && run.warning_codes?.length > 0) {
      documentChapters.prepend(element("p", "warning-message", `部分处理完成：${run.warning_codes.join("、")}`));
    }
    resultContent.replaceChildren(
      documentOverview,
      documentToc,
      createDownloadLink(run.run_id, media),
      downloadStatus,
      documentChapters,
    );
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ block: "start" });
  }

  function renderVideoResult(result, run) {
    renderDocumentOverview(result.summary, run, "视频");
    renderChapterResult(result.chapters ?? []);
  }

  function renderAudioResult(result, run) {
    renderDocumentOverview(result.summary, run, "音频");
    renderChapterResult(result.chapters ?? []);
  }

  function renderImageResult(result, run) {
    const document = result.document;
    const card = element("article", "result-summary");
    card.append(
      element("p", "result-source", run.original_filename ? `图片文件：${run.original_filename}` : "图片知识文档"),
      element("h3", null, document.title),
      element("p", null, document.overview_zh || "未提供核心概览。"),
    );
    documentOverview.replaceChildren(card);
    documentToc.replaceChildren();
    const content = element("article", "image-document");
    content.append(element("h3", "result-section-title", "图片内容"));
    document.content_blocks?.forEach((block) => {
      if (block.content_type === "DESCRIPTION") {
        content.append(element("p", "chapter-body", block.text));
        return;
      }
      content.append(
        element("h4", null, block.content_type),
        element("p", "chapter-body", block.text),
      );
    });
    if (document.claims?.length) {
      content.append(element("h3", "result-section-title", "关键结论"));
      const claims = element("ul");
      document.claims.forEach((claim) => claims.append(element("li", null, claim.text)));
      content.append(claims);
    }
    documentChapters.replaceChildren(content);
  }

  function renderChapterResult(chapters) {
    renderDocumentToc(chapters);
    const chapterList = element("ol", "segment-list");
    chapters.forEach((chapter, chapterIndex) => {
      const item = element("li", "segment-card");
      item.id = `chapter-${chapterIndex + 1}`;
      item.append(
        element("span", "timecode", timeRange(chapter.start_ms, chapter.end_ms)),
        element("h4", null, chapter.title),
        element("p", "segment-summary", chapter.summary_zh),
      );
      chapter.body_blocks?.forEach((block) => item.append(renderBodyBlock(block)));
      appendChapterClaims(item, chapter);
      chapterList.append(item);
    });
    documentChapters.replaceChildren(chapterList);
  }

  function renderDocumentOverview(summary, run, mediaLabel = "视频") {
    const card = element("article", "result-summary");
    card.append(
      element("p", "result-source", run.original_filename ? `${mediaLabel}文件：${run.original_filename}` : "知识文档"),
      element("h3", null, summary.title),
      element("p", null, summary.overview_zh || "未提供核心概览。"),
    );
    documentOverview.replaceChildren(card);
  }

  function renderDocumentToc(chapters) {
    documentToc.replaceChildren(element("h3", "result-section-title", "目录"));
    const list = element("ol");
    chapters.forEach((chapter, index) => {
      const link = element("a", null, `${index + 1}. ${chapter.title}`);
      link.href = `#chapter-${index + 1}`;
      const item = element("li");
      item.append(link, element("span", "toc-time", timeRange(chapter.start_ms, chapter.end_ms)));
      list.append(item);
    });
    documentToc.append(list);
  }

  function createDownloadLink(runId, media = currentMediaConfig()) {
    const link = element("a", "document-download", "下载 Markdown");
    link.href = mediaResultPath(media, runId, "/document");
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

  function renderBodyBlock(block) {
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

  function appendChapterClaims(parent, chapter) {
    const claims = (chapter.claims ?? []).filter((claim) => claim.text);
    if (claims.length === 0) return;
    const section = element("section", "chapter-claims");
    section.append(element("h5", null, "本章结论"));
    const list = element("ul");
    claims.forEach((claim) => list.append(element("li", null, claim.text)));
    section.append(list);
    parent.append(section);
  }

  function scopeHeaders() {
    return { "X-Tenant-Id": SCOPE.tenantId, "X-Application-Id": SCOPE.applicationId };
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
