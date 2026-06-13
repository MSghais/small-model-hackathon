import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.14.0/+esm";

const $ = (sel) => document.querySelector(sel);
const SLIDE_PIPELINE_STEPS = [
  "Load language model",
  "Gather lesson sources",
  "Generate slide outline",
  "Build PPTX, DOCX, and HTML exports",
];

const state = {
  workspaceTopic: "photosynthesis",
  workspaceSessionId: "",
  workspaceDocIds: [],
  discoveredUrls: [],
  selectedUrls: [],
  slideDiscoveredUrls: [],
  slideSelectedUrls: [],
  researchChatHistory: [],
  debugChatHistory: [],
  voiceMode: "lesson",
  history: [],
  downloads: null,
  client: null,
  progressTimer: null,
  progressStartedAt: null,
  voicePresets: null,
  modelChoices: null,
  recordingTarget: null,
  browserRecorder: null,
  browserRecordChunks: [],
  pendingVoiceAudioPath: null,
  pendingCoachAudioPath: null,
  useBrowserMic: true,
};

function effectiveTopic(local) {
  const localVal = (local || "").trim();
  if (localVal) return localVal;
  return (state.workspaceTopic || "").trim();
}

function selectedWorkspaceDocIds() {
  const boxes = document.querySelectorAll("#workspace-doc-list input[type=checkbox]:checked");
  return [...boxes].map((el) => el.value);
}

function effectiveDocIds(localIds) {
  if (localIds && localIds.length) return localIds;
  const selected = selectedWorkspaceDocIds();
  if (selected.length) return selected;
  return state.workspaceDocIds;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdownLite(text) {
  const safe = escapeHtml(stripMd(text || ""));
  return safe
    .replace(/\n/g, "<br>")
    .replace(/\[(\d+)\]/g, "<sup>[$1]</sup>");
}

function stripMd(text) {
  return String(text).replace(/\*\*/g, "").replace(/`/g, "");
}

function fileUrl(path) {
  if (!path) return "";
  return `/file=${encodeURIComponent(path)}`;
}

function setTracePanel(panelId, data) {
  const panel = $(panelId);
  if (!panel) return;
  const html = data?.trace_html || "";
  if (html) {
    panel.innerHTML = html;
    panel.closest("details")?.classList.remove("hidden");
  } else if (data?.trace_summary || data?.trace_json) {
    const parts = [];
    if (data.trace_summary) {
      parts.push(`<pre class="studio-trace-summary">${escapeHtml(data.trace_summary)}</pre>`);
    }
    if (data.trace_json) {
      parts.push(`<pre class="studio-trace-json">${escapeHtml(data.trace_json)}</pre>`);
    }
    panel.innerHTML = parts.join("");
  }
}

function getIngestWorkflow() {
  return $("#ingest-workflow")?.value || "direct";
}

function syncIngestWorkflowUi() {
  const mode = getIngestWorkflow();
  $("#ingest-discover-row")?.classList.toggle("hidden", mode !== "select");
  $("#ingest-auto-row")?.classList.toggle("hidden", mode !== "auto");
  $("#url-choices-panel")?.classList.toggle(
    "hidden",
    mode !== "select" || !state.discoveredUrls.length
  );
}

function syncSlideSourceUi() {
  const mode = $("#slide-source-mode")?.value || "";
  const isWeb = mode === "web";
  $("#slide-web-workflow-wrap")?.classList.toggle("hidden", !isWeb);
  $("#slide-web-discover-wrap")?.classList.toggle("hidden", !isWeb);
  if (isWeb && $("#slide-search-workflow")?.value === "two_step") {
    $("#slide-url-choices-panel")?.classList.toggle(
      "hidden",
      !state.slideDiscoveredUrls.length
    );
  } else {
    $("#slide-url-choices-panel")?.classList.add("hidden");
  }
}

function syncResearchLayout() {
  syncIngestWorkflowUi();
  syncSlideSourceUi();
  updateResearchDocCount(state.workspaceDocIds?.length || 0);
}

function updateResearchDocCount(count) {
  const badge = $("#research-doc-count");
  if (!badge) return;
  if (!count) {
    badge.classList.add("hidden");
    badge.textContent = "0 docs";
    return;
  }
  badge.classList.remove("hidden");
  badge.textContent = count === 1 ? "1 doc" : `${count} docs`;
}

function openResearchView() {
  document.querySelector('.nav-item[data-view="research"]')?.click();
  window.setTimeout(() => $("#research-question")?.focus(), 80);
}

function getSelectedDiscoveredUrls(listId = "#url-choices-list") {
  const boxes = document.querySelectorAll(`${listId} input[type=checkbox]:checked`);
  return [...boxes].map((el) => el.value);
}

function renderUrlChoices(urls, selected, listId, panelId, urlState) {
  urlState.discovered = urls || [];
  urlState.selected = selected?.length ? selected : [...urlState.discovered];
  const list = $(listId);
  const panel = $(panelId);
  if (!urlState.discovered.length) {
    if (list) list.innerHTML = "";
    panel?.classList.add("hidden");
    return;
  }
  list.innerHTML = urlState.discovered
    .map((url) => {
      const checked = urlState.selected.includes(url) ? "checked" : "";
      const label = url.length > 72 ? `${url.slice(0, 69)}…` : url;
      return `<label class="url-choice-item"><input type="checkbox" value="${escapeHtml(url)}" ${checked} /><span title="${escapeHtml(url)}">${escapeHtml(label)}</span></label>`;
    })
    .join("");
  panel?.classList.remove("hidden");
}

function renderResearchUrlChoices(urls, selected) {
  state.discoveredUrls = urls || [];
  state.selectedUrls = selected?.length ? selected : [...state.discoveredUrls];
  const list = $("#url-choices-list");
  const panel = $("#url-choices-panel");
  if (!state.discoveredUrls.length) {
    list.innerHTML = "";
    panel?.classList.add("hidden");
    return;
  }
  list.innerHTML = state.discoveredUrls
    .map((url) => {
      const checked = state.selectedUrls.includes(url) ? "checked" : "";
      const label = url.length > 72 ? `${url.slice(0, 69)}…` : url;
      return `<label class="url-choice-item"><input type="checkbox" value="${escapeHtml(url)}" ${checked} /><span title="${escapeHtml(url)}">${escapeHtml(label)}</span></label>`;
    })
    .join("");
  list.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.addEventListener("change", syncUrlSelectAll);
  });
  syncUrlSelectAll();
  if (getIngestWorkflow() === "select") panel?.classList.remove("hidden");
}

function renderSlideUrlChoices(urls, selected) {
  state.slideDiscoveredUrls = urls || [];
  state.slideSelectedUrls = selected?.length ? selected : [...state.slideDiscoveredUrls];
  renderUrlChoices(
    urls,
    selected,
    "#slide-url-choices-list",
    "#slide-url-choices-panel",
    { discovered: state.slideDiscoveredUrls, selected: state.slideSelectedUrls }
  );
  syncSlideSourceUi();
}

function syncUrlSelectAll() {
  const boxes = [...document.querySelectorAll("#url-choices-list input[type=checkbox]")];
  const selectAll = $("#url-select-all");
  if (!selectAll || !boxes.length) return;
  const checkedCount = boxes.filter((b) => b.checked).length;
  selectAll.checked = checkedCount === boxes.length;
  selectAll.indeterminate = checkedCount > 0 && checkedCount < boxes.length;
  state.selectedUrls = getSelectedDiscoveredUrls();
}

function applyIngestResult(data) {
  $("#ingest-status").textContent = stripMd(data.status || "Ingest complete.");
  state.workspaceSessionId = data.session_id || state.workspaceSessionId;
  $("#workspace-session").value = state.workspaceSessionId;
  $("#documents-panel").innerHTML =
    data.documents_html || '<p class="studio-empty-docs">No documents indexed yet.</p>';
  renderWorkspaceDocList(data.documents || []);
  setTracePanel("#research-trace-panel", data);
  updateResearchRagBadge();
  updateResearchDocCount((data.documents || []).length);
}

async function discoverSources() {
  const topic = effectiveTopic("");
  if (!topic) {
    showError("Set a workspace topic before discovering sources.");
    return;
  }
  const data = await callApi("discover_sources", [topic, state.workspaceSessionId]);
  $("#ingest-status").textContent = stripMd(data.status || "Discovery complete.");
  renderResearchUrlChoices(data.urls || [], data.selected_urls || data.urls || []);
  if (data.session_id) {
    state.workspaceSessionId = data.session_id;
    $("#workspace-session").value = data.session_id;
  }
  setTracePanel("#research-trace-panel", data);
  await refreshWorkspaceSessions(state.workspaceSessionId);
}

async function discoverSlideSources() {
  const topic = effectiveTopic($("#lesson-topic")?.value);
  if (!topic) {
    showError("Set a topic before discovering sources.");
    return;
  }
  const data = await callApi("discover_sources", [topic, state.workspaceSessionId]);
  renderSlideUrlChoices(data.urls || [], data.selected_urls || data.urls || []);
}

async function autoSearchIngest() {
  const topic = effectiveTopic("");
  if (!topic) {
    showError("Set a workspace topic before auto-ingest.");
    return;
  }
  const data = await callApi("auto_search_ingest", [topic, state.workspaceSessionId]);
  applyIngestResult(data);
  state.discoveredUrls = [];
  state.selectedUrls = [];
  renderResearchUrlChoices([], []);
  await refreshWorkspaceSessions(state.workspaceSessionId);
}

async function ingestSources({ urlsText = "", selectedUrls = [], pendingFiles = null } = {}) {
  const topic = effectiveTopic("");
  const workflow = getIngestWorkflow();
  let selected = selectedUrls;
  if (workflow === "select") selected = getSelectedDiscoveredUrls();
  const pasted = workflow === "direct" ? urlsText : urlsText || $("#ingest-url").value.trim();
  const paths = [];
  const files = pendingFiles || $("#ingest-file").files;
  if (files?.length) {
    for (const file of files) {
      const b64 = await fileToBase64(file);
      const saved = await callApi("save_upload", [file.name, b64]);
      paths.push(saved.path);
    }
  }
  if (!pasted && !selected.length && !paths.length) {
    showError("Add URLs, select suggested sources, or upload a file — then ingest.");
    return;
  }
  const data = await callApi("ingest_sources", [
    topic,
    state.workspaceSessionId,
    pasted,
    selected,
    paths,
  ]);
  applyIngestResult(data);
  if (pasted) $("#ingest-url").value = "";
  if (files?.length) $("#ingest-file").value = "";
  await refreshWorkspaceSessions(state.workspaceSessionId);
}

function renderResearchChat() {
  const container = $("#research-chat-messages");
  if (!state.researchChatHistory.length) {
    container.innerHTML =
      '<p class="research-chat-empty">Ingest sources, then ask questions — answers include citations from your library.</p>';
    return;
  }
  container.innerHTML = state.researchChatHistory
    .map((msg) => {
      const role = msg.role === "user" ? "user" : "assistant";
      const body = renderMarkdownLite(msg.content || "");
      return `<div class="research-chat-bubble research-chat-${role}"><div class="research-chat-role">${role === "user" ? "You" : "ResearchMind"}</div><div class="research-chat-body">${body}</div></div>`;
    })
    .join("");
  container.scrollTop = container.scrollHeight;
}

function renderDebugChat() {
  const container = $("#debug-chat-messages");
  if (!state.debugChatHistory.length) {
    container.innerHTML =
      '<p class="research-chat-empty">Send a message to test the active local model.</p>';
    return;
  }
  container.innerHTML = state.debugChatHistory
    .map(([user, assistant]) => {
      return `<div class="research-chat-bubble research-chat-user"><div class="research-chat-role">You</div><div class="research-chat-body">${renderMarkdownLite(user)}</div></div><div class="research-chat-bubble research-chat-assistant"><div class="research-chat-role">Model</div><div class="research-chat-body">${renderMarkdownLite(assistant)}</div></div>`;
    })
    .join("");
  container.scrollTop = container.scrollHeight;
}

function updateResearchRagBadge() {
  const badge = $("#research-rag-badge");
  if (!badge) return;
  const nDocs = (state.workspaceDocIds || []).length;
  const selected = selectedWorkspaceDocIds().length;
  if (selected) badge.textContent = `RAG · ${selected} doc(s)`;
  else if (nDocs) badge.textContent = `RAG · ${nDocs} in session`;
  else badge.textContent = "RAG · corpus";
}

async function askResearchQuestion() {
  const question = $("#research-question").value.trim();
  if (!question) {
    showError("Enter a question.");
    return;
  }
  const docIds = effectiveDocIds([]);
  const data = await callApi("research_chat", [
    question,
    state.workspaceSessionId,
    docIds,
    state.researchChatHistory,
  ]);
  state.researchChatHistory = data.history || [];
  renderResearchChat();
  $("#research-question").value = "";
  $("#research-chat-status").textContent = stripMd(data.rag_hint || "");
  setTracePanel("#research-trace-panel", data);
  updateResearchRagBadge();
}

async function sendDebugMessage() {
  const message = $("#debug-message").value.trim();
  if (!message) {
    showError("Enter a message.");
    return;
  }
  const useRag = $("#debug-use-rag").checked;
  const docIds = effectiveDocIds([]);
  const modelKey = $("#debug-model-key")?.value || "";
  const data = await callApi("debug_chat", [
    message,
    state.debugChatHistory,
    useRag,
    state.workspaceSessionId,
    docIds,
    modelKey,
  ]);
  state.debugChatHistory = data.history || [];
  renderDebugChat();
  $("#debug-message").value = "";
  setTracePanel("#debug-trace-panel", data);
}

function updateProjectTitle() {
  const topic = state.workspaceTopic || "";
  const short = topic.split(" for ")[0] || topic || "Project";
  $("#project-title").textContent = short.slice(0, 40);
}

function updateWorkspaceRagHint() {
  const nDocs = selectedWorkspaceDocIds().length;
  const sid = state.workspaceSessionId;
  let hint = "RAG scope: entire indexed corpus (all sessions).";
  if (sid) {
    hint = nDocs
      ? `RAG scope: ${nDocs} selected document(s) in session.`
      : "RAG scope: all documents in session.";
  }
  const el = $("#workspace-rag-hint");
  if (el) el.textContent = hint;
}

async function getClient() {
  if (!state.client) {
    state.client = await Client.connect(window.location.origin);
  }
  return state.client;
}

function setLoading(on) {
  $("#studio-loading").classList.toggle("hidden", !on);
}

function startProgressPanel() {
  const panel = $("#progress-panel");
  const stepsEl = $("#progress-steps");
  panel.classList.remove("hidden");
  state.progressStartedAt = Date.now();
  stepsEl.innerHTML = SLIDE_PIPELINE_STEPS.map(
    (label, index) =>
      `<li data-step="${index}" class="progress-step pending">${label}</li>`
  ).join("");
  $("#progress-log").classList.add("hidden");
  $("#progress-log").textContent = "";
  $("#progress-eta").textContent = "Est. remaining: calculating…";
  updateProgressElapsed();
  if (state.progressTimer) clearInterval(state.progressTimer);
  state.progressTimer = setInterval(updateProgressElapsed, 500);
}

function updateProgressElapsed() {
  if (!state.progressStartedAt) return;
  const elapsed = (Date.now() - state.progressStartedAt) / 1000;
  $("#progress-elapsed").textContent = `Elapsed: ${elapsed.toFixed(1)}s`;
  const eta = estimateRemaining(elapsed);
  $("#progress-eta").textContent =
    eta !== null ? `Est. remaining: ~${Math.max(0, Math.round(eta))}s` : "";
}

function estimateRemaining(elapsed) {
  if (elapsed < 3) return null;
  const stepNodes = [...document.querySelectorAll("#progress-steps .progress-step")];
  const activeIndex = stepNodes.findIndex((node) => node.classList.contains("active"));
  const doneCount = stepNodes.filter((node) => node.classList.contains("done")).length;
  const progress = Math.max((doneCount + (activeIndex >= 0 ? 0.35 : 0)) / stepNodes.length, 0.15);
  return elapsed / progress - elapsed;
}

function markProgressStep(index, status) {
  const node = document.querySelector(`#progress-steps [data-step="${index}"]`);
  if (!node) return;
  node.classList.remove("pending", "active", "done");
  node.classList.add(status);
}

function advanceProgressWhileWaiting() {
  let current = 0;
  markProgressStep(current, "active");
  const timer = setInterval(() => {
    if (!$("#progress-panel") || $("#progress-panel").classList.contains("hidden")) {
      clearInterval(timer);
      return;
    }
    if (current < SLIDE_PIPELINE_STEPS.length - 1) {
      markProgressStep(current, "done");
      current += 1;
      markProgressStep(current, "active");
    }
  }, 9000);
  return timer;
}

function finishProgressPanel(data) {
  if (state.progressTimer) {
    clearInterval(state.progressTimer);
    state.progressTimer = null;
  }
  const stepsEl = $("#progress-steps");
  const traceSteps = data?.progress?.steps || [];
  if (traceSteps.length) {
    stepsEl.innerHTML = traceSteps
      .map((step) => {
        const duration = step.duration_s != null ? ` (${step.duration_s}s)` : "";
        const detail = step.detail ? ` — ${step.detail}` : "";
        return `<li class="progress-step done">${step.label}${duration}${detail}</li>`;
      })
      .join("");
  } else {
    document.querySelectorAll("#progress-steps .progress-step").forEach((node) => {
      node.classList.remove("pending", "active");
      node.classList.add("done");
    });
  }
  if (data?.progress_log) {
    const logEl = $("#progress-log");
    const log = data.progress_log;
    if (/<[a-z][\s\S]*>/i.test(log)) logEl.innerHTML = log;
    else logEl.textContent = stripMd(log);
    logEl.classList.remove("hidden");
  }
  if (data?.elapsed_seconds != null) {
    $("#progress-elapsed").textContent = `Elapsed: ${Number(data.elapsed_seconds).toFixed(1)}s`;
  }
  $("#progress-eta").textContent = "Complete";
  setTracePanel("#slides-trace-panel", data);
}

function showError(msg) {
  const el = $("#studio-error");
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.classList.remove("hidden");
}

function unwrapApiPayload(result) {
  const raw = result?.data ?? result;
  if (Array.isArray(raw)) {
    if (raw.length === 1 && raw[0] !== null && typeof raw[0] === "object") return raw[0];
    return raw;
  }
  return raw;
}

async function callApi(name, args = []) {
  setLoading(true);
  showError("");
  try {
    const client = await getClient();
    const result = await client.predict(`/${name}`, args);
    const data = unwrapApiPayload(result);
    if (data && data.ok === false) throw new Error(data.error || "Request failed");
    return data;
  } catch (err) {
    const message = err?.message || String(err);
    showError(`${message} — try Classic UI at /classic`);
    throw err;
  } finally {
    setLoading(false);
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function uploadFile(file) {
  const b64 = await fileToBase64(file);
  const saved = await callApi("save_upload", [file.name, b64]);
  return saved.path;
}

function renderWorkspaceDocList(docs) {
  const container = $("#workspace-doc-list");
  if (!docs?.length) {
    container.innerHTML = '<p class="status-text">No documents in this session yet.</p>';
    state.workspaceDocIds = [];
    updateWorkspaceRagHint();
    updateResearchDocCount(0);
    return;
  }
  state.workspaceDocIds = docs.map((d) => d.id);
  container.innerHTML = docs
    .map(
      (d) =>
        `<label class="workspace-doc-item"><input type="checkbox" value="${d.id}" checked />${escapeHtml(d.title)}</label>`
    )
    .join("");
  container.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.addEventListener("change", () => {
      updateWorkspaceRagHint();
      updateResearchRagBadge();
    });
  });
  updateWorkspaceRagHint();
  updateResearchRagBadge();
  updateResearchDocCount(docs.length);
}

async function refreshWorkspaceSessions(selectId) {
  const data = await callApi("list_sessions", []);
  const sessions = data.sessions || [];
  const select = $("#workspace-session");
  const current = selectId || state.workspaceSessionId;
  select.innerHTML =
    '<option value="">New session (on ingest)</option>' +
    sessions.map((s) => `<option value="${s.id}">${s.label || s.topic}</option>`).join("");
  if (current && sessions.some((s) => s.id === current)) {
    select.value = current;
    state.workspaceSessionId = current;
  } else {
    const hint = (state.workspaceTopic || "").toLowerCase();
    const match = sessions.find((s) => (s.topic || "").toLowerCase().includes(hint));
    if (match) {
      select.value = match.id;
      state.workspaceSessionId = match.id;
      updateProjectTitle();
    }
  }
}

async function refreshDocuments() {
  const data = await callApi("list_documents", [state.workspaceSessionId]);
  $("#documents-panel").innerHTML =
    data.documents_html || '<p class="studio-empty-docs">No documents indexed yet.</p>';
  if (data.session_id) {
    state.workspaceSessionId = data.session_id;
    $("#workspace-session").value = data.session_id;
  }
  renderWorkspaceDocList(data.documents || []);
  const mem = $("#workspace-memory");
  if (mem && data.memory_markdown) {
    mem.textContent = stripMd(data.memory_markdown);
  }
}

async function initVoicePresets() {
  const data = await callApi("voice_presets", []);
  state.voicePresets = data;
  const langSelect = $("#coach-language");
  const asrSelect = $("#coach-asr");
  if (langSelect) {
    langSelect.innerHTML = (data.languages || [])
      .map((o) => `<option value="${o.value}">${o.label}</option>`)
      .join("");
    langSelect.value = data.default_language || "en";
  }
  if (asrSelect) {
    asrSelect.innerHTML = (data.asr_presets || [])
      .map((o) => `<option value="${o.value}">${o.label}</option>`)
      .join("");
    asrSelect.value = data.default_asr || "";
  }
}

async function initSettings() {
  const data = await callApi("model_choices", []);
  state.modelChoices = data;
  $("#settings-active-model").textContent = `${data.active_label} (${data.active_backend})`;
  $("#settings-voice-stack").textContent = data.voice_stack || "";
  $("#settings-paths").textContent = data.paths || "";
  const status = await callApi("model_status", []);
  $("#settings-status").innerHTML = renderMarkdownLite(status.status_markdown || "");

  const wrap = $("#settings-model-select-wrap");
  const debugWrap = $("#debug-model-wrap");
  const select = $("#settings-model-key");
  const debugSelect = $("#debug-model-key");
  if (data.allow_model_switch && data.choices?.length) {
    wrap?.classList.remove("hidden");
    debugWrap?.classList.remove("hidden");
    const options = data.choices
      .map((c) => `<option value="${c.key}">${c.label}</option>`)
      .join("");
    if (select) {
      select.innerHTML = options;
      select.value = data.active_model;
    }
    if (debugSelect) {
      debugSelect.innerHTML = options;
      debugSelect.value = data.active_model;
    }
  }
}

function openSettingsDrawer() {
  $("#settings-drawer")?.classList.remove("hidden");
  $("#settings-drawer")?.setAttribute("aria-hidden", "false");
}

function closeSettingsDrawer() {
  $("#settings-drawer")?.classList.add("hidden");
  $("#settings-drawer")?.setAttribute("aria-hidden", "true");
}

async function reloadModelFromSettings() {
  const key = $("#settings-model-key")?.value || "";
  const data = await callApi("reload_model", [key]);
  $("#settings-status").innerHTML = renderMarkdownLite(data.status_markdown || "Reloaded.");
}

async function initWorkspace() {
  $("#workspace-topic").value = state.workspaceTopic;
  syncResearchLayout();
  updateProjectTitle();
  updateResearchRagBadge();
  await refreshWorkspaceSessions();
  await refreshDocuments();
  await initVoicePresets();
  await initSettings();
  const recStatus = await callApi("recording_status", []);
  state.useBrowserMic = !recStatus.backend || /unavailable|no capture/i.test(recStatus.message || "");
}

async function ingestUrl() {
  await ingestSources({ urlsText: $("#ingest-url").value.trim() });
}

async function ingestFiles(files) {
  if (!files?.length) return;
  await ingestSources({ pendingFiles: files });
}

async function generateSlides() {
  const topic = effectiveTopic($("#lesson-topic").value);
  const grade = $("#lesson-grade").value;
  const slideCount = Number($("#slide-count").value);
  const useRag = $("#use-rag").checked;
  const docIds = effectiveDocIds([]);
  const sourceMode = $("#slide-source-mode")?.value || "";
  const searchWorkflow = $("#slide-search-workflow")?.value || "two_step";
  const urlsText = $("#slide-urls-text")?.value.trim() || "";
  const selectedUrls = getSelectedDiscoveredUrls("#slide-url-choices-list");

  const filePaths = [];
  const slideFiles = $("#slide-source-files")?.files;
  if (slideFiles?.length) {
    for (const file of slideFiles) {
      filePaths.push(await uploadFile(file));
    }
  }

  startProgressPanel();
  const waitTimer = advanceProgressWhileWaiting();
  let data;
  try {
    data = await callApi("generate_slides", [
      topic,
      grade,
      slideCount,
      state.workspaceSessionId,
      useRag,
      docIds,
      sourceMode,
      searchWorkflow,
      urlsText,
      selectedUrls,
      filePaths,
    ]);
  } catch (_err) {
    $("#progress-eta").textContent = "Failed";
    throw _err;
  } finally {
    clearInterval(waitTimer);
    if (state.progressTimer) {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
    }
  }

  finishProgressPanel(data);
  $("#generate-status").textContent = stripMd(data.status || "Slides generated.");
  const canvasHtml =
    data.canvas_html ||
    (data.preview_html ? `<div class="studio-canvas-inner">${data.preview_html}</div>` : "");
  $("#slide-canvas").innerHTML =
    canvasHtml || '<div class="studio-canvas-empty"><p>Preview unavailable.</p></div>';

  const galleryEl = $("#slide-gallery");
  if (data.gallery_html) {
    galleryEl.innerHTML = data.gallery_html;
    galleryEl.classList.remove("hidden");
  } else if (data.gallery?.length) {
    galleryEl.innerHTML = data.gallery
      .map(
        (path, i) =>
          `<a class="studio-gallery-item" href="${fileUrl(path)}" target="_blank" rel="noopener"><img src="${fileUrl(path)}" alt="Slide ${i + 1}" loading="lazy" /></a>`
      )
      .join("");
    galleryEl.classList.remove("hidden");
  } else {
    galleryEl.classList.add("hidden");
    galleryEl.innerHTML = "";
  }

  state.downloads = data.downloads;
  const dl = $("#downloads");
  if (data.downloads?.pptx) {
    dl.classList.remove("hidden");
    dl.innerHTML = `
      <a href="${fileUrl(data.downloads.pptx)}" download>PPTX</a>
      <a href="${fileUrl(data.downloads.docx)}" download>DOCX</a>
      <a href="${fileUrl(data.downloads.html)}" download>HTML</a>`;
    $("#btn-export").disabled = false;
  }
}

function renderVoiceReply(data) {
  $("#voice-reply").textContent = data.assistant || data.status || "";
  const out = $("#voice-audio-out");
  if (data.voiceout_path) {
    out.innerHTML = `<audio controls src="${fileUrl(data.voiceout_path)}"></audio>`;
  } else {
    out.innerHTML = "";
  }
}

async function sendVoiceTurn() {
  const message = $("#voice-message").value.trim();
  const topic = effectiveTopic("");
  const useRag = $("#use-rag").checked;
  const docIds = effectiveDocIds([]);
  const language = state.voicePresets?.default_language || "en";
  const data = await callApi("teacher_voice_turn", [
    message,
    state.voiceMode,
    topic,
    state.workspaceSessionId,
    useRag,
    state.history,
    docIds,
    language,
    null,
  ]);
  state.history = data.history || [];
  renderVoiceReply(data);
}

async function sendVoiceAudioTurn(audioPath) {
  const topic = effectiveTopic("");
  const useRag = $("#use-rag").checked;
  const docIds = effectiveDocIds([]);
  const language = state.voicePresets?.default_language || "en";
  const asr = state.voicePresets?.default_asr || null;
  const data = await callApi("teacher_voice_audio_turn", [
    audioPath,
    state.voiceMode,
    topic,
    state.workspaceSessionId,
    useRag,
    state.history,
    docIds,
    language,
    asr,
  ]);
  state.history = data.history || [];
  if (data.user_text) $("#voice-message").value = data.user_text;
  renderVoiceReply(data);
}

async function analyzePitchWithPath(audioPath) {
  const language = $("#coach-language")?.value || "en";
  const asr = $("#coach-asr")?.value || null;
  const speakRewrite = $("#coach-speak-rewrite")?.checked || false;
  $("#coach-panel").innerHTML = `
    <div class="studio-coach-panel studio-coach-live">
      <div class="studio-coach-header"><span class="studio-coach-dot"></span>
      <span class="studio-coach-label">Analyzing…</span></div>
    </div>`;
  const data = await callApi("analyze_pitch", [audioPath, language, asr, speakRewrite]);
  $("#coach-panel").innerHTML = data.coach_panel_html || "";
}

async function analyzePitch() {
  let path = state.pendingCoachAudioPath;
  const file = $("#coach-audio").files?.[0];
  if (file) path = await uploadFile(file);
  if (!path) {
    showError("Record or upload audio to analyze.");
    return;
  }
  await analyzePitchWithPath(path);
}

async function startBrowserRecording(statusEl) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  state.browserRecordChunks = [];
  state.browserRecorder = new MediaRecorder(stream);
  state.browserRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) state.browserRecordChunks.push(e.data);
  };
  state.browserRecorder.start();
  if (statusEl) statusEl.textContent = "Recording… click Stop when done.";
}

async function stopBrowserRecording(statusEl) {
  return new Promise((resolve, reject) => {
    const recorder = state.browserRecorder;
    if (!recorder) {
      reject(new Error("No active recording."));
      return;
    }
    recorder.onstop = async () => {
      recorder.stream.getTracks().forEach((t) => t.stop());
      state.browserRecorder = null;
      const blob = new Blob(state.browserRecordChunks, { type: "audio/webm" });
      state.browserRecordChunks = [];
      try {
        const file = new File([blob], "browser_recording.webm", { type: "audio/webm" });
        const path = await uploadFile(file);
        if (statusEl) statusEl.textContent = "Recording saved.";
        resolve(path);
      } catch (err) {
        reject(err);
      }
    };
    recorder.stop();
  });
}

async function startRecording(target, statusEl, startBtn, stopBtn) {
  state.recordingTarget = target;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  if (state.useBrowserMic) {
    try {
      await startBrowserRecording(statusEl);
    } catch (err) {
      startBtn.disabled = false;
      stopBtn.disabled = true;
      showError(`Microphone error: ${err.message}`);
    }
    return;
  }
  try {
    const maxSec = state.voicePresets?.max_seconds || 30;
    const data = await callApi("recording_start", [maxSec]);
    if (statusEl) statusEl.textContent = stripMd(data.status || "Recording…");
  } catch (_err) {
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

async function stopRecording(statusEl, startBtn, stopBtn) {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  let path = null;
  if (state.useBrowserMic) {
    path = await stopBrowserRecording(statusEl);
  } else {
    const data = await callApi("recording_stop", []);
    path = data.path;
    if (statusEl) statusEl.textContent = stripMd(data.status || "Recording saved.");
  }
  if (state.recordingTarget === "voice") state.pendingVoiceAudioPath = path;
  if (state.recordingTarget === "coach") state.pendingCoachAudioPath = path;
  state.recordingTarget = null;
  return path;
}

async function sendVoiceFromRecording() {
  let path = state.pendingVoiceAudioPath;
  const file = $("#voice-audio-upload").files?.[0];
  if (file) path = await uploadFile(file);
  if (!path) {
    showError("Record or upload audio first.");
    return;
  }
  await sendVoiceAudioTurn(path);
}

function bindUi() {
  $("#slide-count").addEventListener("input", (e) => {
    $("#slide-count-val").textContent = e.target.value;
  });

  document.querySelectorAll(".nav-item[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item[data-view]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $(".workspace").dataset.view = btn.dataset.view;
      syncResearchLayout();
      $("#sidebar").classList.remove("open");
    });
  });

  $("#btn-open-settings")?.addEventListener("click", openSettingsDrawer);
  $("#btn-close-settings")?.addEventListener("click", closeSettingsDrawer);
  $("#settings-backdrop")?.addEventListener("click", closeSettingsDrawer);
  $("#btn-reload-model")?.addEventListener("click", () => reloadModelFromSettings().catch(() => {}));

  $("#btn-open-research-view")?.addEventListener("click", openResearchView);
  $("#sidebar-open")?.addEventListener("click", () => $("#sidebar").classList.add("open"));
  $("#sidebar-close")?.addEventListener("click", () => $("#sidebar").classList.remove("open"));

  $("#workspace-topic").addEventListener("input", (e) => {
    state.workspaceTopic = e.target.value.trim();
    updateProjectTitle();
  });

  $("#workspace-session").addEventListener("change", (e) => {
    state.workspaceSessionId = e.target.value;
    refreshDocuments().catch(() => {});
  });

  $("#workspace-refresh-sessions").addEventListener("click", () => {
    refreshWorkspaceSessions(state.workspaceSessionId).catch(() => {});
  });

  $("#btn-ingest-url").addEventListener("click", () => ingestUrl().catch(() => {}));
  $("#ingest-file").addEventListener("change", (e) => ingestFiles(e.target.files).catch(() => {}));
  $("#ingest-workflow")?.addEventListener("change", syncIngestWorkflowUi);
  $("#btn-discover").addEventListener("click", () => discoverSources().catch(() => {}));
  $("#btn-auto-ingest").addEventListener("click", () => autoSearchIngest().catch(() => {}));
  $("#url-select-all")?.addEventListener("change", (e) => {
    document.querySelectorAll("#url-choices-list input[type=checkbox]").forEach((box) => {
      box.checked = e.target.checked;
    });
    syncUrlSelectAll();
  });

  $("#slide-source-mode")?.addEventListener("change", syncSlideSourceUi);
  $("#slide-search-workflow")?.addEventListener("change", syncSlideSourceUi);
  $("#btn-slide-discover")?.addEventListener("click", () => discoverSlideSources().catch(() => {}));

  $("#btn-research-ask").addEventListener("click", () => askResearchQuestion().catch(() => {}));
  $("#research-question")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askResearchQuestion().catch(() => {});
    }
  });

  $("#btn-generate").addEventListener("click", () => generateSlides().catch(() => {}));
  $("#btn-voice-send").addEventListener("click", () => sendVoiceTurn().catch(() => {}));
  $("#btn-voice-audio-send").addEventListener("click", () => sendVoiceFromRecording().catch(() => {}));
  $("#btn-analyze").addEventListener("click", () => analyzePitch().catch(() => {}));
  $("#btn-debug-send").addEventListener("click", () => sendDebugMessage().catch(() => {}));
  $("#debug-message")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendDebugMessage().catch(() => {});
    }
  });

  $("#btn-voice-record-start")?.addEventListener("click", () =>
    startRecording("voice", $("#voice-record-status"), $("#btn-voice-record-start"), $("#btn-voice-record-stop")).catch(() => {})
  );
  $("#btn-voice-record-stop")?.addEventListener("click", () =>
    stopRecording($("#voice-record-status"), $("#btn-voice-record-start"), $("#btn-voice-record-stop")).catch(() => {})
  );
  $("#btn-coach-record-start")?.addEventListener("click", () =>
    startRecording("coach", $("#coach-record-status"), $("#btn-coach-record-start"), $("#btn-coach-record-stop")).catch(() => {})
  );
  $("#btn-coach-record-stop")?.addEventListener("click", () =>
    stopRecording($("#coach-record-status"), $("#btn-coach-record-start"), $("#btn-coach-record-stop")).catch(() => {})
  );

  $("#btn-export").addEventListener("click", () => {
    const p = state.downloads?.pptx;
    if (p) window.open(fileUrl(p), "_blank");
  });

  $("#btn-new-session").addEventListener("click", () => {
    state.workspaceSessionId = "";
    state.researchChatHistory = [];
    state.discoveredUrls = [];
    state.selectedUrls = [];
    renderResearchChat();
    renderResearchUrlChoices([], []);
    $("#workspace-session").value = "";
    $("#ingest-status").textContent =
      "Set workspace topic and ingest sources to start a new ResearchMind session.";
    refreshDocuments().catch(() => {});
  });

  document.querySelectorAll(".mode-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-card").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.voiceMode = btn.dataset.mode;
    });
  });
}

bindUi();
initWorkspace().catch((err) => {
  console.error(err);
  showError("Could not connect to Studio API. Open /classic for full Gradio UI.");
});
