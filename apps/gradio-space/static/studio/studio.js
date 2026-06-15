import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.14.0/+esm";

const $ = (sel) => document.querySelector(sel);
const THEME_KEY = "studio-theme";

function getPreferredTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme === "dark" ? "dark" : "light";
  localStorage.setItem(THEME_KEY, theme);
  const icon = $("#theme-icon");
  if (icon) icon.textContent = theme === "dark" ? "light_mode" : "dark_mode";
  const checkbox = $("#theme-toggle");
  if (checkbox) checkbox.checked = theme === "dark";
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  applyTheme(current === "dark" ? "light" : "dark");
}

applyTheme(getPreferredTheme());

function appOrigin() {
  const { protocol, hostname, port } = window.location;
  if (protocol === "https:") {
    return window.location.origin;
  }
  const isLocal =
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]" ||
    hostname === "0.0.0.0";
  if (isLocal) {
    return window.location.origin;
  }
  // HF Spaces: TLS terminates at the edge; Gradio client must use https.
  const portSuffix = port ? `:${port}` : "";
  return `https://${hostname}${portSuffix}`;
}

const SLIDE_PIPELINE_STEPS = [
  "Load language model",
  "Gather lesson sources",
  "Generate slide outline",
  "Build PPTX, DOCX, and HTML exports",
];

const state = {
  workspaceTopic: "small model finetuning",
  workspaceSessionId: "",
  workspaceDocIds: [],
  discoveredUrls: [],
  selectedUrls: [],
  slideDiscoveredUrls: [],
  slideSelectedUrls: [],
  lessonsDiscoveredUrls: [],
  lessonsSelectedUrls: [],
  researchChatHistory: [],
  debugChatHistory: [],
  lessonsMode: "lesson",
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
  pendingLessonsAudioPath: null,
  holdMicActive: false,
  useBrowserMic: true,
  presenterSlides: [],
  presenterIndex: 0,
  fromConversation: false,
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
  setWorkspaceView("research");
  window.setTimeout(() => $("#research-question")?.focus(), 80);
}

function setWorkspaceView(view) {
  const btn = document.querySelector(`.nav-item[data-view="${view}"]`);
  if (btn) btn.click();
}

function hasChatHistory(kind) {
  if (kind === "research") return state.researchChatHistory.length > 0;
  if (kind === "voice") return state.history.length > 0;
  if (kind === "debug") return state.debugChatHistory.length > 0;
  return false;
}

function syncChatToSlidesButtons() {
  const researchBtn = $("#btn-research-to-slides");
  const lessonsBtn = $("#btn-lessons-to-slides");
  const chatBtn = $("#btn-chat-to-slides");
  if (researchBtn) researchBtn.disabled = !hasChatHistory("research");
  if (lessonsBtn) lessonsBtn.disabled = !hasChatHistory("voice");
  if (chatBtn) chatBtn.disabled = !hasChatHistory("debug");
}

function pickHistory(kind) {
  if (kind === "research") {
    return { history: state.researchChatHistory, historyKind: "research" };
  }
  if (kind === "voice") {
    return { history: state.history, historyKind: "voice" };
  }
  return { history: state.debugChatHistory, historyKind: "debug" };
}

function buildPresenterSlidesFromData(data) {
  const slides = [];
  if (data.gallery?.length) {
    for (const path of data.gallery) {
      slides.push({ type: "image", src: fileUrl(path), notes: "" });
    }
    return slides;
  }

  const canvasHost = document.createElement("div");
  const canvasHtml =
    data.canvas_html ||
    (data.preview_html ? `<div class="studio-canvas-inner">${data.preview_html}</div>` : "");
  canvasHost.innerHTML = canvasHtml || "";
  const cards = canvasHost.querySelectorAll(".lesson-slide");
  cards.forEach((card) => {
    const noteEl = card.querySelector(".speaker-note");
    const notes = noteEl ? noteEl.textContent.replace(/^Teacher note:\s*/i, "").trim() : "";
    slides.push({ type: "html", html: card.outerHTML, notes });
  });
  return slides;
}

function setPresenterEnabled(enabled) {
  const presentBtn = $("#btn-present");
  if (presentBtn) presentBtn.disabled = !enabled;
}

function renderPresenterSlide() {
  const slideEl = $("#presenter-slide");
  const counterEl = $("#presenter-counter");
  const notesEl = $("#presenter-notes");
  const slides = state.presenterSlides;
  if (!slideEl || !slides.length) return;

  const index = Math.max(0, Math.min(state.presenterIndex, slides.length - 1));
  state.presenterIndex = index;
  const slide = slides[index];
  slideEl.classList.remove("presenter-fade");
  void slideEl.offsetWidth;
  slideEl.classList.add("presenter-fade");

  if (slide.type === "image") {
    slideEl.innerHTML = `<img src="${slide.src}" alt="Slide ${index + 1}" />`;
  } else {
    slideEl.innerHTML = slide.html || "";
  }

  if (counterEl) counterEl.textContent = `${index + 1} / ${slides.length}`;
  if (notesEl) {
    notesEl.textContent = slide.notes || "No speaker notes for this slide.";
  }
}

function openPresenter() {
  if (!state.presenterSlides.length) return;
  const overlay = $("#presenter-overlay");
  if (!overlay) return;
  state.presenterIndex = 0;
  renderPresenterSlide();
  overlay.classList.remove("hidden");
  overlay.setAttribute("aria-hidden", "false");
}

function closePresenter() {
  const overlay = $("#presenter-overlay");
  if (!overlay) return;
  overlay.classList.add("hidden");
  overlay.setAttribute("aria-hidden", "true");
}

function presenterNext() {
  if (!state.presenterSlides.length) return;
  if (state.presenterIndex < state.presenterSlides.length - 1) {
    state.presenterIndex += 1;
    renderPresenterSlide();
  }
}

function presenterPrev() {
  if (!state.presenterSlides.length) return;
  if (state.presenterIndex > 0) {
    state.presenterIndex -= 1;
    renderPresenterSlide();
  }
}

function pulsePresentButton() {
  const btn = $("#btn-present");
  if (!btn) return;
  btn.classList.remove("btn-present-pulse");
  void btn.offsetWidth;
  btn.classList.add("btn-present-pulse");
  window.setTimeout(() => btn.classList.remove("btn-present-pulse"), 2600);
}

function renderSlideGenerationResult(data, { scrollToCanvas = false, pulsePresent = false } = {}) {
  finishProgressPanel(data);
  $("#generate-status").textContent = stripMd(data.status || "Slides generated.");
  const canvasHtml =
    data.canvas_html ||
    (data.preview_html ? `<div class="studio-canvas-inner">${data.preview_html}</div>` : "");
  $("#slide-canvas-content").innerHTML =
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
  state.presenterSlides = buildPresenterSlidesFromData(data);
  setPresenterEnabled(state.presenterSlides.length > 0);

  const dl = $("#downloads");
  if (data.downloads?.pptx) {
    dl.classList.remove("hidden");
    dl.innerHTML = `
      <a href="${fileUrl(data.downloads.pptx)}" download>PPTX</a>
      <a href="${fileUrl(data.downloads.docx)}" download>DOCX</a>
      <a href="${fileUrl(data.downloads.html)}" download>HTML</a>`;
    $("#btn-export").disabled = false;
    const exportBtn = $("#btn-export");
    if (exportBtn) exportBtn.textContent = "Download PPTX";
    syncLayoutOffsets();
  }

  const outlineDetails = $("#slide-outline-details");
  const outlineEl = $("#slide-outline");
  if (data.outline_md) {
    outlineEl.innerHTML = renderMarkdownLite(data.outline_md);
    outlineDetails?.classList.remove("hidden");
  } else {
    outlineEl.innerHTML = "";
    outlineDetails?.classList.add("hidden");
  }

  setTracePanel("#slides-trace-panel", data);

  if (scrollToCanvas) {
    $("#slide-canvas")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  if (pulsePresent && state.presenterSlides.length) {
    pulsePresentButton();
  }
}

async function collectSlideGenerationParams() {
  const topic = effectiveTopic($("#lesson-topic").value);
  const grade = $("#lesson-grade").value;
  const slideCount = Number($("#slide-count").value);
  const useRag = Boolean($("#lessons-use-rag")?.checked);
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
  return {
    topic,
    grade,
    slideCount,
    sessionId: state.workspaceSessionId,
    useRag,
    docIds,
    sourceMode,
    searchWorkflow,
    urlsText,
    selectedUrls,
    filePaths,
  };
}

async function runSlideGenerationApi(apiName, apiArgs) {
  startProgressPanel();
  const waitTimer = advanceProgressWhileWaiting();
  try {
    return await callApi(apiName, apiArgs);
  } finally {
    clearInterval(waitTimer);
    if (state.progressTimer) {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
    }
  }
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

function lessonsEffectiveTopic() {
  return effectiveTopic($("#lessons-topic")?.value || "");
}

function lessonsUseRag() {
  return Boolean($("#lessons-use-rag")?.checked);
}

function lessonsLanguage() {
  const select = $("#lessons-language");
  if (!select) return "en";
  if (select.value === "other") {
    return ($("#lessons-other-lang")?.value.trim() || "en").toLowerCase();
  }
  return select.value || "en";
}

function lessonsCoachVariant() {
  return $("#lessons-coach-variant")?.value || "tiny-aya-global";
}

function lessonsAutoSpeak() {
  return Boolean($("#lessons-auto-speak")?.checked);
}

function lessonsHasVoiceOut(language) {
  const code = (language || "en").split("-")[0];
  return (state.voicePresets?.voice_languages || []).includes(code);
}

function chatMessageText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const textPart = content.find((part) => typeof part === "string");
    return textPart || "";
  }
  if (typeof content === "object" && content.text) return String(content.text);
  return String(content);
}

function ingestSucceeded(status) {
  const text = (status || "").toLowerCase();
  return !(
    text.includes("error") ||
    text.includes("enter a research topic") ||
    text.includes("add urls") ||
    text.includes("no verified urls found")
  );
}

function chatMessageAudio(content) {
  if (!Array.isArray(content)) return null;
  const filePart = content.find((part) => part && typeof part === "object" && part.path);
  return filePart?.path || null;
}

function applyLessonsIngestResult(data) {
  $("#lessons-ingest-status").textContent = stripMd(data.status || "Ingest complete.");
  state.workspaceSessionId = data.session_id || state.workspaceSessionId;
  $("#workspace-session").value = state.workspaceSessionId;
  if (data.documents_html) {
    $("#documents-panel").innerHTML = data.documents_html;
  }
  renderWorkspaceDocList(data.documents || []);
  updateResearchRagBadge();
  updateResearchDocCount((data.documents || []).length);
  if (ingestSucceeded(data.status)) {
    const rag = $("#lessons-use-rag");
    if (rag) rag.checked = true;
  }
}

async function discoverLessonsSources() {
  const topic = lessonsEffectiveTopic();
  if (!topic) {
    showError("Set a lesson or workspace topic before discovering sources.");
    return;
  }
  await withRegionLoading($(".lessons-rail-controls"), "Discovering sources…", async () => {
    const data = await callApi("discover_sources", [topic, state.workspaceSessionId]);
    $("#lessons-ingest-status").textContent = stripMd(data.status || "Discovery complete.");
    renderLessonsUrlChoices(data.urls || [], data.selected_urls || data.urls || []);
    if (data.session_id) {
      state.workspaceSessionId = data.session_id;
      $("#workspace-session").value = data.session_id;
    }
    await refreshWorkspaceSessions(state.workspaceSessionId);
  });
}

async function autoLessonsIngest() {
  const topic = lessonsEffectiveTopic();
  if (!topic) {
    showError("Set a lesson or workspace topic before auto-ingest.");
    return;
  }
  await withRegionLoading($(".lessons-rail-controls"), "Auto-ingesting sources…", async () => {
    const data = await callApi("auto_search_ingest", [topic, state.workspaceSessionId]);
    applyLessonsIngestResult(data);
    state.lessonsDiscoveredUrls = [];
    state.lessonsSelectedUrls = [];
    renderLessonsUrlChoices([], []);
    await refreshWorkspaceSessions(state.workspaceSessionId);
  });
}

async function ingestLessonsSources() {
  const topic = lessonsEffectiveTopic();
  const pasted = $("#lessons-urls-text")?.value.trim() || "";
  const selected = getSelectedDiscoveredUrls("#lessons-url-choices-list");
  const files = $("#lessons-ingest-file")?.files;
  if (!pasted && !selected.length && !files?.length) {
    showError("Add URLs, select suggested sources, or upload a file — then ingest.");
    return;
  }
  await withRegionLoading($(".lessons-rail-controls"), "Ingesting sources…", async () => {
    const paths = [];
    if (files?.length) {
      for (const file of files) {
        paths.push(await uploadFile(file));
      }
    }
    const data = await callApi("ingest_sources", [
      topic,
      state.workspaceSessionId,
      pasted,
      selected,
      paths,
    ]);
    applyLessonsIngestResult(data);
    if (pasted) $("#lessons-urls-text").value = "";
    if (files?.length) $("#lessons-ingest-file").value = "";
    await refreshWorkspaceSessions(state.workspaceSessionId);
  });
}

function syncLessonsModeUi() {
  const placeholders = {
    explain: "e.g. How does finetuning differ from pretraining?",
    lesson: "What is the difference between pretraining and finetuning a small model?",
  };
  const messageEl = $("#lessons-message");
  if (messageEl) messageEl.placeholder = placeholders[state.lessonsMode] || placeholders.lesson;
}

function syncLessonsLanguageUi() {
  const isOther = $("#lessons-language")?.value === "other";
  $("#lessons-other-lang-wrap")?.classList.toggle("hidden", !isOther);
  const lang = lessonsLanguage();
  const note = state.voicePresets?.voiceout_note || "";
  const voiceHint = lessonsHasVoiceOut(lang)
    ? note
    : "VoiceOut not available for this language — text replies only.";
  const noteEl = $("#lessons-voiceout-note");
  if (noteEl) noteEl.textContent = voiceHint;
}

function renderLessonsChat() {
  const container = $("#lessons-chat-messages");
  if (!container) return;
  if (!state.history.length) {
    container.innerHTML =
      '<p class="research-chat-empty">Choose a language, then type, speak, or upload audio to start your lesson.</p>';
    syncChatToSlidesButtons();
    return;
  }
  const parts = [];
  for (const item of state.history) {
    if (item && typeof item === "object" && item.role) {
      const role = item.role === "user" ? "user" : "assistant";
      const label = role === "user" ? "You" : "Teacher";
      let body = renderMarkdownLite(chatMessageText(item.content));
      const audioPath = chatMessageAudio(item.content) || item.voiceout_path || null;
      if (audioPath) {
        body += `<audio class="chat-audio-inline" controls autoplay src="${fileUrl(audioPath)}"></audio>`;
      }
      if (role === "assistant" && item.rag_references) {
        body += `<div class="lessons-rag-refs">${renderMarkdownLite(item.rag_references)}</div>`;
      }
      parts.push(
        `<div class="research-chat-bubble research-chat-${role}"><div class="research-chat-role">${label}</div><div class="research-chat-body">${body}</div></div>`
      );
    } else if (Array.isArray(item) && item.length === 2) {
      const [user, assistant] = item;
      parts.push(
        `<div class="research-chat-bubble research-chat-user"><div class="research-chat-role">You</div><div class="research-chat-body">${renderMarkdownLite(user)}</div></div>` +
          `<div class="research-chat-bubble research-chat-assistant"><div class="research-chat-role">Teacher</div><div class="research-chat-body">${renderMarkdownLite(assistant)}</div></div>`
      );
    }
  }
  container.innerHTML = parts.join("");
  container.scrollTop = container.scrollHeight;
  syncChatToSlidesButtons();
}

function renderLessonsUrlChoices(urls, selected) {
  state.lessonsDiscoveredUrls = urls || [];
  state.lessonsSelectedUrls = selected?.length ? selected : [...state.lessonsDiscoveredUrls];
  renderUrlChoices(
    urls,
    selected,
    "#lessons-url-choices-list",
    "#lessons-url-choices-panel",
    { discovered: state.lessonsDiscoveredUrls, selected: state.lessonsSelectedUrls }
  );
}

function applyVoiceIngestResult(data) {
  applyLessonsIngestResult(data);
}

async function discoverVoiceSources() {
  return discoverLessonsSources();
}

async function autoVoiceIngest() {
  return autoLessonsIngest();
}

async function ingestVoiceSources() {
  return ingestLessonsSources();
}

function syncVoiceModeUi() {
  syncLessonsModeUi();
}

function renderVoiceChat() {
  renderLessonsChat();
}

function renderVoiceUrlChoices(urls, selected) {
  renderLessonsUrlChoices(urls, selected);
}

function voiceMessageText(content) {
  return chatMessageText(content);
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
  await withRegionLoading($(".card-ingest"), "Discovering sources…", async () => {
    const data = await callApi("discover_sources", [topic, state.workspaceSessionId]);
    $("#ingest-status").textContent = stripMd(data.status || "Discovery complete.");
    renderResearchUrlChoices(data.urls || [], data.selected_urls || data.urls || []);
    if (data.session_id) {
      state.workspaceSessionId = data.session_id;
      $("#workspace-session").value = data.session_id;
    }
    setTracePanel("#research-trace-panel", data);
    await refreshWorkspaceSessions(state.workspaceSessionId);
  });
}

async function discoverSlideSources() {
  const topic = effectiveTopic($("#lesson-topic")?.value);
  if (!topic) {
    showError("Set a topic before discovering sources.");
    return;
  }
  await withRegionLoading($(".controls-panel"), "Discovering sources…", async () => {
    const data = await callApi("discover_sources", [topic, state.workspaceSessionId]);
    renderSlideUrlChoices(data.urls || [], data.selected_urls || data.urls || []);
  });
}

async function autoSearchIngest() {
  const topic = effectiveTopic("");
  if (!topic) {
    showError("Set a workspace topic before auto-ingest.");
    return;
  }
  await withRegionLoading($(".card-ingest"), "Auto-ingesting sources…", async () => {
    const data = await callApi("auto_search_ingest", [topic, state.workspaceSessionId]);
    applyIngestResult(data);
    state.discoveredUrls = [];
    state.selectedUrls = [];
    renderResearchUrlChoices([], []);
    await refreshWorkspaceSessions(state.workspaceSessionId);
  });
}

async function ingestSources({ urlsText = "", selectedUrls = [], pendingFiles = null } = {}) {
  const topic = effectiveTopic("");
  const workflow = getIngestWorkflow();
  let selected = selectedUrls;
  if (workflow === "select") selected = getSelectedDiscoveredUrls();
  const pasted = workflow === "direct" ? urlsText : urlsText || $("#ingest-url").value.trim();
  const files = pendingFiles || $("#ingest-file").files;
  if (!pasted && !selected.length && !files?.length) {
    showError("Add URLs, select suggested sources, or upload a file — then ingest.");
    return;
  }
  await withRegionLoading($(".card-ingest"), "Ingesting sources…", async () => {
    const paths = [];
    if (files?.length) {
      for (const file of files) {
        const b64 = await fileToBase64(file);
        const saved = await callApi("save_upload", [file.name, b64]);
        paths.push(saved.path);
      }
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
  });
}

function renderResearchChat() {
  const container = $("#research-chat-messages");
  if (!state.researchChatHistory.length) {
    container.innerHTML =
      '<p class="research-chat-empty">Ingest sources, then ask questions — answers include citations from your library.</p>';
    syncChatToSlidesButtons();
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
  syncChatToSlidesButtons();
}

function renderDebugChat() {
  const container = $("#debug-chat-messages");
  if (!state.debugChatHistory.length) {
    container.innerHTML =
      '<p class="research-chat-empty">Ask the local model — turn on RAG to ground answers in your library.</p>';
    syncChatToSlidesButtons();
    return;
  }
  container.innerHTML = state.debugChatHistory
    .map(([user, assistant]) => {
      return `<div class="research-chat-bubble research-chat-user"><div class="research-chat-role">You</div><div class="research-chat-body">${renderMarkdownLite(user)}</div></div><div class="research-chat-bubble research-chat-assistant"><div class="research-chat-role">Model</div><div class="research-chat-body">${renderMarkdownLite(assistant)}</div></div>`;
    })
    .join("");
  container.scrollTop = container.scrollHeight;
  syncChatToSlidesButtons();
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
  await withRegionLoading($("#research-chat-panel .card-chat"), "Searching sources…", async () => {
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
  });
}

async function sendDebugMessage() {
  const message = $("#debug-message").value.trim();
  if (!message) {
    showError("Enter a message.");
    return;
  }
  const useRag = $("#debug-use-rag").checked;
  const debugSession = $("#debug-session")?.value || "";
  const debugDocIds = selectedDebugDocIds();
  const workspaceDocIds = selectedWorkspaceDocIds();
  const modelKey = $("#debug-model-key")?.value || "";
  await withRegionLoading($(".coach-debug-card"), "Thinking…", async () => {
    const data = await callApi("debug_chat", [
      message,
      state.debugChatHistory,
      useRag,
      debugSession,
      debugDocIds,
      modelKey,
      state.workspaceSessionId,
      workspaceDocIds,
    ]);
    state.debugChatHistory = data.history || [];
    renderDebugChat();
    $("#debug-message").value = "";
    if (data.rag_hint) {
      $("#debug-rag-hint").textContent = stripMd(data.rag_hint);
    }
    setTracePanel("#debug-trace-panel", data);
  });
}

function effectiveDebugSessionId() {
  return ($("#debug-session")?.value || "").trim() || state.workspaceSessionId;
}

function selectedDebugDocIds() {
  const boxes = document.querySelectorAll("#debug-doc-list input[type=checkbox]");
  if (!boxes.length) return [];
  return [...document.querySelectorAll("#debug-doc-list input[type=checkbox]:checked")].map(
    (el) => el.value
  );
}

function renderDebugDocList(docs) {
  const container = $("#debug-doc-list");
  if (!container) return;
  if (!docs?.length) {
    container.innerHTML = '<p class="status-text">No documents in this session yet.</p>';
    updateDebugRagHint();
    return;
  }
  container.innerHTML = docs
    .map(
      (d) =>
        `<label class="workspace-doc-item"><input type="checkbox" value="${d.id}" checked />${escapeHtml(d.title)}</label>`
    )
    .join("");
  container.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.addEventListener("change", updateDebugRagHint);
  });
  updateDebugRagHint();
}

function updateDebugRagHint() {
  const el = $("#debug-rag-hint");
  if (!el) return;
  const sid = effectiveDebugSessionId();
  const selected = selectedDebugDocIds();
  const total = document.querySelectorAll("#debug-doc-list input[type=checkbox]").length;
  if (selected.length && selected.length < total) {
    el.textContent = `RAG scope: ${selected.length} selected document(s).`;
  } else if (sid) {
    el.textContent = total
      ? `RAG scope: all ${total} document(s) in session.`
      : "RAG scope: all documents in session.";
  } else {
    el.textContent = "RAG scope: entire indexed corpus (all sessions).";
  }
}

async function refreshDebugSessions(selectId) {
  const data = await callApi("list_sessions", []);
  const sessions = data.sessions || [];
  const select = $("#debug-session");
  if (!select) return;
  const current = selectId ?? select.value;
  select.innerHTML =
    '<option value="">Workspace default</option>' +
    sessions.map((s) => `<option value="${s.id}">${s.label || s.topic}</option>`).join("");
  if (current && sessions.some((s) => s.id === current)) {
    select.value = current;
  }
}

async function refreshDebugDocuments() {
  const sessionId = effectiveDebugSessionId();
  const data = await callApi("list_documents", [sessionId]);
  renderDebugDocList(data.documents || []);
}

function updateProjectTitle() {
  const topic = state.workspaceTopic || "";
  const short = topic.split(" for ")[0] || topic || "Project";
  const title = short.slice(0, 40);
  const el = $("#project-title");
  if (el) {
    el.textContent = title;
    el.title = topic || title;
  }
  updateWorkspaceContextSummary();
}

function updateWorkspaceContextSummary() {
  const el = $("#workspace-context-summary-text");
  if (!el) return;
  const topic = (state.workspaceTopic || "Workspace").trim();
  const shortTopic = (topic.split(" for ")[0] || topic || "Workspace").slice(0, 32);
  const sessionSel = $("#workspace-session");
  let sessionLabel = "New session";
  if (sessionSel?.value) {
    const label = sessionSel.selectedOptions[0]?.textContent?.trim() || "Session";
    sessionLabel = label.length > 22 ? `${label.slice(0, 19)}…` : label;
  }
  el.textContent = `${shortTopic} · ${sessionLabel}`;
  el.title = topic ? `${topic} · ${sessionLabel}` : sessionLabel;
}

function syncViewChrome(view) {
  const active = view || $(".workspace")?.dataset.view || "slides";
  document.body.dataset.view = active;
}

function openSidebar() {
  $("#sidebar")?.classList.add("open");
  $("#sidebar-backdrop")?.classList.remove("hidden");
  document.body.classList.add("sidebar-open");
}

function closeSidebar() {
  $("#sidebar")?.classList.remove("open");
  $("#sidebar-backdrop")?.classList.add("hidden");
  document.body.classList.remove("sidebar-open");
}

function syncLayoutOffsets() {
  const topbar = $(".topbar");
  const ctxBar = $("#workspace-context-bar");
  if (!topbar || !ctxBar) return;
  document.documentElement.style.setProperty("--topbar-h", `${topbar.offsetHeight}px`);
  document.documentElement.style.setProperty("--context-bar-h", `${ctxBar.offsetHeight}px`);
}

function bindLayoutSync() {
  syncLayoutOffsets();
  window.addEventListener("resize", syncLayoutOffsets);
  const ctxBar = $("#workspace-context-bar");
  if (ctxBar && typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => syncLayoutOffsets());
    ro.observe(ctxBar);
    if ($(".topbar")) ro.observe($(".topbar"));
  }
  $("#workspace-context-mobile")?.addEventListener("toggle", syncLayoutOffsets);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("#sidebar")?.classList.contains("open")) {
      closeSidebar();
    }
  });
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
    state.client = await Client.connect(appOrigin());
  }
  return state.client;
}

let globalLoadingSuppress = 0;

function setLoading(on) {
  if (on && globalLoadingSuppress > 0) return;
  $("#studio-loading").classList.toggle("hidden", !on);
}

function setRegionLoading(container, on, message = "Working…", { overlayEl = null, hint = "" } = {}) {
  if (!container) return;
  let overlay = overlayEl || container.querySelector(":scope > .region-loading");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "region-loading hidden";
    overlay.setAttribute("aria-live", "polite");
    overlay.innerHTML = `
      <div class="region-loading-inner">
        <span class="studio-spinner" aria-hidden="true"></span>
        <p class="region-loading-text"></p>
        <p class="region-loading-hint hidden"></p>
      </div>`;
    container.insertBefore(overlay, container.firstChild);
    if (getComputedStyle(container).position === "static") {
      container.classList.add("region-loading-host");
    }
  }
  const textEl =
    overlay.querySelector(".region-loading-text") || overlay.querySelector("#canvas-overlay-text");
  if (textEl) textEl.textContent = message;
  const hintEl =
    overlay.querySelector(".region-loading-hint") || overlay.querySelector(".canvas-overlay-hint");
  if (hintEl) {
    hintEl.textContent = hint;
    hintEl.classList.toggle("hidden", !hint);
  }
  overlay.classList.toggle("hidden", !on);
  container.setAttribute("aria-busy", on ? "true" : "false");
}

async function withRegionLoading(container, message, fn, options = {}) {
  globalLoadingSuppress += 1;
  setRegionLoading(container, true, message, options);
  try {
    return await fn();
  } finally {
    globalLoadingSuppress -= 1;
    setRegionLoading(container, false, message, options);
  }
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
    showError(`${message} — try Classic UI (?classic)`);
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
  await refreshDebugSessions();
  updateWorkspaceContextSummary();
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

async function initLanguageLessons() {
  const data = await callApi("voice_presets", []);
  state.voicePresets = data;
  const langSelect = $("#lessons-language");
  if (langSelect) {
    const opts = (data.languages || [])
      .map((o) => `<option value="${o.value}">${o.label}</option>`)
      .join("");
    langSelect.innerHTML = `${opts}<option value="other">Other (text only)</option>`;
    langSelect.value = data.default_language || "en";
  }
  const coachEl = document.querySelector(".lessons-coach-model");
  if (coachEl && data.coach_chain_labels?.length) {
    const primary = data.coach_chain_labels[0];
    const fallback = data.coach_chain_labels[1];
    coachEl.textContent = fallback
      ? `Coach: ${primary} (auto-fallback: ${fallback})`
      : `Coach: ${primary}`;
  }
  syncLessonsLanguageUi();
}

async function initVoicePresets() {
  return initLanguageLessons();
}

async function selectActiveModel(key) {
  const data = await callApi("set_active_model", [key]);
  $("#settings-status").innerHTML = renderMarkdownLite(data.status_markdown || "");
  const fresh = await callApi("model_choices", []);
  state.modelChoices = fresh;
  $("#settings-active-model").textContent = `${fresh.active_label} (${fresh.active_backend})`;
  return data;
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
      select.onchange = () => {
        const key = select.value;
        if (debugSelect) debugSelect.value = key;
        selectActiveModel(key).catch(() => {});
      };
    }
    if (debugSelect) {
      debugSelect.innerHTML = options;
      debugSelect.value = data.active_model;
      debugSelect.onchange = () => {
        const key = debugSelect.value;
        if (select) select.value = key;
        selectActiveModel(key).catch(() => {});
      };
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
  syncViewChrome();
  updateProjectTitle();
  updateResearchRagBadge();
  await refreshWorkspaceSessions();
  await refreshDocuments();
  await initLanguageLessons();
  await initSettings();
  syncLessonsModeUi();
  renderLessonsChat();
  await refreshDebugDocuments();
  syncChatToSlidesButtons();
  const recStatus = await callApi("recording_status", []);
  state.useBrowserMic = !recStatus.backend || /unavailable|no capture/i.test(recStatus.message || "");
  syncLayoutOffsets();
}

async function ingestUrl() {
  await ingestSources({ urlsText: $("#ingest-url").value.trim() });
}

async function ingestFiles(files) {
  if (!files?.length) return;
  await ingestSources({ pendingFiles: files });
}

async function generateSlides() {
  const params = await collectSlideGenerationParams();

  await withRegionLoading(
    $("#slide-canvas"),
    "Generating slides…",
    async () => {
      let data;
      try {
        data = await runSlideGenerationApi("generate_slides", [
          params.topic,
          params.grade,
          params.slideCount,
          params.sessionId,
          params.useRag,
          params.docIds,
          params.sourceMode,
          params.searchWorkflow,
          params.urlsText,
          params.selectedUrls,
          params.filePaths,
        ]);
      } catch (_err) {
        $("#progress-eta").textContent = "Failed";
        throw _err;
      }

      state.fromConversation = false;
      renderSlideGenerationResult(data);
    },
    {
      overlayEl: $("#canvas-overlay"),
      hint: "First run may take several minutes on CPU; use GPU Space or fewer slides for a quick demo.",
    }
  );
}

async function generateSlidesFromConversation(kind) {
  const { history, historyKind } = pickHistory(kind);
  if (!history?.length) {
    showError("Start a conversation first.");
    return;
  }

  const params = await collectSlideGenerationParams();
  setWorkspaceView("slides");

  await withRegionLoading(
    $("#slide-canvas"),
    "Generating slides from chat…",
    async () => {
      let data;
      try {
        data = await runSlideGenerationApi("generate_slides_from_conversation", [
          history,
          historyKind,
          params.topic,
          params.grade,
          params.slideCount,
          params.sessionId,
          params.useRag,
          params.docIds,
          params.sourceMode,
          params.searchWorkflow,
          params.urlsText,
          params.selectedUrls,
          params.filePaths,
        ]);
      } catch (_err) {
        $("#progress-eta").textContent = "Failed";
        throw _err;
      }

      state.fromConversation = true;
      renderSlideGenerationResult(data, { scrollToCanvas: true, pulsePresent: true });
    },
    {
      overlayEl: $("#canvas-overlay"),
      hint: "First run may take several minutes on CPU; use GPU Space or fewer slides for a quick demo.",
    }
  );
}

function renderLessonsReply(data) {
  state.history = data.history ?? state.history;
  if (state.history.length) {
    const last = state.history[state.history.length - 1];
    if (last && typeof last === "object" && last.role === "assistant") {
      if (data.rag_references) last.rag_references = data.rag_references;
      if (data.voiceout_path && lessonsAutoSpeak()) last.voiceout_path = data.voiceout_path;
    }
  }
  renderLessonsChat();
  if (data.status) {
    const statusEl = $("#lessons-turn-status");
    if (statusEl) statusEl.textContent = stripMd(data.status);
  }
}

function renderVoiceReply(data, options) {
  renderLessonsReply(data, options);
}

async function sendLanguageLessonTurn({ message = "", audioPath = "" } = {}) {
  const topic = lessonsEffectiveTopic();
  const useRag = lessonsUseRag();
  const docIds = effectiveDocIds([]);
  const language = lessonsLanguage();
  const asr = state.voicePresets?.default_asr || null;
  const autoVoiceout = lessonsAutoSpeak() && lessonsHasVoiceOut(language);
  const coachVariant = lessonsCoachVariant();
  const loadingLabel = message || audioPath ? (message ? "Teacher is thinking…" : "Processing audio…") : "Sending…";

  await withRegionLoading($(".lessons-main-card"), loadingLabel, async () => {
    const data = await callApi("language_lesson_turn", [
      message,
      audioPath || "",
      state.lessonsMode,
      topic,
      state.workspaceSessionId,
      useRag,
      state.history,
      docIds,
      language,
      asr,
      autoVoiceout,
      "",
      coachVariant,
    ]);
    if (data.user_text) {
      $("#lessons-message").value = data.user_text;
    } else if (message) {
      $("#lessons-message").value = "";
    }
    renderLessonsReply(data);
  });
}

async function sendLessonsTurn() {
  const message = $("#lessons-message")?.value.trim() || "";
  let audioPath = state.pendingLessonsAudioPath;
  const file = $("#lessons-audio-upload")?.files?.[0];
  if (file) audioPath = await uploadFile(file);
  if (message) {
    await sendLanguageLessonTurn({ message });
    state.pendingLessonsAudioPath = null;
    return;
  }
  if (audioPath) {
    await sendLanguageLessonTurn({ audioPath });
    state.pendingLessonsAudioPath = null;
    if ($("#lessons-audio-upload")) $("#lessons-audio-upload").value = "";
    return;
  }
  showError("Type a message, hold the mic, or upload audio.");
}

async function sendVoiceTurn() {
  return sendLessonsTurn();
}

async function sendVoiceAudioTurn(audioPath) {
  return sendLanguageLessonTurn({ audioPath });
}

async function clearLessonsConversation() {
  const data = await callApi("teacher_voice_clear", []);
  state.history = [];
  renderLessonsChat();
  if ($("#lessons-message")) $("#lessons-message").value = "";
  const statusEl = $("#lessons-turn-status");
  if (statusEl) statusEl.textContent = stripMd(data.status || "Conversation cleared.");
}

async function clearVoiceConversation() {
  return clearLessonsConversation();
}

async function startLessonsHoldMic(e) {
  if (state.holdMicActive) return;
  state.holdMicActive = true;
  e?.preventDefault();
  const holdBtn = $("#btn-lessons-hold-mic");
  holdBtn?.classList.add("recording");
  await startRecording(
    "lessons",
    $("#lessons-record-status"),
    $("#btn-lessons-record-start"),
    $("#btn-lessons-record-stop")
  );
}

async function stopLessonsHoldMic(e) {
  if (!state.holdMicActive) return;
  state.holdMicActive = false;
  e?.preventDefault();
  $("#btn-lessons-hold-mic")?.classList.remove("recording");
  const path = await stopRecording(
    $("#lessons-record-status"),
    $("#btn-lessons-record-start"),
    $("#btn-lessons-record-stop")
  );
  if (path) await sendLanguageLessonTurn({ audioPath: path });
}

async function sendLessonsFromRecording() {
  let path = state.pendingLessonsAudioPath;
  const file = $("#lessons-audio-upload")?.files?.[0];
  if (file) path = await uploadFile(file);
  if (!path) {
    showError("Record or upload audio first.");
    return;
  }
  await sendLanguageLessonTurn({ audioPath: path });
  state.pendingLessonsAudioPath = null;
}

async function sendVoiceFromRecording() {
  return sendLessonsFromRecording();
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
  if (state.recordingTarget === "lessons") state.pendingLessonsAudioPath = path;
  state.recordingTarget = null;
  return path;
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
      syncViewChrome(btn.dataset.view);
      closeSidebar();
    });
  });

  $("#btn-open-settings")?.addEventListener("click", () => {
    closeSidebar();
    openSettingsDrawer();
  });
  $("#btn-close-settings")?.addEventListener("click", closeSettingsDrawer);
  $("#settings-backdrop")?.addEventListener("click", closeSettingsDrawer);
  $("#theme-toggle")?.addEventListener("change", toggleTheme);
  $("#theme-toggle-btn")?.addEventListener("click", toggleTheme);
  $("#btn-reload-model")?.addEventListener("click", () => reloadModelFromSettings().catch(() => {}));

  $("#btn-open-research-view")?.addEventListener("click", openResearchView);
  $("#sidebar-open")?.addEventListener("click", openSidebar);
  $("#sidebar-close")?.addEventListener("click", closeSidebar);
  $("#sidebar-backdrop")?.addEventListener("click", closeSidebar);

  $("#workspace-topic").addEventListener("input", (e) => {
    state.workspaceTopic = e.target.value.trim();
    updateProjectTitle();
  });

  $("#workspace-session").addEventListener("change", (e) => {
    state.workspaceSessionId = e.target.value;
    updateWorkspaceContextSummary();
    refreshDocuments().catch(() => {});
    refreshDebugDocuments().catch(() => {});
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
  $("#btn-present")?.addEventListener("click", () => openPresenter());
  $("#btn-research-to-slides")?.addEventListener("click", () =>
    generateSlidesFromConversation("research").catch(() => {})
  );
  $("#btn-lessons-to-slides")?.addEventListener("click", () =>
    generateSlidesFromConversation("voice").catch(() => {})
  );
  $("#btn-chat-to-slides")?.addEventListener("click", () =>
    generateSlidesFromConversation("debug").catch(() => {})
  );
  $("#btn-presenter-close")?.addEventListener("click", closePresenter);
  $("#btn-presenter-backdrop")?.addEventListener("click", closePresenter);
  $("#btn-presenter-prev")?.addEventListener("click", presenterPrev);
  $("#btn-presenter-next")?.addEventListener("click", presenterNext);
  document.addEventListener("keydown", (e) => {
    const overlay = $("#presenter-overlay");
    if (!overlay || overlay.classList.contains("hidden")) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closePresenter();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      presenterNext();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      presenterPrev();
    }
  });

  $("#btn-lessons-send")?.addEventListener("click", () => sendLessonsTurn().catch(() => {}));
  $("#lessons-message")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendLessonsTurn().catch(() => {});
    }
  });
  $("#btn-lessons-discover")?.addEventListener("click", () => discoverLessonsSources().catch(() => {}));
  $("#btn-lessons-auto-ingest")?.addEventListener("click", () => autoLessonsIngest().catch(() => {}));
  $("#btn-lessons-ingest")?.addEventListener("click", () => ingestLessonsSources().catch(() => {}));
  $("#lessons-ingest-file")?.addEventListener("change", () => ingestLessonsSources().catch(() => {}));
  $("#btn-lessons-clear")?.addEventListener("click", () => clearLessonsConversation().catch(() => {}));
  $("#lessons-language")?.addEventListener("change", syncLessonsLanguageUi);
  $("#lessons-other-lang")?.addEventListener("input", syncLessonsLanguageUi);
  $("#lessons-audio-upload")?.addEventListener("change", () => sendLessonsTurn().catch(() => {}));

  const holdMic = $("#btn-lessons-hold-mic");
  if (holdMic) {
    holdMic.addEventListener("mousedown", (e) => startLessonsHoldMic(e).catch(() => {}));
    holdMic.addEventListener("mouseup", (e) => stopLessonsHoldMic(e).catch(() => {}));
    holdMic.addEventListener("mouseleave", (e) => {
      if (state.holdMicActive) stopLessonsHoldMic(e).catch(() => {});
    });
    holdMic.addEventListener("touchstart", (e) => startLessonsHoldMic(e).catch(() => {}), { passive: false });
    holdMic.addEventListener("touchend", (e) => stopLessonsHoldMic(e).catch(() => {}));
  }

  $("#btn-lessons-record-start")?.addEventListener("click", () =>
    startRecording(
      "lessons",
      $("#lessons-record-status"),
      $("#btn-lessons-record-start"),
      $("#btn-lessons-record-stop")
    ).catch(() => {})
  );
  $("#btn-lessons-record-stop")?.addEventListener("click", () =>
    stopRecording(
      $("#lessons-record-status"),
      $("#btn-lessons-record-start"),
      $("#btn-lessons-record-stop")
    ).catch(() => {})
  );

  $("#btn-debug-send").addEventListener("click", () => sendDebugMessage().catch(() => {}));

  $("#debug-session")?.addEventListener("change", () => refreshDebugDocuments().catch(() => {}));
  $("#debug-refresh-sessions")?.addEventListener("click", () => {
    refreshDebugSessions().catch(() => {});
    refreshDebugDocuments().catch(() => {});
  });
  $("#debug-use-rag")?.addEventListener("change", updateDebugRagHint);
  $("#debug-message")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendDebugMessage().catch(() => {});
    }
  });

  $("#btn-export").addEventListener("click", () => {
    const p = state.downloads?.pptx;
    if (p) window.open(fileUrl(p), "_blank");
  });

  $("#btn-new-session").addEventListener("click", () => {
    state.workspaceSessionId = "";
    state.researchChatHistory = [];
    state.debugChatHistory = [];
    state.discoveredUrls = [];
    state.selectedUrls = [];
    renderResearchChat();
    renderDebugChat();
    renderResearchUrlChoices([], []);
    $("#workspace-session").value = "";
    $("#ingest-status").textContent =
      "Set workspace topic and ingest sources to start a new ResearchMind session.";
    refreshDocuments().catch(() => {});
  });

  document.querySelectorAll("#lessons-modes .mode-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#lessons-modes .mode-card").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.lessonsMode = btn.dataset.mode;
      syncLessonsModeUi();
    });
  });

  syncLessonsModeUi();
  bindLayoutSync();
}

bindUi();
initWorkspace().catch((err) => {
  console.error(err);
  showError("Could not connect to Studio API. Open ?classic for full Gradio UI.");
});
