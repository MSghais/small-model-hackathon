import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.14.0/+esm";

const $ = (sel) => document.querySelector(sel);
const SLIDE_PIPELINE_STEPS = [
  "Load language model",
  "Gather lesson sources",
  "Generate slide outline",
  "Build PPTX, DOCX, and HTML exports",
];

const state = {
  sessionId: "",
  topic: "Photosynthesis for 6th Grade",
  voiceMode: "lesson",
  history: [],
  downloads: null,
  client: null,
  progressTimer: null,
  progressStartedAt: null,
};

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
        const duration =
          step.duration_s != null ? ` (${step.duration_s}s)` : "";
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
    logEl.textContent = stripMd(data.progress_log);
    logEl.classList.remove("hidden");
  }
  if (data?.elapsed_seconds != null) {
    $("#progress-elapsed").textContent = `Elapsed: ${Number(data.elapsed_seconds).toFixed(1)}s`;
  }
  $("#progress-eta").textContent = "Complete";
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
    if (raw.length === 1 && raw[0] !== null && typeof raw[0] === "object") {
      return raw[0];
    }
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
    if (data && data.ok === false) {
      throw new Error(data.error || "Request failed");
    }
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
    reader.onload = () => {
      const raw = reader.result.split(",")[1];
      resolve(raw);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function refreshDocuments() {
  const data = await callApi("list_documents", [state.sessionId]);
  $("#documents-panel").innerHTML =
    data.documents_html ||
    '<p class="studio-empty-docs">No documents indexed yet.</p>';
  if (data.session_id) state.sessionId = data.session_id;
}

async function initSessions() {
  const data = await callApi("list_sessions", []);
  const sessions = data.sessions || [];
  const match = sessions.find((s) =>
    (s.topic || "").toLowerCase().includes("photosynthesis")
  );
  if (match) {
    state.sessionId = match.id;
    $("#project-title").textContent = match.topic || "Photosynthesis";
  }
  await refreshDocuments();
}

async function ingestUrl() {
  const url = $("#ingest-url").value.trim();
  state.topic = $("#lesson-topic").value.trim() || state.topic;
  const data = await callApi("ingest_url", [state.topic, url, state.sessionId]);
  $("#ingest-status").textContent = stripMd(data.status || "Ingest complete.");
  state.sessionId = data.session_id || state.sessionId;
  $("#documents-panel").innerHTML = data.documents_html || "";
}

async function ingestFiles(files) {
  if (!files?.length) return;
  state.topic = $("#lesson-topic").value.trim() || state.topic;
  const paths = [];
  for (const file of files) {
    const b64 = await fileToBase64(file);
    const saved = await callApi("save_upload", [file.name, b64]);
    paths.push(saved.path);
  }
  const data = await callApi("ingest_files", [state.topic, state.sessionId, paths]);
  $("#ingest-status").textContent = stripMd(data.status || "Upload ingested.");
  state.sessionId = data.session_id || state.sessionId;
  $("#documents-panel").innerHTML = data.documents_html || "";
}

function stripMd(text) {
  return String(text).replace(/\*\*/g, "").replace(/`/g, "");
}

async function generateSlides() {
  const topic = $("#lesson-topic").value.trim();
  const grade = $("#lesson-grade").value;
  const slideCount = Number($("#slide-count").value);
  const useRag = $("#use-rag").checked;
  state.topic = topic;

  startProgressPanel();
  const waitTimer = advanceProgressWhileWaiting();
  let data;
  try {
    data = await callApi("generate_slides", [
      topic,
      grade,
      slideCount,
      state.sessionId,
      useRag,
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
    (data.preview_html
      ? `<div class="studio-canvas-inner">${data.preview_html}</div>`
      : "");
  $("#slide-canvas").innerHTML =
    canvasHtml ||
    '<div class="studio-canvas-empty"><p>Preview unavailable.</p></div>';

  state.downloads = data.downloads;
  const dl = $("#downloads");
  if (data.downloads?.pptx) {
    dl.classList.remove("hidden");
    dl.innerHTML = `
      <a href="/file=${encodeURIComponent(data.downloads.pptx)}" download>PPTX</a>
      <a href="/file=${encodeURIComponent(data.downloads.docx)}" download>DOCX</a>
      <a href="/file=${encodeURIComponent(data.downloads.html)}" download>HTML</a>`;
    $("#btn-export").disabled = false;
  }
}

async function sendVoiceTurn() {
  const message = $("#voice-message").value.trim();
  const topic = $("#lesson-topic").value.trim();
  const useRag = $("#use-rag").checked;
  const data = await callApi("teacher_voice_turn", [
    message,
    state.voiceMode,
    topic,
    state.sessionId,
    useRag,
    state.history,
  ]);
  state.history = data.history || [];
  $("#voice-reply").textContent = data.assistant || data.status || "";
}

async function analyzePitch() {
  const file = $("#coach-audio").files?.[0];
  if (!file) {
    showError("Choose an audio file to analyze.");
    return;
  }
  $("#coach-panel").innerHTML = `
    <div class="studio-coach-panel studio-coach-live">
      <div class="studio-coach-header"><span class="studio-coach-dot"></span>
      <span class="studio-coach-label">Analyzing…</span></div>
    </div>`;
  const b64 = await fileToBase64(file);
  const saved = await callApi("save_upload", [file.name, b64]);
  const data = await callApi("analyze_pitch", [saved.path]);
  $("#coach-panel").innerHTML = data.coach_panel_html || "";
}

function bindUi() {
  $("#slide-count").addEventListener("input", (e) => {
    $("#slide-count-val").textContent = e.target.value;
  });

  document.querySelectorAll(".nav-item[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item[data-view]").forEach((b) =>
        b.classList.remove("active")
      );
      btn.classList.add("active");
      $(".workspace").dataset.view = btn.dataset.view;
      $("#sidebar").classList.remove("open");
    });
  });

  $("#sidebar-open")?.addEventListener("click", () =>
    $("#sidebar").classList.add("open")
  );
  $("#sidebar-close")?.addEventListener("click", () =>
    $("#sidebar").classList.remove("open")
  );

  $("#btn-ingest-url").addEventListener("click", () => ingestUrl().catch(() => {}));
  $("#ingest-file").addEventListener("change", (e) =>
    ingestFiles(e.target.files).catch(() => {})
  );
  $("#btn-generate").addEventListener("click", () => generateSlides().catch(() => {}));
  $("#btn-voice-send").addEventListener("click", () => sendVoiceTurn().catch(() => {}));
  $("#btn-analyze").addEventListener("click", () => analyzePitch().catch(() => {}));

  $("#btn-export").addEventListener("click", () => {
    const p = state.downloads?.pptx;
    if (p) window.open(`/file=${encodeURIComponent(p)}`, "_blank");
  });

  $("#btn-new-session").addEventListener("click", () => {
    state.sessionId = "";
    $("#ingest-status").textContent =
      "Enter a topic and ingest sources to start a new ResearchMind session.";
    refreshDocuments().catch(() => {});
  });

  document.querySelectorAll(".mode-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-card").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.voiceMode = btn.dataset.mode;
    });
  });

  $("#lesson-topic").addEventListener("change", (e) => {
    state.topic = e.target.value;
    const short = state.topic.split(" for ")[0] || state.topic;
    $("#project-title").textContent = short.slice(0, 40);
  });
}

bindUi();
initSessions().catch((err) => {
  console.error(err);
  showError("Could not connect to Studio API. Open /classic for full Gradio UI.");
});
