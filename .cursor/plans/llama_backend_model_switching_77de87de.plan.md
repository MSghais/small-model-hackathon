---
name: Llama backend model switching
overview: Add an official MiniCPM5-1B GGUF preset for the llama.cpp / Llama Champion path, then wire a shared runtime model selector so local dev can switch between transformers and llama.cpp backends (and other presets) from Gradio Settings and Studio — not just the Chat debug tab.
todos:
  - id: add-gguf-preset
    content: Add minicpm5-1b-gguf preset to models.yaml and document in .env.example
    status: pending
  - id: runtime-model-state
    content: Add set_runtime_model_key() and make get_active_model_key() runtime-aware in model_loading.py
    status: pending
  - id: classic-ui-sync
    content: Wire Settings + Chat dropdowns to set_runtime_model_key; reload on change
    status: pending
  - id: studio-api-sync
    content: Add api_set_active_model + studio.js settings dropdown handler; sync debug picker
    status: pending
  - id: tests-docs
    content: Test preset parsing + runtime key override; document local switching in USAGE.md
    status: pending
isProject: false
---

# Llama backend + runtime model switching (local dev)

## What already exists

Your repo already has **two inference backends** behind one factory — no new backend code is required:

```mermaid
flowchart LR
  GradioUI[Gradio Classic + Studio]
  ModelLoading[model_loading.py]
  Factory[factory.py]
  LlamaCpp[LlamaCppBackend]
  Transformers[TransformersBackend]
  GradioUI --> ModelLoading --> Factory
  Factory -->|preset.backend=llama_cpp| LlamaCpp
  Factory -->|preset.backend=transformers| Transformers
```

- Presets live in [`models.yaml`](models.yaml); backend is chosen **per preset**, not via a separate toggle.
- Switching transformers → llama.cpp means switching preset, e.g. `minicpm5-1b` → `minicpm5-1b-gguf` (to be added).
- [`libs/inference/src/inference/llama_cpp.py`](libs/inference/src/inference/llama_cpp.py) downloads GGUF from Hub and runs `create_chat_completion`.
- [`ALLOW_MODEL_SWITCH`](libs/inference/src/inference/config.py) gates dropdowns in Settings, Chat, and Studio debug — but **only Chat/Debug actually pass the selected key to inference**.

### Current gap (why switching feels broken)

[`get_active_model_key()`](apps/gradio-space/src/gradio_space/model_loading.py) always returns the **startup** preset from env/`models.yaml`:

```12:13:apps/gradio-space/src/gradio_space/model_loading.py
def get_active_model_key() -> str:
    return _app_config.active_model
```

Lesson slides, ResearchMind, EchoCoach, TeacherVoice, and Studio Research/Slides all call `get_active_model_key()` — so changing the Settings dropdown only updates the status panel, not the model used by those tabs.

---

## Step 1 — Add MiniCPM5-1B GGUF preset (OpenBMB + llama.cpp)

Official GGUF is published at [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) (`MiniCPM5-1B-Q4_K_M.gguf`, ~657 MB). Add to [`models.yaml`](models.yaml):

```yaml
  minicpm5-1b-gguf:
    label: MiniCPM5 1B (GGUF / llama.cpp)
    backend: llama_cpp
    model_repo: openbmb/MiniCPM5-1B-GGUF
    model_file: MiniCPM5-1B-Q4_K_M.gguf
    n_ctx: 8192
    n_gpu_layers: 0
```

Also update [`.env.example`](.env.example) with a commented dev block:

```bash
ALLOW_MODEL_SWITCH=true
ACTIVE_MODEL=minicpm5-1b          # startup default
# switch in UI to minicpm5-1b-gguf for llama.cpp
```

Prefetch locally (optional, speeds first load):

```bash
uv run python scripts/download_model.py --preset minicpm5-1b-gguf
```

This satisfies the **Llama Champion** path while keeping the OpenBMB / Tiny Titan story (same model, different runtime). LoRA/merged lesson presets remain **transformers-only** — llama.cpp cannot load PEFT adapters.

---

## Step 2 — Shared runtime model state

Extend [`model_loading.py`](apps/gradio-space/src/gradio_space/model_loading.py):

| Function | Behavior |
|----------|----------|
| `set_runtime_model_key(key: str) -> str` | Validate key exists; if changed, call `reset_backend()` and clear load cache for old key; return label for UI |
| `get_active_model_key()` | Return `_runtime_model_key` if set, else `_app_config.active_model` |
| `reload_model(key)` | Also call `set_runtime_model_key(key)` so reload pins the selection app-wide |

This is a small, centralized change — every tab that already calls `get_active_model_key()` will automatically respect the runtime selection once Settings updates it.

---

## Step 3 — Classic Gradio UI wiring

### Settings panel ([`settings_panel.py`](apps/gradio-space/src/gradio_space/ui/settings_panel.py))

On dropdown `.change`:
1. Call `set_runtime_model_key(selected_key)`
2. Update status markdown (existing `model_status`)
3. Optionally auto-reload weights (or keep explicit "Reload model" button — recommend **reload on change** for dev UX)

Return `model_dropdown` from `build_settings_panel()` (already does) and expose it to [`app.py`](apps/gradio-space/src/gradio_space/app.py) if needed for cross-tab sync.

### Chat tab ([`tabs/chat.py`](apps/gradio-space/src/gradio_space/tabs/chat.py))

When `allow_model_switch` is on:
- On Chat model dropdown change → `set_runtime_model_key(mkey)` so Chat and Settings stay in sync
- Default dropdown value = `get_active_model_key()` (runtime-aware)

### App header badge (small UX win)

When `allow_model_switch` is false, keep current read-only badge. When true, show active preset + backend in Settings accordion header so devs always know which backend is live.

---

## Step 4 — Studio UI wiring

In [`api/studio.py`](apps/gradio-space/src/gradio_space/api/studio.py):

- Add `api_set_active_model(model_key: str)` → calls `set_runtime_model_key`, returns updated `model_status`
- Register as `@server.api(name="set_active_model")`
- `api_model_choices()` should report `active_model=get_active_model_key()` (runtime-aware)
- `api_reload_model()` already accepts `model_key`; ensure it calls `set_runtime_model_key` too

In [`static/studio/studio.js`](apps/gradio-space/static/studio/studio.js) `initSettings()`:
- On `#settings-model-key` change → `callApi("set_active_model", [key])` then refresh status
- Keep debug chat dropdown in sync with settings dropdown

Studio Research + Slides already delegate to helpers that use `get_active_model_key()` — no per-endpoint `model_key` param needed once runtime state exists.

---

## Step 5 — Dev workflow (how you use it)

```bash
# .env
ALLOW_MODEL_SWITCH=true
ACTIVE_MODEL=minicpm5-1b
```

```bash
uv sync --all-packages
uv run --package gradio-space python -m gradio_space.server
```

| Goal | Action |
|------|--------|
| Transformers MiniCPM5 | Select `minicpm5-1b` in Settings (or leave startup default) |
| llama.cpp MiniCPM5 (Llama track) | Select `minicpm5-1b-gguf` — backend switches automatically |
| Fine-tuned lesson LoRA | Select `minicpm5-1b-lesson-lora` (transformers only) |
| Compare Qwen GGUF baseline | Select `qwen3b-gguf` |

**There is no separate "backend" dropdown** — backend follows the preset. Dropdown labels already include backend hints; optionally prefix choices with `[llama.cpp]` / `[transformers]` in `model_choices()` for clarity.

### Compatibility notes to surface in Settings status

- `minicpm-v-4.6` (multimodal) requires transformers — warn if selected while on llama_cpp-only tabs
- LoRA/merged local presets require transformers
- First llama.cpp load downloads ~657 MB GGUF from Hub (subsequent loads use cache)

---

## Step 6 — Tests and docs

- Extend [`libs/inference/tests/test_config.py`](libs/inference/tests/test_config.py) to assert `minicpm5-1b-gguf` parses with `backend=llama_cpp`
- Add a small unit test for `set_runtime_model_key` / `get_active_model_key` override in gradio-space tests (or inference tests if kept in `model_loading.py`)
- Add a short "Switching models locally" subsection to [`USAGE.md`](USAGE.md) and [`apps/gradio-space/README.md`](apps/gradio-space/README.md)

---

## Architecture after changes

```mermaid
sequenceDiagram
  participant Dev as Dev_UI_Settings
  participant ML as model_loading
  participant Factory as inference_factory
  participant Tab as Any_Tab_or_Studio_API

  Dev->>ML: set_runtime_model_key("minicpm5-1b-gguf")
  ML->>Factory: reset_backend()
  Tab->>ML: get_active_model_key()
  ML-->>Tab: "minicpm5-1b-gguf"
  Tab->>ML: ensure_model_loaded(key)
  ML->>Factory: get_backend(key).load()
  Note over Factory: LlamaCppBackend for GGUF preset
```

---

## Out of scope (per your choices)

- Pinning HF Space to Llama GGUF for judges (deployment config only — set `ACTIVE_MODEL=minicpm5-1b-gguf` in Space secrets; keep `ALLOW_MODEL_SWITCH=false`)
- Converting fine-tuned LoRA weights to GGUF (OpenBMB documents `convert_hf_to_gguf.py` if needed later)
- Separate backend-only toggle (preset-based switching is simpler and already matches factory design)
