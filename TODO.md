# Hackathon badge/track TODO

Strategy: one **Backyard AI** submission, stacking as many merit badges, sponsor
awards, and special awards as credibly fit the small-model / local-first story.
Deadline: **June 15, 2026** (Space + demo video + social post).

This PR (`feat/finetuning_model`) focuses on **🎯 Well-Tuned** + **Modal**. Everything
below is parked for follow-up PRs.

## In this PR (finetuning + Modal) — done here
- [x] Make published adapters **public** so judges can verify the Well-Tuned badge
      (`research/modal/experiments.yaml`: `private: false`).
- [x] Add hackathon discoverability tags + license to the published model card
      (`research/modal/_common.py: render_model_card`).

## 🦙 Llama Champion badge (cheap, high value)
- [ ] Run the Space on the **llama.cpp / GGUF** backend (`libs/inference/src/inference/llama_cpp.py`).
- [x] Add `minicpm-v-4.6-gguf` preset (`openbmb/MiniCPM-V-4.6-gguf`) — OpenBMB multimodal on llama.cpp.
- [ ] Document the llama.cpp path in README + Space (`ACTIVE_MODEL=minicpm-v-4.6-gguf`).

## 📓 Field Notes badge (cheapest miss — no blog exists yet)
- [ ] Write a blog post / report on the fine-tuning + Modal pipeline:
      skill-matrix QLoRA -> lm-eval -> per-skill gate -> Hub publish.
- [ ] Publish it (HF blog / personal) and link from README.
- [ ] This badge + the others clinches **Bonus Quest Champion ($2k)**.

## README + submission hygiene
- [ ] Update README badge checklist to reflect full strategy (add Llama Champion, Field Notes).
- [ ] Best Demo: polished demo video (real teacher -> topic -> .pptx download -> trace).
- [ ] Social post published (required for submission).
- [ ] Community Choice: share the Space widely.

## Decided NOT to chase (conflicts with MiniCPM / local-first core)
- OpenAI Track — requires OpenAI models; collides with Tiny Titan / OpenBMB / Off-the-Grid.
- NVIDIA Nemotron — requires Nemotron model; same conflict.
- Thousand Token Wood — different main track; can't be in both.

## Badge scorecard (target = all 6 + Bonus Quest Champion)
- [x] 🔌 Off the Grid — local inference only
- [x] 🎨 Off-Brand — custom Studio UI (Gradio 6 Server mode)
- [x] 📡 Sharing is Caring — agent trace upload
- [~] 🎯 Well-Tuned — pipeline ready; needs a passing public adapter on the Hub
- [ ] 🦙 Llama Champion — see above
- [ ] 📓 Field Notes — see above
