import gradio as gr

from gradio_space.research_helpers import (
    list_session_choices,
    rag_aware_chat,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
)
from gradio_space.ui.components import build_advanced_panel, DOC_CHOICE_LIST_CLASSES, tab_hero
from inference.config import get_app_config

_app_config = get_app_config()


def build_chat_tab() -> None:
    tab_hero(
        "Test the active local model with optional ResearchMind RAG.",
    )
    gr.HTML(
        '<span class="dev-tab-badge">Developer</span> '
        "Plain chat or corpus-grounded answers — traces appear in Advanced when RAG is on."
    )

    model_key = _app_config.active_model

    with gr.Group():
        gr.Markdown("#### RAG scope")
        with gr.Row():
            use_rag = gr.Checkbox(label="Use ResearchMind RAG", value=False)
            session_dd = gr.Dropdown(
                label="Session",
                choices=list_session_choices(),
                value="",
                interactive=True,
                scale=3,
            )
            refresh_sessions_btn = gr.Button("↻", size="sm", scale=0, min_width=40)

        doc_dd = gr.CheckboxGroup(
            label="Documents to search (empty = all docs in session, or entire corpus if no session)",
            choices=[],
            value=[],
            elem_classes=DOC_CHOICE_LIST_CLASSES,
        )
        rag_hint = gr.Markdown(value=rag_scope_hint("", []))

    advanced = build_advanced_panel()

    if _app_config.allow_model_switch and len(_app_config.models) > 1:
        model_dropdown = gr.Dropdown(
            choices=_app_config.model_choices(),
            value=_app_config.active_model,
            label="Model preset (debug override)",
        )

        def _chat(message, history, mkey, use_rag_flag, sid, docs):
            reply, trace_json, trace_summary = rag_aware_chat(
                message, history, mkey, use_rag_flag, sid, docs
            )
            return reply, trace_json, trace_summary

        chat_iface = gr.ChatInterface(
            fn=_chat,
            additional_outputs=[advanced.trace_box, advanced.trace_summary],
            additional_inputs=[model_dropdown, use_rag, session_dd, doc_dd],
            examples=[
                [
                    "What do my ingested sources say about AI agents?",
                    _app_config.active_model,
                    True,
                    "",
                    [],
                ],
                [
                    "Hello! What can you help me with?",
                    _app_config.active_model,
                    False,
                    "",
                    [],
                ],
            ],
        )
    else:

        def _chat(message, history, use_rag_flag, sid, docs):
            reply, trace_json, trace_summary = rag_aware_chat(
                message, history, model_key, use_rag_flag, sid, docs
            )
            return reply, trace_json, trace_summary

        chat_iface = gr.ChatInterface(
            fn=_chat,
            additional_outputs=[advanced.trace_box, advanced.trace_summary],
            additional_inputs=[use_rag, session_dd, doc_dd],
            examples=[
                ["What do my ingested sources say about AI agents?", True, "", []],
                ["Hello! What can you help me with?", False, "", []],
            ],
        )

    _ = chat_iface  # keep reference for linter

    def _update_hint(sid: str, docs: list[str] | None, rag_on: bool) -> str:
        if not rag_on:
            return "_Plain chat — model only, no document retrieval._"
        return rag_scope_hint(sid, docs)

    refresh_sessions_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    session_dd.change(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    ).then(
        fn=_update_hint,
        inputs=[session_dd, doc_dd, use_rag],
        outputs=[rag_hint],
    )
    doc_dd.change(fn=_update_hint, inputs=[session_dd, doc_dd, use_rag], outputs=[rag_hint])
    use_rag.change(fn=_update_hint, inputs=[session_dd, doc_dd, use_rag], outputs=[rag_hint])
