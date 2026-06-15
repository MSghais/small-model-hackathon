import gradio as gr

from gradio_space.model_loading import get_active_model_key, set_runtime_model_key
from gradio_space.research_helpers import (
    list_session_choices,
    rag_aware_chat,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
    resolve_doc_ids,
    resolve_session,
)
from gradio_space.ui.components import (
    build_advanced_panel,
    DOC_CHOICE_LIST_CLASSES,
    tab_hero,
    WorkspaceWidgets,
)
from inference.config import get_app_config

_app_config = get_app_config()


def build_chat_tab(workspace: WorkspaceWidgets) -> None:
    tab_hero(
        "Test the active local model with optional ResearchMind RAG.",
    )
    gr.HTML(
        '<span class="dev-tab-badge">Developer</span> '
        "Plain chat or corpus-grounded answers — traces appear in Advanced when RAG is on."
    )

    model_key = get_active_model_key()

    with gr.Group():
        gr.Markdown("#### RAG scope (override workspace defaults)")
        with gr.Row():
            use_rag = gr.Checkbox(label="Use ResearchMind RAG", value=False)
            session_dd = gr.Dropdown(
                label="Session (empty = workspace default)",
                choices=list_session_choices(),
                value="",
                interactive=True,
                scale=3,
            )
            refresh_sessions_btn = gr.Button("↻", size="sm", scale=0, min_width=40)

        doc_dd = gr.CheckboxGroup(
            label="Documents (empty = workspace default or all in session)",
            choices=[],
            value=[],
            elem_classes=DOC_CHOICE_LIST_CLASSES,
        )
        rag_hint = gr.Markdown(value=rag_scope_hint("", []))

    advanced = build_advanced_panel()

    if _app_config.allow_model_switch and len(_app_config.models) > 1:
        model_dropdown = gr.Dropdown(
            choices=_app_config.model_choices(),
            value=get_active_model_key(),
            label="Model preset (debug override)",
        )

        def _on_model_change(mkey: str) -> None:
            set_runtime_model_key(mkey)

        model_dropdown.change(fn=_on_model_change, inputs=model_dropdown)

        def _chat(message, history, mkey, use_rag_flag, sid, docs, ws_sid, ws_docs):
            set_runtime_model_key(mkey)
            sid = resolve_session(sid, ws_sid)
            docs = resolve_doc_ids(docs, ws_docs)
            reply, trace_json, trace_summary = rag_aware_chat(
                message, history, mkey, use_rag_flag, sid, docs
            )
            return reply, trace_json, trace_summary

        chat_iface = gr.ChatInterface(
            fn=_chat,
            additional_outputs=[advanced.trace_box, advanced.trace_summary],
            additional_inputs=[
                model_dropdown,
                use_rag,
                session_dd,
                doc_dd,
                workspace.session_dd,
                workspace.doc_dd,
            ],
            examples=[
                [
                    "What do my ingested sources say about AI agents?",
                    get_active_model_key(),
                    True,
                    "",
                    [],
                    "",
                    [],
                ],
                [
                    "Hello! What can you help me with?",
                    get_active_model_key(),
                    False,
                    "",
                    [],
                    "",
                    [],
                ],
            ],
        )
    else:

        def _chat(message, history, use_rag_flag, sid, docs, ws_sid, ws_docs):
            sid = resolve_session(sid, ws_sid)
            docs = resolve_doc_ids(docs, ws_docs)
            reply, trace_json, trace_summary = rag_aware_chat(
                message, history, model_key, use_rag_flag, sid, docs
            )
            return reply, trace_json, trace_summary

        chat_iface = gr.ChatInterface(
            fn=_chat,
            additional_outputs=[advanced.trace_box, advanced.trace_summary],
            additional_inputs=[
                use_rag,
                session_dd,
                doc_dd,
                workspace.session_dd,
                workspace.doc_dd,
            ],
            examples=[
                ["What do my ingested sources say about AI agents?", True, "", [], "", []],
                ["Hello! What can you help me with?", False, "", [], "", []],
            ],
        )

    _ = chat_iface  # keep reference for linter

    def _update_hint(
        sid: str,
        docs: list[str] | None,
        rag_on: bool,
        ws_sid: str,
        ws_docs: list[str] | None,
    ) -> str:
        if not rag_on:
            return "_Plain chat — model only, no document retrieval._"
        sid = resolve_session(sid, ws_sid)
        docs = resolve_doc_ids(docs, ws_docs)
        return rag_scope_hint(sid, docs)

    refresh_sessions_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    session_dd.change(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    ).then(
        fn=_update_hint,
        inputs=[session_dd, doc_dd, use_rag, workspace.session_dd, workspace.doc_dd],
        outputs=[rag_hint],
    )
    doc_dd.change(
        fn=_update_hint,
        inputs=[session_dd, doc_dd, use_rag, workspace.session_dd, workspace.doc_dd],
        outputs=[rag_hint],
    )
    use_rag.change(
        fn=_update_hint,
        inputs=[session_dd, doc_dd, use_rag, workspace.session_dd, workspace.doc_dd],
        outputs=[rag_hint],
    )

    def _sync_session_from_workspace(ws_session: str, local_session: str) -> str:
        if not (local_session or "").strip():
            return ws_session
        return local_session

    workspace.session_dd.change(
        fn=_sync_session_from_workspace,
        inputs=[workspace.session_dd, session_dd],
        outputs=[session_dd],
    ).then(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    )
