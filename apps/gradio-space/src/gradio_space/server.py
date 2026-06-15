from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from gradio import mount_gradio_app

from gradio_space.api.studio import register_studio_apis
from gradio_space.app import build_demo
from gradio_space.model_loading import preload_active_model
from gradio_space.spaces_runtime import is_hf_gradio_runtime
from gradio_space.tabs.education_pptx import gradio_allowed_paths
from gradio_space.tabs.echo_coach import echo_coach_allowed_paths
from gradio_space.tabs.research_mind import researchmind_allowed_paths
from gradio_space.tabs.teacher_voice import teacher_voice_allowed_paths
from gradio_space.ui.theme import get_theme, load_css

_PKG_ROOT = Path(__file__).resolve().parent
_APP_ROOT = _PKG_ROOT.parents[1]
_STATIC_DIR = _APP_ROOT / "static" / "studio"
_STUDIO_ASSET_VERSION = "20260615d"
_STUDIO_INDEX_HTML = _STATIC_DIR / "index.html"


def _studio_index_html() -> str:
    return _STUDIO_INDEX_HTML.read_text().replace(
        "{{STUDIO_ASSET_VERSION}}",
        _STUDIO_ASSET_VERSION,
    )


def _all_allowed_paths() -> list[str]:
    paths: list[str] = []
    for fn in (
        gradio_allowed_paths,
        researchmind_allowed_paths,
        echo_coach_allowed_paths,
        teacher_voice_allowed_paths,
    ):
        paths.extend(fn())
    return list(dict.fromkeys(paths))


def _register_hf_https_middleware(server: gr.Server) -> None:
    """HF terminates TLS; the app sees HTTP and Gradio may emit http:// asset URLs."""
    if not os.environ.get("SPACE_ID"):
        return

    @server.middleware("http")
    async def force_https_scheme(request, call_next):
        request.scope["scheme"] = "https"
        return await call_next(request)


def _wants_classic_ui(request: Request) -> bool:
    return "classic" in request.query_params


def create_server() -> gr.Server:
    server = gr.Server(title="Build Small Studio")
    _register_hf_https_middleware(server)

    register_studio_apis(server)

    if _STATIC_DIR.is_dir():
        server.mount("/static/studio", StaticFiles(directory=str(_STATIC_DIR)), name="studio_static")

    @server.get("/")
    async def studio_index(request: Request):
        if _wants_classic_ui(request):
            # Relative path keeps huggingface.co/spaces/.../classic on HTTPS (not http hf.space).
            return RedirectResponse(url="classic", status_code=302)
        return HTMLResponse(_studio_index_html())

    @server.get("/studio")
    async def studio_alias(request: Request):
        if _wants_classic_ui(request):
            return RedirectResponse(url="classic", status_code=302)
        return HTMLResponse(_studio_index_html())

    demo = build_demo()
    mount_gradio_app(
        server,
        demo,
        path="/classic",
        theme=get_theme(),
        css=load_css(),
        allowed_paths=_all_allowed_paths(),
        footer_links=[],
    )

    return server


def main() -> None:
    if not is_hf_gradio_runtime():
        preload_active_model()
    server = create_server()
    port = int(os.environ.get("PORT", "7860"))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    print(
        f"\n  Build Small Studio: http://127.0.0.1:{port}/\n"
        f"  Classic Gradio UI:  http://127.0.0.1:{port}/classic\n"
        f"  Bound address: {server_name}:{port}\n"
    )
    server.launch(
        server_name=server_name,
        server_port=port,
        footer_links=[],
        allowed_paths=_all_allowed_paths(),
        show_error=True,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
