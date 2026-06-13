from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gradio import mount_gradio_app

from gradio_space.api.studio import register_studio_apis
from gradio_space.app import build_demo
from gradio_space.model_loading import preload_active_model
from gradio_space.tabs.education_pptx import gradio_allowed_paths
from gradio_space.tabs.echo_coach import echo_coach_allowed_paths
from gradio_space.tabs.research_mind import researchmind_allowed_paths
from gradio_space.tabs.teacher_voice import teacher_voice_allowed_paths
from gradio_space.ui.theme import get_theme, load_css

_PKG_ROOT = Path(__file__).resolve().parent
_APP_ROOT = _PKG_ROOT.parents[1]
_STATIC_DIR = _APP_ROOT / "static" / "studio"


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


def create_server() -> gr.Server:
    server = gr.Server(title="Build Small Studio")

    register_studio_apis(server)

    if _STATIC_DIR.is_dir():
        server.mount("/static/studio", StaticFiles(directory=str(_STATIC_DIR)), name="studio_static")

    @server.get("/")
    async def studio_index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @server.get("/studio")
    async def studio_alias() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

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
    )


if __name__ == "__main__":
    main()
