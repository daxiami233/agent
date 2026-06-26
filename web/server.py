"""FastAPI server for the React chat UI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_runtime.logging import format_runtime_log_content, runtime_log_path

from .session import WebEvent, WebSession


STATIC_DIR = Path(__file__).with_name("static")


class ChatRequest(BaseModel):
    """Incoming user message."""

    conversation_id: str = ""
    message: str = ""
    request_id: str
    reasoning_enabled: bool = True


class ConversationRequest(BaseModel):
    """Create or ensure a conversation."""

    id: str | None = None
    title: str = "新对话"
    project_id: str | None = None


class ProjectRequest(BaseModel):
    """Add a project directory to the web workspace."""

    path: str
    name: str | None = None


class SelectProjectRequest(BaseModel):
    """Select a project without creating a conversation."""

    project_id: str


class CancelRequest(BaseModel):
    """Cancel an active generation."""

    request_id: str


def create_app(session: WebSession | None = None) -> FastAPI:
    """Build the FastAPI app around a shared local session."""

    web_session = session or WebSession()
    app = FastAPI(title="Agent Runtime")
    app.state.session = web_session

    @app.middleware("http")
    async def no_store(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/bootstrap")
    def bootstrap() -> JSONResponse:
        return JSONResponse(
            {"events": [_event_payload(event) for event in web_session.boot_events()]}
        )

    @app.get("/api/commands")
    def commands() -> JSONResponse:
        return JSONResponse(
            {
                "commands": [
                    {"name": name, "description": description}
                    for name, description in web_session.commands.completion_rows()
                ]
            }
        )

    @app.get("/api/logs/runtime")
    def runtime_logs() -> JSONResponse:
        path = runtime_log_path()
        if not path.exists():
            return JSONResponse(
                {"path": str(path), "content": "", "exists": False, "error": ""}
            )
        try:
            size = path.stat().st_size
            tail_bytes = 512 * 1024
            with path.open("rb") as f:
                if size > tail_bytes:
                    f.seek(size - tail_bytes)
                    f.readline()
                content = f.read().decode("utf-8", errors="replace")
        except OSError as exc:
            return JSONResponse(
                {
                    "path": str(path),
                    "content": "",
                    "exists": True,
                    "error": str(exc),
                },
                status_code=500,
            )
        return JSONResponse(
            {
                "path": str(path),
                "content": format_runtime_log_content(content),
                "exists": True,
                "error": "",
            }
        )

    @app.delete("/api/logs/runtime")
    def clear_runtime_logs() -> JSONResponse:
        path = runtime_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        except OSError as exc:
            return JSONResponse(
                {
                    "path": str(path),
                    "content": "",
                    "exists": path.exists(),
                    "error": str(exc),
                },
                status_code=500,
            )
        return JSONResponse(
            {"path": str(path), "content": "", "exists": True, "error": ""}
        )

    @app.get("/api/conversations")
    def conversations(project_id: str | None = None) -> JSONResponse:
        return JSONResponse({"items": web_session.list_conversations(project_id)})

    @app.post("/api/conversations")
    def create_conversation(payload: ConversationRequest) -> JSONResponse:
        return JSONResponse(
            web_session.create_conversation(
                conversation_id=payload.id,
                title=payload.title,
                project_id=payload.project_id,
            )
        )

    @app.get("/api/projects")
    def projects() -> JSONResponse:
        return JSONResponse({"items": web_session.list_projects()})

    @app.post("/api/projects")
    def create_project(payload: ProjectRequest) -> JSONResponse:
        try:
            return JSONResponse(web_session.add_project(payload.path, payload.name))
        except (OSError, RuntimeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/projects/pick")
    def pick_project() -> JSONResponse:
        try:
            path = _pick_directory()
            if not path:
                return JSONResponse({"cancelled": True})
            project = web_session.add_project(path)
            return JSONResponse({"cancelled": False, "project": project})
        except (OSError, RuntimeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/projects/select")
    def select_project(payload: SelectProjectRequest) -> JSONResponse:
        try:
            return JSONResponse(web_session.select_project(payload.project_id))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> JSONResponse:
        try:
            web_session.delete_project(project_id)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse({"ok": True})

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> JSONResponse:
        web_session.delete_conversation(conversation_id)
        return JSONResponse({"ok": True})

    @app.post("/api/cancel")
    def cancel(payload: CancelRequest) -> JSONResponse:
        web_session.cancel(payload.request_id)
        return JSONResponse({"ok": True})

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> StreamingResponse:
        conversation_id = payload.conversation_id or web_session.create_conversation()["id"]
        stream = _stream_events(
            web_session.submit(
                conversation_id,
                payload.message,
                request_id=payload.request_id,
                reasoning_enabled=payload.reasoning_enabled,
            ),
            session=web_session,
            request_id=payload.request_id,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.get("/")
    @app.head("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount(
        "/assets",
        StaticFiles(directory=STATIC_DIR / "assets", check_dir=False),
        name="assets",
    )

    @app.get("/{path:path}")
    def fallback(path: str) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _stream_events(
    events: Iterator[WebEvent],
    *,
    session: WebSession,
    request_id: str,
) -> Iterator[str]:
    try:
        for event in events:
            yield _format_sse(event)
        yield _format_sse(WebEvent("done", {}))
    finally:
        session.finish_request(request_id)


def _format_sse(event: WebEvent) -> str:
    payload = json.dumps(_event_payload(event), ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n"


def _event_payload(event: WebEvent) -> dict[str, Any]:
    return {"type": event.type, **event.payload}


def _pick_directory() -> str:
    if sys.platform == "darwin":
        return _pick_directory_macos()
    return _pick_directory_tkinter()


def _pick_directory_macos() -> str:
    script = 'POSIX path of (choose folder with prompt "选择项目文件夹")'
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    detail = (result.stderr or result.stdout).strip()
    if result.returncode == 1 and "-128" in detail:
        return ""
    raise RuntimeError(detail or "无法打开系统文件夹选择器")


def _pick_directory_tkinter() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境不支持系统文件夹选择器") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(title="选择项目文件夹", mustexist=True)
    finally:
        root.destroy()


def run_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    app = create_app()
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Thread(target=_open_browser_soon, args=(url,), daemon=True).start()
    print(f"Agent Runtime web UI running at {url}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _open_browser_soon(url: str) -> None:
    time.sleep(0.4)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agent Runtime web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    run_server(args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
