"""Filesystem-backed artifacts for large runtime outputs."""

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4


DEFAULT_DATA_DIR = Path.home() / ".agent-runtime"


def artifact_dir(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    base = Path(os.getenv("AGENT_RUNTIME_DATA_DIR", DEFAULT_DATA_DIR)).expanduser()
    return base / "artifacts"


def write_text_artifact(
    content: str,
    *,
    label: str,
    suffix: str = ".txt",
    base_dir: Path | str | None = None,
) -> str:
    root = artifact_dir(base_dir)
    day_dir = root / time.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label)
    filename = f"{int(time.time() * 1000)}-{uuid4().hex[:10]}-{safe_label}{suffix}"
    path = day_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)
