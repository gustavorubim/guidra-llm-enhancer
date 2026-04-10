from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def _editable_project_root() -> Path | None:
    try:
        distribution = metadata.distribution("decomp-clarifier")
    except metadata.PackageNotFoundError:
        return None

    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    payload = json.loads(raw)
    if not payload.get("dir_info", {}).get("editable"):
        return None

    parsed = urlparse(payload["url"])
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    return Path(path)


def install() -> None:
    project_root = _editable_project_root()
    if project_root is None:
        return
    src_path = project_root / "src"
    if not src_path.is_dir():
        return
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


install()
