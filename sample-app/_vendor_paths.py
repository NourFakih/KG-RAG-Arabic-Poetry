from __future__ import annotations

import sys
from pathlib import Path


def _candidate_dirs(base: Path):
    """
    Yield plausible locations for the cloned nano-graphrag repo relative to `base`.
    """
    suffixes = ("nano-graphrag", "nano_graphrag")
    for ancestor in (base, base.parent, base.parent.parent):
        if ancestor is None:
            continue
        for suffix in suffixes:
            yield ancestor / suffix


def _bootstrap() -> str | None:
    here = Path(__file__).resolve().parent
    for candidate in _candidate_dirs(here):
        package_dir = candidate / "nano_graphrag"
        if package_dir.exists():
            path_str = str(candidate)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path_str
    return None


NANO_GRAPHRAG_PATH = _bootstrap()
