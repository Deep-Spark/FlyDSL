# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Guards that limit sensitive debug surfaces on packaged installs.

Release wheels must not let external users dump backend intermediate IR or ISA
via ``FLYDSL_DUMP_IR`` / related debug knobs. Source checkouts and editable
installs keep the full dump path for developers.

Internal override (not for external docs): set ``FLYDSL_ALLOW_IR_DUMP=1``.
"""

import json
import os
from functools import lru_cache
from pathlib import Path


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _distribution_is_editable(dist) -> bool:
    """Return True when ``dist`` is an editable install (PEP 610 direct_url)."""
    try:
        raw = dist.read_text("direct_url.json")
    except FileNotFoundError:
        return False
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(data.get("dir_info", {}).get("editable"))


def _running_from_source_tree() -> bool:
    """True when imported ``flydsl`` still lives in this repository layout."""
    try:
        import flydsl
    except ImportError:
        return False
    pkg_dir = Path(flydsl.__file__).resolve().parent
    # Expected layout: <repo>/python/flydsl/__init__.py
    repo_root = pkg_dir.parent.parent
    return (repo_root / "setup.py").is_file() and (repo_root / "python" / "flydsl").is_dir()


@lru_cache(maxsize=1)
def ir_dump_allowed() -> bool:
    """Whether IR/ISA dump surfaces may be enabled in this process.

    Allowed when:
    - ``FLYDSL_ALLOW_IR_DUMP`` is truthy, or
    - ``flydsl`` is imported from this repository's source tree, or
    - ``flydsl`` is not installed as a packaged distribution, or
    - ``flydsl`` is an editable install.

    Denied for normal ``pip install`` of a wheel/sdist (release artifact).
    """
    if _env_truthy("FLYDSL_ALLOW_IR_DUMP"):
        return True

    if _running_from_source_tree():
        return True

    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:  # pragma: no cover
        return True

    try:
        dist = distribution("flydsl")
    except PackageNotFoundError:
        return True

    if _distribution_is_editable(dist):
        return True
    return False


def assert_ir_dump_allowed(*, feature: str = "FLYDSL_DUMP_IR") -> None:
    """Raise if a packaged wheel install tries to enable IR/ISA dump features."""
    if ir_dump_allowed():
        return
    raise RuntimeError(
        f"{feature} is disabled for packaged FlyDSL installs (wheel/sdist). "
        "Backend intermediate IR and ISA dumps are not available in release "
        "artifacts. Use a source checkout or editable install for development "
        "dumps, or set FLYDSL_ALLOW_IR_DUMP=1 only in trusted internal environments."
    )


def clear_ir_dump_allowed_cache() -> None:
    """Reset the cached allow decision (tests only)."""
    ir_dump_allowed.cache_clear()
