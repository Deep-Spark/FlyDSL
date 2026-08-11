# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Guards that limit sensitive debug surfaces on packaged installs.

Release wheels must not let external users dump backend intermediate IR or ISA
via ``FLYDSL_DUMP_IR`` / related debug knobs. Source checkouts and editable
installs keep the full dump path for developers. Packaged installs ignore
``FLYDSL_ALLOW_IR_DUMP`` and other dump-related environment overrides.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

_pass_pipeline_authorized: ContextVar[bool] = ContextVar("flydsl_pass_pipeline_authorized", default=False)
_passmanager_guard_installed = False


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
def is_packaged_install() -> bool:
    """True for a normal pip-installed wheel/sdist (not source tree or editable)."""
    if _running_from_source_tree():
        return False

    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:  # pragma: no cover
        return False

    try:
        dist = distribution("flydsl")
    except PackageNotFoundError:
        return False

    return not _distribution_is_editable(dist)


@lru_cache(maxsize=1)
def ir_dump_allowed() -> bool:
    """Whether IR/ISA dump surfaces may be enabled in this process.

    Allowed when:
    - ``flydsl`` is imported from this repository's source tree, or
    - ``flydsl`` is not installed as a packaged distribution, or
    - ``flydsl`` is an editable install.

    Denied for normal ``pip install`` of a wheel/sdist (release artifact).
    ``FLYDSL_ALLOW_IR_DUMP`` is ignored.
    """
    return not is_packaged_install()


def assert_ir_dump_allowed(*, feature: str = "FLYDSL_DUMP_IR") -> None:
    """Raise if a packaged wheel install tries to enable IR/ISA dump features."""
    if ir_dump_allowed():
        return
    raise RuntimeError(
        f"{feature} is disabled for packaged FlyDSL installs (wheel/sdist). "
        "Backend intermediate IR and ISA dumps are not available in release "
        "artifacts. Use a source checkout or editable install for development dumps."
    )


def clear_ir_dump_allowed_cache() -> None:
    """Reset the cached allow decision (tests only)."""
    is_packaged_install.cache_clear()
    ir_dump_allowed.cache_clear()


@contextmanager
def authorize_pass_pipeline():
    """Allow PassManager.parse for the official compiler pipeline only."""
    token = _pass_pipeline_authorized.set(True)
    try:
        yield
    finally:
        _pass_pipeline_authorized.reset(token)


def assert_passmanager_parse_allowed() -> None:
    """Raise if a packaged install tries to parse a pass pipeline without authorization."""
    if is_packaged_install() and not _pass_pipeline_authorized.get():
        raise RuntimeError(
            "PassManager.parse is disabled for packaged FlyDSL installs. "
            "Use @flyc.jit / @flyc.kernel compilation rather than constructing "
            "backend pass pipelines directly."
        )


def assert_isa_format_allowed() -> None:
    """Raise if a packaged or stripped build tries to emit format=isa."""
    from .dump_support import DUMP_SUPPORT

    if is_packaged_install() or not DUMP_SUPPORT:
        raise RuntimeError("gpu-module-to-binary{format=isa} is disabled for packaged FlyDSL installs.")


def install_passmanager_guard() -> None:
    """Wrap flydsl._mlir.passmanager.PassManager.parse on packaged installs."""
    global _passmanager_guard_installed
    if _passmanager_guard_installed:
        return
    if not is_packaged_install():
        return
    try:
        from flydsl._mlir.passmanager import PassManager
    except ImportError:  # pragma: no cover
        return
    original = PassManager.parse
    if getattr(original, "_flydsl_passmanager_guarded", False):
        _passmanager_guard_installed = True
        return

    def parse(pipeline, *args, **kwargs):
        assert_passmanager_parse_allowed()
        return original(pipeline, *args, **kwargs)

    parse._flydsl_passmanager_guarded = True
    PassManager.parse = parse
    _passmanager_guard_installed = True
