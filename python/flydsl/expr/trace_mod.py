# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Traced score_mod / mask_mod fragments for FlyDSL (V3-1).

Decorators produce reusable ``TracedScoreMod`` / ``TracedMaskMod`` objects that:
* validate a restricted ``fx.*`` body at decorate time
* expose ``apply(...)`` for inlining into ``@flyc.kernel`` traces
* expose ``eval_host(...)`` by calling the original Python function
* expose a stable ``fingerprint`` for later JIT cache keys (V3-2)

This module is target-neutral. Attention kernels are not modified here.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

__all__ = [
    "TracedScoreMod",
    "TracedMaskMod",
    "trace_score_mod",
    "trace_mask_mod",
    "where",
    "SCORE_MOD_PARAMS",
    "MASK_MOD_PARAMS",
]

SCORE_MOD_PARAMS: Tuple[str, ...] = ("score", "batch", "head", "q_idx", "kv_idx")
MASK_MOD_PARAMS: Tuple[str, ...] = ("batch", "head", "q_idx", "kv_idx")

# Call / attribute names allowed inside a traced mod body (V3-1 whitelist).
_ALLOWED_CALL_NAMES = frozenset(
    {
        "where",
        "select",
        "tanh",
        "Float32",
        "Int32",
        "Boolean",
        "float",
        "int",
        "bool",
    }
)

_FORBIDDEN_STMT_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Raise,
    ast.Assert,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)


def where(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Select ``true_value`` or ``false_value`` on host or in a DSL kernel.

    Host path (plain Python scalars / bools): returns a Python ternary result.
    Kernel / IR path: lowers via ``Numeric.select`` / ``arith.select``.
    """
    # Host-fast path: all plain Python.
    if isinstance(condition, bool) and _is_host_operand(true_value) and _is_host_operand(false_value):
        return true_value if condition else false_value
    if isinstance(condition, (int, float)) and not isinstance(condition, bool):
        # Treat nonzero as true for host ints/floats (rare for preds).
        if _is_host_operand(true_value) and _is_host_operand(false_value):
            return true_value if bool(condition) else false_value

    from .numeric import Boolean, Numeric, as_numeric

    cond = condition
    if not isinstance(cond, Numeric):
        try:
            cond = as_numeric(cond)
        except Exception as exc:
            raise TypeError(f"where() condition must be bool/Numeric, got {type(condition).__name__}") from exc
    if not isinstance(cond, Boolean) and hasattr(cond, "__dsl_bool__"):
        cond = cond.__dsl_bool__()
    if (
        isinstance(cond, Numeric)
        and cond.is_static()
        and _is_host_operand(true_value)
        and _is_host_operand(false_value)
    ):
        return true_value if bool(cond.value) else false_value
    return cond.select(true_value, false_value)


def _is_host_operand(value: Any) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    try:
        from .numeric import Numeric

        return isinstance(value, Numeric) and value.is_static() and isinstance(value.value, (bool, int, float))
    except Exception:
        return False


def _validate_scalar_closure(fn: Callable[..., Any]) -> dict[str, Any]:
    freevars = fn.__code__.co_freevars
    closure = fn.__closure__
    if not freevars:
        return {}
    if closure is None:
        raise ValueError(f"{fn.__name__}: expected closure cells for freevars {freevars}")
    out: dict[str, Any] = {}
    for name, cell in zip(freevars, closure):
        try:
            val = cell.cell_contents
        except ValueError as exc:
            raise ValueError(f"{fn.__name__}: closure {name!r} is empty") from exc
        if isinstance(val, bool):
            out[name] = bool(val)
        elif isinstance(val, int) and not isinstance(val, bool):
            out[name] = int(val)
        elif isinstance(val, float):
            out[name] = float(val)
        else:
            raise ValueError(f"{fn.__name__}: closure {name!r} must be bool/int/float scalar, got {type(val).__name__}")
    return out


def _call_basename(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _validate_mod_ast(fn: Callable[..., Any], *, kind: str) -> ast.AST:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except OSError as exc:
        raise ValueError(f"{fn.__name__}: cannot read source for tracing validation") from exc
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise ValueError(f"{fn.__name__}: cannot parse source: {exc}") from exc

    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError(f"{fn.__name__}: expected a function definition")
    fdef = tree.body[0]
    if isinstance(fdef, ast.AsyncFunctionDef):
        raise ValueError(f"{fn.__name__}: async functions are not supported")

    for stmt in fdef.body:
        for node in ast.walk(stmt):
            if isinstance(node, _FORBIDDEN_STMT_TYPES):
                raise ValueError(
                    f"{fn.__name__}: control-flow / statement {type(node).__name__} "
                    f"is not allowed in V3-1 {kind} (use fx.where instead of if)"
                )
            if isinstance(node, ast.IfExp):
                raise ValueError(f"{fn.__name__}: Python ternary is not allowed in V3-1 {kind}; use fx.where(...)")
            if isinstance(node, ast.Call):
                name = _call_basename(node.func)
                if name is None or name not in _ALLOWED_CALL_NAMES:
                    raise ValueError(
                        f"{fn.__name__}: call to {ast.dump(node.func)!r} is not in the "
                        f"V3-1 whitelist {_ALLOWED_CALL_NAMES}"
                    )
            if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise ValueError(
                    f"{fn.__name__}: binary op {type(node.op).__name__} is not allowed " f"(only + - * / in V3-1)"
                )
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                raise ValueError(f"{fn.__name__}: unary op {type(node.op).__name__} is not allowed")
    return tree


def _validate_signature(fn: Callable[..., Any], expected: Tuple[str, ...], *, kind: str) -> None:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if any(p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in params):
        raise ValueError(f"{fn.__name__}: {kind} must use plain positional parameters only")
    if len(params) != len(expected):
        raise ValueError(f"{fn.__name__}: {kind} expects {len(expected)} params {expected}, got {len(params)}")
    names = tuple(p.name for p in params)
    if names != expected:
        raise ValueError(f"{fn.__name__}: {kind} parameter names must be {expected}, got {names}")


def _fingerprint(kind: str, fn: Callable[..., Any], closure: Mapping[str, Any]) -> str:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        fdef = tree.body[0]
        if isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fdef.name = "_traced_mod"
            fdef.decorator_list = []
            normalized_src = ast.dump(fdef, include_attributes=False)
        else:
            normalized_src = src
    except (OSError, SyntaxError, IndexError):
        normalized_src = fn.__code__.co_code.hex()
    payload = {
        "kind": kind,
        "src": normalized_src,
        "closure": {k: closure[k] for k in sorted(closure)},
    }
    blob = repr(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _as_host_score(score: Any) -> float:
    if isinstance(score, bool):
        raise TypeError("score must be a float, not bool")
    if isinstance(score, (int, float)):
        return float(score)
    try:
        from .numeric import Numeric

        if isinstance(score, Numeric) and score.is_static():
            return float(score.value)
    except Exception:
        pass
    raise TypeError(f"eval_host score must be a host float/int, got {type(score).__name__}")


def _as_host_index(name: str, value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    try:
        from .numeric import Numeric

        if isinstance(value, Numeric) and value.is_static():
            return int(value.value)
    except Exception:
        pass
    raise TypeError(f"eval_host {name} must be a host int, got {type(value).__name__}")


def _host_tanh(x: Any, *, fastmath=None, **kwargs: Any) -> Any:
    """Host-safe tanh for ``eval_host``; falls back to the IR ``tanh`` under a DSL context."""
    import math as _py_math

    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _py_math.tanh(float(x))
    try:
        from .numeric import Numeric

        if isinstance(x, Numeric) and x.is_static() and isinstance(x.value, (int, float)):
            return type(x)(_py_math.tanh(float(x.value)))
    except Exception:
        pass
    ir_tanh = getattr(_host_tanh, "_ir_tanh", None)
    if ir_tanh is None:
        from . import math as math_mod

        ir_tanh = math_mod.tanh
    return ir_tanh(x, fastmath=fastmath, **kwargs)


class _host_math_overrides:
    """Temporarily patch ``fx.tanh`` so traced mods can ``eval_host`` without MLIR."""

    def __enter__(self):
        import flydsl.expr as fx_mod

        from . import math as math_mod

        self._fx = fx_mod
        self._math = math_mod
        self._old_fx = getattr(fx_mod, "tanh", None)
        self._old_math = math_mod.tanh
        _host_tanh._ir_tanh = self._old_math  # type: ignore[attr-defined]
        math_mod.tanh = _host_tanh
        fx_mod.tanh = _host_tanh
        return self

    def __exit__(self, exc_type, exc, tb):
        self._math.tanh = self._old_math
        if self._old_fx is not None:
            self._fx.tanh = self._old_fx
        if getattr(_host_tanh, "_ir_tanh", None) is self._old_math:
            _host_tanh._ir_tanh = None  # type: ignore[attr-defined]
        return False


@dataclass(frozen=True)
class TracedScoreMod:
    """Reusable score_mod fragment (inline via ``apply``, host via ``eval_host``)."""

    _fn: Callable[..., Any]
    _closure: Mapping[str, Any]
    fingerprint: str
    name: str

    @property
    def __name__(self) -> str:
        return self.name

    def apply(self, score: Any, batch: Any, head: Any, q_idx: Any, kv_idx: Any) -> Any:
        """Inline the mod body into the current DSL / kernel trace."""
        return self._fn(score, batch, head, q_idx, kv_idx)

    def eval_host(self, score: Any, batch: Any, head: Any, q_idx: Any, kv_idx: Any) -> float:
        """Evaluate the original Python function on host scalars."""
        with _host_math_overrides():
            out = self._fn(
                _as_host_score(score),
                _as_host_index("batch", batch),
                _as_host_index("head", head),
                _as_host_index("q_idx", q_idx),
                _as_host_index("kv_idx", kv_idx),
            )
        if isinstance(out, bool):
            raise TypeError("score_mod must return a float score, got bool")
        if isinstance(out, (int, float)):
            return float(out)
        try:
            from .numeric import Numeric

            if isinstance(out, Numeric) and out.is_static():
                return float(out.value)
        except Exception:
            pass
        raise TypeError(f"score_mod host result must be float-like, got {type(out).__name__}")


@dataclass(frozen=True)
class TracedMaskMod:
    """Reusable mask_mod fragment (inline via ``apply``, host via ``eval_host``)."""

    _fn: Callable[..., Any]
    _closure: Mapping[str, Any]
    fingerprint: str
    name: str

    @property
    def __name__(self) -> str:
        return self.name

    def apply(self, batch: Any, head: Any, q_idx: Any, kv_idx: Any) -> Any:
        """Inline the mod body into the current DSL / kernel trace."""
        return self._fn(batch, head, q_idx, kv_idx)

    def eval_host(self, batch: Any, head: Any, q_idx: Any, kv_idx: Any) -> bool:
        """Evaluate the original Python function on host scalars."""
        with _host_math_overrides():
            out = self._fn(
                _as_host_index("batch", batch),
                _as_host_index("head", head),
                _as_host_index("q_idx", q_idx),
                _as_host_index("kv_idx", kv_idx),
            )
        if isinstance(out, bool):
            return bool(out)
        if isinstance(out, (int, float)):
            return bool(out)
        try:
            from .numeric import Numeric

            if isinstance(out, Numeric) and out.is_static():
                return bool(out.value)
        except Exception:
            pass
        raise TypeError(f"mask_mod host result must be bool-like, got {type(out).__name__}")


def _decorate(fn: Callable[..., Any], *, kind: str, expected: Tuple[str, ...], cls: type):
    if not callable(fn):
        raise TypeError(f"@{kind} expects a function, got {type(fn).__name__}")
    _validate_signature(fn, expected, kind=kind)
    _validate_mod_ast(fn, kind=kind)
    closure = _validate_scalar_closure(fn)
    fp = _fingerprint(kind, fn, closure)
    return cls(_fn=fn, _closure=dict(closure), fingerprint=fp, name=fn.__name__)


def trace_score_mod(fn: Optional[Callable[..., Any]] = None):
    """Decorator: build a ``TracedScoreMod`` from a restricted score_mod function."""

    def wrap(f: Callable[..., Any]) -> TracedScoreMod:
        return _decorate(f, kind="trace_score_mod", expected=SCORE_MOD_PARAMS, cls=TracedScoreMod)

    if fn is None:
        return wrap
    return wrap(fn)


def trace_mask_mod(fn: Optional[Callable[..., Any]] = None):
    """Decorator: build a ``TracedMaskMod`` from a restricted mask_mod function."""

    def wrap(f: Callable[..., Any]) -> TracedMaskMod:
        return _decorate(f, kind="trace_mask_mod", expected=MASK_MOD_PARAMS, cls=TracedMaskMod)

    if fn is None:
        return wrap
    return wrap(fn)
