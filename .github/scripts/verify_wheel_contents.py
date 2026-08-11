#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import zipfile
from pathlib import Path

# Wheel root should only contain package payload and dist-info metadata.
_ALLOWED_TOP_LEVEL_PATTERNS = ("flydsl", "flydsl-*.dist-info")
# Release artifacts should not ship these trees.
_FORBIDDEN_PATH_PATTERNS = (
    "*/tests/*",
    "*/test/*",
    "*/examples/*",
    "*/docs/*",
    "*/.github/*",
)
# Packaged wheels must include the IR/ISA dump guard used to deny FLYDSL_DUMP_IR
# for external installs.
_REQUIRED_MEMBERS = (
    "flydsl/utils/release_guard.py",
    "flydsl/utils/dump_support.py",
)


def _resolve_wheels(patterns: list[str]) -> list[Path]:
    wheels: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            wheels.extend(p for p in matches if p.is_file())
            continue
        p = Path(pattern)
        if p.is_file():
            wheels.append(p)
    # De-dup while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in wheels:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _is_allowed_top_level(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in _ALLOWED_TOP_LEVEL_PATTERNS)


def _is_forbidden_member(member: str) -> bool:
    normalized = member.strip("/")
    return any(fnmatch.fnmatch(normalized, pat) for pat in _FORBIDDEN_PATH_PATTERNS)


def _verify_one_wheel(wheel_path: Path) -> tuple[list[str], dict]:
    violations: list[str] = []
    summary = {"wheel": wheel_path.name, "top_level": {}, "file_count": 0}

    with zipfile.ZipFile(wheel_path) as zf:
        members = [m for m in zf.namelist() if m and not m.endswith("/")]
        member_set = set(members)
        summary["file_count"] = len(members)

        for required in _REQUIRED_MEMBERS:
            if required not in member_set:
                violations.append(f"{wheel_path.name}: missing required release member `{required}`")

        for member in members:
            top = member.split("/", 1)[0]
            summary["top_level"][top] = summary["top_level"].get(top, 0) + 1
            if not _is_allowed_top_level(top):
                violations.append(f"{wheel_path.name}: unexpected top-level entry `{top}` from `{member}`")

            if _is_forbidden_member(member):
                violations.append(f"{wheel_path.name}: forbidden release path `{member}`")

            if member == "flydsl/utils/release_guard.py":
                text = zf.read(member).decode("utf-8", errors="replace")
                if "def ir_dump_allowed" not in text or "def is_packaged_install" not in text:
                    violations.append(
                        f"{wheel_path.name}: `{member}` is present but missing IR dump guard symbols"
                    )
                if "_env_truthy(\"FLYDSL_ALLOW_IR_DUMP\")" in text or "_env_truthy('FLYDSL_ALLOW_IR_DUMP')" in text:
                    violations.append(
                        f"{wheel_path.name}: `{member}` still treats FLYDSL_ALLOW_IR_DUMP as an override"
                    )

            if member == "flydsl/utils/dump_support.py":
                text = zf.read(member).decode("utf-8", errors="replace")
                if "DUMP_SUPPORT = False" not in text or "def require_dump_support" not in text:
                    violations.append(
                        f"{wheel_path.name}: `{member}` must bake DUMP_SUPPORT = False for release wheels"
                    )

    return violations, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate wheel payload stays minimal: only flydsl + dist-info top-level entries, "
            "no tests/examples/docs payload, and the IR/ISA dump release guard is present."
        )
    )
    parser.add_argument("wheels", nargs="+", help="Wheel files or glob patterns (e.g. dist/*.whl)")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional output path for payload summary JSON.",
    )
    args = parser.parse_args()

    wheel_paths = _resolve_wheels(args.wheels)
    if not wheel_paths:
        print("ERROR: no wheel files found for inputs:", ", ".join(args.wheels), file=sys.stderr)
        return 2

    all_violations: list[str] = []
    summaries: list[dict] = []
    for wheel in wheel_paths:
        violations, summary = _verify_one_wheel(wheel)
        all_violations.extend(violations)
        summaries.append(summary)

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summaries, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if all_violations:
        print("Wheel content gate failed:")
        for v in all_violations:
            print(f"- {v}")
        return 1

    print(f"Content gate passed for {len(wheel_paths)} wheel(s): payload matches release allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
