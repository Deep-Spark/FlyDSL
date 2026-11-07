#!/usr/bin/env python3
"""Run MLIR FileCheck tests with optional rocdl/ixdl filtering."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _resolve_filecheck(build_dir: Path) -> Path:
    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        for line in cmake_cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MLIR_DIR:"):
                mlir_dir = line.split("=", 1)[1].strip()
                candidate = (Path(mlir_dir) / "../../../bin/FileCheck").resolve()
                if candidate.exists() and os.access(candidate, os.X_OK):
                    return candidate
                break
    which = subprocess.run(["which", "FileCheck"], text=True, capture_output=True, check=False)
    if which.returncode == 0:
        candidate = Path(which.stdout.strip())
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("FileCheck not found")


def _collect_cases(
    mlir_root: Path,
    exclude_rocdl_related: bool,
    include_ixdl_related: bool,
    exclude_ixdl_related: bool,
) -> list[tuple[Path, list[str]]]:
    if include_ixdl_related and exclude_ixdl_related:
        raise ValueError("include_ixdl_related and exclude_ixdl_related cannot both be true")
    selected: list[tuple[Path, list[str]]] = []
    for path in sorted(mlir_root.rglob("*.mlir")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        run_lines = [ln[len("// RUN:") :].strip() for ln in lines if ln.startswith("// RUN:")]
        if not run_lines:
            continue
        non_comment = "\n".join(ln for ln in lines if not ln.strip().startswith("//"))
        non_comment_l = non_comment.lower()
        run_l = "\n".join(run_lines).lower()
        rocdl_related = (
            "fly_rocdl" in non_comment_l
            or re.search(r"(^|[^a-z])rocdl([^a-z]|$)", non_comment_l) is not None
            or "convert-fly-to-rocdl" in run_l
            or "convert-gpu-to-rocdl" in run_l
        )
        if exclude_rocdl_related and rocdl_related:
            continue
        is_ixdl_related = ("ixdl" in text.lower()) or ("iluvatar" in text.lower())
        if include_ixdl_related and not is_ixdl_related:
            continue
        if exclude_ixdl_related and is_ixdl_related:
            continue
        selected.append((path, run_lines))
    return selected


def _run_case(repo_root: Path, path: Path, run_lines: list[str], fly_opt: Path, filecheck: Path) -> tuple[bool, str]:
    for idx, run in enumerate(run_lines, 1):
        cmd = (
            run.replace("%fly-opt", str(fly_opt))
            .replace("%FileCheck", str(filecheck))
            .replace("%s", str(path))
            .replace("FileCheck", str(filecheck))
        )
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            output = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
            lines = [ln for ln in output.splitlines() if ln.strip()]
            tail = "\n".join(lines[-8:])
            return False, f"RUN#{idx}\n{tail}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MLIR FileCheck with rocdl/ixdl filtering")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--mlir-root", default="tests/mlir", help="MLIR tests root directory")
    parser.add_argument("--build-dir", default="build-fly", help="Build directory containing fly-opt")
    parser.add_argument(
        "--exclude-rocdl-related",
        action="store_true",
        help="Exclude files containing rocdl/fly_rocdl in file content",
    )
    parser.add_argument(
        "--include-ixdl-related",
        action="store_true",
        help="Run only files containing ixdl/iluvatar in file content",
    )
    parser.add_argument(
        "--exclude-ixdl-related",
        action="store_true",
        help="Exclude files containing ixdl/iluvatar in file content",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    mlir_root = (repo_root / args.mlir_root).resolve()
    build_dir = (repo_root / args.build_dir).resolve()
    fly_opt = build_dir / "bin" / "fly-opt"
    if not fly_opt.exists():
        raise RuntimeError(f"fly-opt not found: {fly_opt}")

    filecheck = _resolve_filecheck(build_dir)
    cases = _collect_cases(
        mlir_root,
        args.exclude_rocdl_related,
        args.include_ixdl_related,
        args.exclude_ixdl_related,
    )

    print(f"fly-opt: {fly_opt}")
    print(f"FileCheck: {filecheck}")
    print(f"selected_cases: {len(cases)}")

    passed = 0
    failed: list[tuple[str, str]] = []
    for path, run_lines in cases:
        ok, info = _run_case(repo_root, path, run_lines, fly_opt, filecheck)
        rel = path.relative_to(repo_root).as_posix()
        if ok:
            passed += 1
            print(f"PASS {rel}")
        else:
            failed.append((rel, info))
            print(f"FAIL {rel}")

    print(f"SUMMARY total={len(cases)} passed={passed} failed={len(failed)}")
    if failed:
        print("--- FAIL DETAILS ---")
        for rel, info in failed:
            print(rel)
            print(info)
            print("-----")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
