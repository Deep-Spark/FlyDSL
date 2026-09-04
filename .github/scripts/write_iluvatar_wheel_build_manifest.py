#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Write Iluvatar wheel build provenance (manifest JSON + runner log)."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def _wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.endswith(".dist-info/METADATA"):
                for line in zf.read(name).decode("utf-8", errors="replace").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
    return ""


def _resolve_wheels(patterns: list[str]) -> list[Path]:
    wheels: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_file() and p.suffix == ".whl":
            wheels.append(p)
            continue
        # Path.glob only accepts relative patterns.
        if p.is_absolute():
            matches = sorted(p.parent.glob(p.name))
        else:
            matches = sorted(Path().glob(pattern))
        wheels.extend(m for m in matches if m.is_file() and m.suffix == ".whl")
    seen: set[Path] = set()
    out: list[Path] = []
    for wheel in wheels:
        rp = wheel.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(wheel)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", help="Wheel paths or globs (e.g. dist/*.whl)")
    parser.add_argument("--manifest-json", required=True, help="Output build-manifest.json path")
    parser.add_argument(
        "--summary-md",
        default="",
        help="Optional markdown path (also printed to stdout)",
    )
    args = parser.parse_args()

    wheels = _resolve_wheels(args.wheels)
    if not wheels:
        print("ERROR: no wheel files found", flush=True)
        return 2

    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "").strip()
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    run_url = f"{server_url}/{repository}/actions/runs/{run_id}" if repository and run_id else ""

    wheel_entries = []
    for wheel in wheels:
        wheel_entries.append(
            {
                "filename": wheel.name,
                "path": str(wheel.as_posix()),
                "size_bytes": wheel.stat().st_size,
                "version": _wheel_version(wheel),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flydsl_commit": os.environ.get("FLYDSL_COMMIT", "").strip(),
        "flydsl_commit_short": os.environ.get("FLYDSL_COMMIT_SHORT", "").strip(),
        "ixcc_commit": os.environ.get("IXCC_COMMIT", "").strip(),
        "ixcc_commit_short": os.environ.get("IXCC_COMMIT_SHORT", "").strip(),
        "ixcc_mlir_cmake": os.environ.get("IXCC_MLIR_CMAKE", "").strip(),
        "ixcc_variant": os.environ.get("IXCC_VARIANT", "").strip(),
        "docker_image": os.environ.get("DOCKER_IMAGE", "").strip(),
        "python_versions": os.environ.get("PYTHON_VERSIONS", "").strip(),
        "github_run_id": run_id,
        "github_run_attempt": run_attempt,
        "github_run_url": run_url,
        "github_sha": os.environ.get("GITHUB_SHA", "").strip(),
        "github_ref": os.environ.get("GITHUB_REF", "").strip(),
        "artifact_name": os.environ.get("WHEEL_ARTIFACT_NAME", "").strip(),
        "wheels": wheel_entries,
    }

    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "## Iluvatar wheel build provenance",
        "",
        f"- FlyDSL commit: `{manifest['flydsl_commit'] or '<unknown>'}`",
        f"- ixcc commit: `{manifest['ixcc_commit'] or '<unknown>'}`",
        f"- GitHub run id: `{manifest['github_run_id'] or '<unknown>'}`",
        f"- Artifact name: `{manifest['artifact_name'] or '<unknown>'}`",
        f"- Docker image: `{manifest['docker_image'] or '<unknown>'}`",
        f"- IXCC variant: `{manifest['ixcc_variant'] or '<unknown>'}`",
        f"- MLIR cmake: `{manifest['ixcc_mlir_cmake'] or '<unknown>'}`",
    ]
    if run_url:
        lines.append(f"- Run URL: {run_url}")
    lines.extend(["", "| Wheel | Version | Size (bytes) |", "|---|---|---|"])
    for entry in wheel_entries:
        lines.append(f"| `{entry['filename']}` | `{entry['version']}` | {entry['size_bytes']} |")
    lines.append("")
    lines.append(f"Manifest: `{manifest_path.as_posix()}`")
    lines.append("")
    summary = "\n".join(lines) + "\n"

    if args.summary_md:
        summary_path = Path(args.summary_md)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")

    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
