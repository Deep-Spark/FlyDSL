#!/usr/bin/env python3
"""Download previous perf-metrics artifact from the last successful workflow run."""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


def api_get(url: str, token: str, accept: str = "application/vnd.github+json") -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def api_get_json(url: str, token: str) -> Any:
    return json.loads(api_get(url, token).decode("utf-8"))


def extract_result_from_zip(raw_zip: bytes) -> Optional[Dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        # Prefer result.json at any path inside the artifact zip.
        names = zf.namelist()
        candidates = [n for n in names if n.endswith("result.json") or n == "result.json"]
        if not candidates:
            # Some uploads may store the file at the zip root with another name.
            candidates = [n for n in names if n.endswith(".json")]
        for name in candidates:
            try:
                payload = json.loads(zf.read(name).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and ("cases" in payload or "kernels" in payload or "metrics" in payload):
                return payload
    return None


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    branch = os.environ.get("PERF_BRANCH", "iluvatar")
    run_id = int(os.environ.get("GITHUB_RUN_ID", "0"))
    workflow_file = os.environ.get("PERF_WORKFLOW_FILE", "perf-daily-iluvatar.yml")
    artifact_name = os.environ.get("PERF_METRICS_ARTIFACT", "perf-metrics")

    out_dir = Path(os.environ.get("PERF_DIR", ".perf"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_file = out_dir / "previous_metrics.json"

    if not token:
        print("No GITHUB_TOKEN; skip previous metrics lookup.")
        return 0
    if not repo:
        print("No GITHUB_REPOSITORY; skip previous metrics lookup.")
        return 0

    runs_url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs"
        f"?branch={branch}&status=completed&per_page=30"
    )
    try:
        runs_payload = api_get_json(runs_url, token)
    except urllib.error.HTTPError as exc:
        print(f"::warning::failed to list workflow runs: {exc}")
        return 0

    runs = runs_payload.get("workflow_runs", [])
    if not isinstance(runs, list):
        print("Unexpected workflow runs payload; skip.")
        return 0

    for run in runs:
        rid = int(run.get("id", 0))
        if rid == run_id:
            continue
        if run.get("conclusion") != "success":
            continue

        artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{rid}/artifacts"
        try:
            artifacts_payload = api_get_json(artifacts_url, token)
        except urllib.error.HTTPError as exc:
            print(f"::warning::failed to list artifacts for run {rid}: {exc}")
            continue

        artifacts = artifacts_payload.get("artifacts", [])
        if not isinstance(artifacts, list):
            continue

        target = None
        for art in artifacts:
            if not isinstance(art, dict):
                continue
            if art.get("name") == artifact_name and not art.get("expired", False):
                target = art
                break
        if target is None:
            print(f"No {artifact_name} artifact in run {rid}; try older run.")
            continue

        download_url = target.get("archive_download_url")
        if not download_url:
            continue
        try:
            # archive_download_url redirects to a short-lived zip URL.
            raw_zip = api_get(download_url, token, accept="application/octet-stream")
        except urllib.error.HTTPError as exc:
            print(f"::warning::failed to download artifact from run {rid}: {exc}")
            continue

        found = extract_result_from_zip(raw_zip)
        if found is None:
            print(f"No result.json in {artifact_name} artifact from run {rid}; try older run.")
            continue

        prev_file.write_text(json.dumps(found, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Loaded previous metrics artifact from run {rid}.")
        return 0

    print("No previous successful run with perf-metrics artifact found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
