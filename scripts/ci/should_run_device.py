#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Decide whether ci-device should execute full L2 device tests."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _load_json_file(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_nested(obj: dict[str, Any], *keys: str) -> Any:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _api_get(url: str, token: str) -> dict[str, Any] | list[Any]:
    req = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_pull_request_labels(owner_repo: str, pr_number: int, token: str) -> list[str]:
    url = f"https://api.github.com/repos/{owner_repo}/issues/{pr_number}"
    issue = _api_get(url, token)
    labels = issue.get("labels", []) if isinstance(issue, dict) else []
    return [label.get("name", "") for label in labels if isinstance(label, dict)]


def _fetch_pull_request(owner_repo: str, pr_number: int, token: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}"
    payload = _api_get(url, token)
    return payload if isinstance(payload, dict) else {}


def _fetch_pull_request_files(owner_repo: str, pr_number: int, token: str) -> list[str]:
    page = 1
    files: list[str] = []
    while True:
        url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}/files?" + urllib.parse.urlencode(
            {"per_page": 100, "page": page}
        )
        payload = _api_get(url, token)
        if not isinstance(payload, list) or not payload:
            break
        for item in payload:
            if isinstance(item, dict) and "filename" in item:
                files.append(str(item["filename"]))
        if len(payload) < 100:
            break
        page += 1
    return files


def _fetch_compare_files(owner_repo: str, base_sha: str, head_sha: str, token: str) -> tuple[list[str], bool]:
    url = (
        f"https://api.github.com/repos/{owner_repo}/compare/"
        + urllib.parse.quote(base_sha)
        + "..."
        + urllib.parse.quote(head_sha)
    )
    payload = _api_get(url, token)
    if not isinstance(payload, dict):
        return [], False
    files: list[str] = []
    for item in payload.get("files", []) or []:
        if isinstance(item, dict) and "filename" in item:
            files.append(str(item["filename"]))
    return files, bool(payload.get("truncated"))


def _matches_path_filters(changed_files: list[str], path_filters: list[str]) -> tuple[bool, list[str]]:
    matches: list[str] = []
    for changed_file in changed_files:
        if any(fnmatch.fnmatch(changed_file, pattern) for pattern in path_filters):
            matches.append(changed_file)
    return (len(matches) > 0), matches


def _resolve_pytest_targets(include_globs: list[str], exclude_globs: list[str]) -> list[str]:
    repo_root = pathlib.Path(".")
    collected: set[str] = set()
    for pattern in include_globs:
        for path in repo_root.glob(pattern):
            if path.is_file():
                rel = path.as_posix()
                collected.add(rel)
    if not collected:
        return []
    targets = sorted(collected)
    if not exclude_globs:
        return targets
    filtered: list[str] = []
    for rel in targets:
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude_globs):
            continue
        filtered.append(rel)
    return filtered


def _write_output(key: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as out:
        out.write(f"{key}={value}\n")


def _is_cross_repo_pr(pull_request: dict[str, Any], default_repo: str) -> bool:
    head_repo = pull_request.get("head", {}).get("repo", {})
    base_repo = pull_request.get("base", {}).get("repo", {})
    head_full_name = str(head_repo.get("full_name", "")).strip().lower()
    base_full_name = str(base_repo.get("full_name", "")).strip().lower()
    repo_full_name = default_repo.strip().lower()
    if not base_full_name:
        base_full_name = repo_full_name
    if not head_full_name:
        return False
    return head_full_name != base_full_name


def _decide(
    event_name: str,
    event_payload: dict[str, Any],
    config: dict[str, Any],
    repository: str,
    token: str,
) -> tuple[bool, str, list[str], list[str]]:
    device_cfg = _get_nested(config, "device") or {}
    force_label = str(device_cfg.get("force_label", "run-device-ci"))
    path_filters = list(device_cfg.get("path_filters", []))
    skip_msg = str(
        device_cfg.get(
            "required_skip_message",
            "Skipped by policy: no matched paths and no run-device-ci label.",
        )
    )

    if event_name == "workflow_dispatch":
        inputs = event_payload.get("inputs", {})
        force_run = _as_bool(inputs.get("force_run", False))
        if force_run:
            return True, "Running by workflow_dispatch force_run=true.", [], []
        return False, "Skipped by policy: workflow_dispatch force_run=false.", [], []

    pr_number: int | None = None
    labels: list[str] = []
    changed_files: list[str] = []

    if event_name == "pull_request":
        pull_request = event_payload.get("pull_request", {})
        pr_number = pull_request.get("number")
        labels = [item.get("name", "") for item in pull_request.get("labels", []) if isinstance(item, dict)]
        if _is_cross_repo_pr(pull_request, repository):
            return False, "Skipped by policy: fork pull_request cannot auto-run ci-device.", [], []
        changed_files = _fetch_pull_request_files(repository, int(pr_number), token)
    elif event_name == "workflow_run":
        workflow_run = event_payload.get("workflow_run", {})
        pull_requests = workflow_run.get("pull_requests", [])
        if pull_requests:
            pr_number = pull_requests[0].get("number")
        if pr_number is None:
            return False, "Skipped by policy: workflow_run has no pull_request context.", [], []
        pull_request = _fetch_pull_request(repository, int(pr_number), token)
        if _is_cross_repo_pr(pull_request, repository):
            return False, "Skipped by policy: fork pull_request cannot auto-run ci-device.", [], []
        labels = _fetch_pull_request_labels(repository, int(pr_number), token)
        changed_files = _fetch_pull_request_files(repository, int(pr_number), token)
    elif event_name == "push":
        before_sha = str(event_payload.get("before") or "").strip()
        after_sha = str(event_payload.get("after") or "").strip()
        zero_sha = "0" * 40
        if not after_sha:
            return False, "Skipped by policy: push event has no after SHA.", [], []
        if not before_sha or before_sha == zero_sha:
            return True, "Running because this is a new ref with no compare base.", [], []
        changed_files, truncated = _fetch_compare_files(repository, before_sha, after_sha, token)
        if truncated:
            return (
                True,
                "Running because GitHub compare result is truncated; treating as matched.",
                changed_files,
                changed_files,
            )
        labels = []
    else:
        return False, f"Skipped by policy: unsupported event `{event_name}`.", [], []

    if force_label in labels:
        return True, f"Running by label `{force_label}`.", changed_files, []

    matched, matched_files = _matches_path_filters(changed_files, path_filters)
    if matched:
        return True, "Running by path filter match.", changed_files, matched_files

    return False, skip_msg, changed_files, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=".github/ci-config.yaml", help="Path to CI config file")
    args = parser.parse_args()

    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")

    if not event_name or not event_path:
        print("Missing GITHUB_EVENT_NAME or GITHUB_EVENT_PATH", file=sys.stderr)
        return 2
    needs_api = event_name in {"pull_request", "workflow_run", "push"}
    if needs_api and (not repository or not token):
        print("Missing GITHUB_REPOSITORY or GITHUB_TOKEN", file=sys.stderr)
        return 2

    try:
        config = _load_json_file(pathlib.Path(args.config))
        event_payload = _load_json_file(pathlib.Path(event_path))
        device_cfg = _get_nested(config, "device") or {}
        timeouts_cfg = _get_nested(config, "timeouts") or {}
        runner_labels = list(device_cfg.get("runner_labels", ["self-hosted", "linux", "x64", "gpu-iluvatar"]))
        pytest_args = list(device_cfg.get("pytest_args", ["tests/unit", "-m", "l2_device"]))
        pytest_targets: list[str] = []
        pytest_target_globs = list(device_cfg.get("pytest_target_globs", []))
        pytest_exclude_globs = list(device_cfg.get("pytest_exclude_globs", []))
        pytest_common_args = list(device_cfg.get("pytest_common_args", []))
        if pytest_target_globs:
            pytest_targets = _resolve_pytest_targets(pytest_target_globs, pytest_exclude_globs)
            if not pytest_targets:
                raise ValueError(
                    "device.pytest_target_globs resolved to an empty file list; "
                    "check .github/ci-config.yaml patterns"
                )
            pytest_args = pytest_targets + pytest_common_args
        must_pass_tests = list(
            device_cfg.get(
                "must_pass_tests",
                [
                    "tests/unit/test_iluvatar_binary_pipeline_smoke.py",
                    "tests/unit/test_iluvatar_compile_backend.py",
                    "tests/unit/test_iluvatar_jit_launch_smoke.py",
                    "tests/unit/test_iluvatar_jit_runtime_resolution.py",
                ],
            )
        )
        runtime_smoke_blob_path = str(device_cfg.get("runtime_smoke_blob_path", ""))
        runtime_smoke_kernel = str(device_cfg.get("runtime_smoke_kernel", ""))
        runtime_smoke_launch_kernel = str(device_cfg.get("runtime_smoke_launch_kernel", ""))
        container_image_override = os.environ.get("CI_DEVICE_IMAGE_REF", "").strip()
        container_image = container_image_override or str(
            device_cfg.get("container_image", "ghcr.io/deep-spark/flydsl-iluvatar-ci:stable")
        )
        skip_msg = str(
            device_cfg.get(
                "required_skip_message",
                "Skipped by policy: no matched paths and no run-device-ci label.",
            )
        )
        python_version = str(config.get("python_version", "3.10"))
        device_timeout_minutes = int(timeouts_cfg.get("device_minutes", 90))
        should_run, reason, changed_files, matched_files = _decide(
            event_name=event_name,
            event_payload=event_payload,
            config=config,
            repository=repository,
            token=token,
        )
    except (FileNotFoundError, json.JSONDecodeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"device decision failed: {exc}", file=sys.stderr)
        return 1

    changed_display = ", ".join(changed_files[:20]) if changed_files else "(none)"
    matched_display = ", ".join(matched_files[:20]) if matched_files else "(none)"
    pytest_targets_display = ", ".join(pytest_targets[:20]) if pytest_targets else "(none)"
    print(f"should_run={str(should_run).lower()}")
    print(f"reason={reason}")
    print(f"changed_files={changed_display}")
    print(f"matched_files={matched_display}")
    print(f"pytest_targets_count={len(pytest_targets)}")
    print(f"pytest_targets_preview={pytest_targets_display}")

    _write_output("should_run", str(should_run).lower())
    _write_output("reason", reason)
    _write_output("changed_files", changed_display)
    _write_output("matched_files", matched_display)
    _write_output("pytest_targets_count", str(len(pytest_targets)))
    _write_output("pytest_targets_preview", pytest_targets_display)
    _write_output("runner_labels_json", json.dumps(runner_labels))
    _write_output("skip_message", skip_msg)
    _write_output("device_pytest_args_json", json.dumps(pytest_args))
    _write_output("device_must_pass_tests_json", json.dumps(must_pass_tests))
    _write_output("runtime_smoke_blob_path", runtime_smoke_blob_path)
    _write_output("runtime_smoke_kernel", runtime_smoke_kernel)
    _write_output("runtime_smoke_launch_kernel", runtime_smoke_launch_kernel)
    _write_output("container_image", container_image)
    _write_output("python_version", python_version)
    _write_output("device_timeout_minutes", str(device_timeout_minutes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
