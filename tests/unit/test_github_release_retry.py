# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Tests for the release API rate-limit recovery helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "util"))

import github_release_retry as release_retry  # noqa: E402


def test_identifies_primary_and_secondary_rate_limit_responses():
    assert release_retry._is_rate_limit_error(
        "API rate limit exceeded for installation"
    )
    assert release_retry._is_rate_limit_error(
        "You have exceeded a secondary rate limit"
    )
    assert not release_retry._is_rate_limit_error("HTTP 422: validation failed")


def test_retry_waits_then_returns(monkeypatch):
    calls = []
    sleeps = []

    def operation():
        calls.append("call")
        if len(calls) == 1:
            raise release_retry.RateLimited("API rate limit exceeded")
        return "done"

    monkeypatch.setattr(release_retry, "_rate_reset_delay", lambda: None)
    monkeypatch.setattr(release_retry.time, "sleep", sleeps.append)
    deadline = release_retry.time.monotonic() + 60
    assert release_retry._retry("release creation", operation, deadline) == "done"
    assert sleeps == [15]


def test_publish_recovers_a_partial_draft_and_does_not_reupload_assets(
    tmp_path, monkeypatch
):
    notes = tmp_path / "notes.md"
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    for path in (notes, first, second):
        path.write_text(path.name, encoding="utf-8")

    events = []
    assets = {first.name}

    monkeypatch.setattr(release_retry, "_release_exists", lambda *_: True)
    monkeypatch.setattr(
        release_retry,
        "_release_json",
        lambda *_: {
            "assets": [{"name": name, "state": "uploaded"} for name in sorted(assets)]
        },
    )

    def upload(_repo, _tag, asset):
        events.append(Path(asset).name)
        assets.add(Path(asset).name)
        return "uploaded"

    monkeypatch.setattr(release_retry, "_upload_asset", upload)
    monkeypatch.setattr(
        release_retry,
        "_publish",
        lambda *_: events.append("publish") or "published",
    )

    release_retry.publish_release(
        repo="amd/gaia",
        tag="v0.25.4",
        title="v0.25.4",
        notes_file=str(notes),
        assets=[str(first), str(second)],
        prerelease=False,
    )

    assert events == [second.name, "publish"]


def test_publish_retries_an_asset_after_a_rate_limit(tmp_path, monkeypatch):
    notes = tmp_path / "notes.md"
    asset = tmp_path / "package.whl"
    notes.write_text("notes", encoding="utf-8")
    asset.write_text("package", encoding="utf-8")

    uploads = []
    sleeps = []
    assets = set()
    monkeypatch.setattr(release_retry, "_release_exists", lambda *_: True)
    monkeypatch.setattr(
        release_retry,
        "_release_json",
        lambda *_: {
            "assets": [{"name": name, "state": "uploaded"} for name in sorted(assets)]
        },
    )
    monkeypatch.setattr(release_retry, "_rate_reset_delay", lambda: None)
    monkeypatch.setattr(release_retry.time, "sleep", sleeps.append)

    def upload(_repo, _tag, path):
        uploads.append(path)
        if len(uploads) == 1:
            raise release_retry.RateLimited("API rate limit exceeded")
        assets.add(Path(path).name)
        return "uploaded"

    monkeypatch.setattr(release_retry, "_upload_asset", upload)
    monkeypatch.setattr(release_retry, "_publish", lambda *_: "published")

    release_retry.publish_release(
        repo="amd/gaia",
        tag="v0.25.4",
        title="v0.25.4",
        notes_file=str(notes),
        assets=[str(asset)],
        prerelease=False,
    )

    assert uploads == [str(asset), str(asset)]
    assert sleeps == [15]


def test_publish_fails_loudly_when_the_wait_budget_is_exhausted(tmp_path, monkeypatch):
    notes = tmp_path / "notes.md"
    notes.write_text("notes", encoding="utf-8")
    monkeypatch.setattr(release_retry, "_release_exists", lambda *_: True)
    monkeypatch.setattr(
        release_retry,
        "_release_json",
        lambda *_: (_ for _ in ()).throw(release_retry.RateLimited("limited")),
    )
    monkeypatch.setattr(release_retry, "_rate_reset_delay", lambda: 1)

    with pytest.raises(release_retry.ReleaseError, match="bounded retry deadline"):
        release_retry.publish_release(
            repo="amd/gaia",
            tag="v0.25.4",
            title="v0.25.4",
            notes_file=str(notes),
            assets=[],
            prerelease=False,
            max_wait_seconds=0,
        )


def test_release_mode_flags_are_preserved(monkeypatch):
    calls = []
    monkeypatch.setattr(
        release_retry,
        "_run",
        lambda args: calls.append(list(args)) or "ok",
    )

    release_retry._create_draft(
        "amd/gaia", "v0.25.4-rc1", "v0.25.4-rc1", "notes.md", True
    )
    release_retry._publish("amd/gaia", "v0.25.4-rc1", "notes.md", True)
    release_retry._create_draft("amd/gaia", "v0.25.4", "v0.25.4", "notes.md", False)
    release_retry._publish("amd/gaia", "v0.25.4", "notes.md", False)

    assert ["--prerelease", "--latest=false"] == calls[0][-2:]
    assert ["--prerelease", "--latest=false"] == calls[1][-2:]
    assert calls[2][-1] == "--latest"
    assert calls[3][-1] == "--latest"


@pytest.mark.parametrize("prerelease", [False, True])
def test_publish_creates_missing_release_before_upload_and_publication(
    tmp_path, monkeypatch, prerelease
):
    """Exercise publish_release's real lookup -> create -> upload path."""
    notes = tmp_path / "notes.md"
    asset = tmp_path / "package.whl"
    notes.write_text("release notes", encoding="utf-8")
    asset.write_text("wheel", encoding="utf-8")
    events = []
    state = {"exists": False, "assets": set()}

    def lookup(*_args):
        events.append("lookup")
        return state["exists"]

    def create(repo, tag, title, notes_path, is_prerelease):
        events.append("create")
        assert (repo, tag, title, notes_path, is_prerelease) == (
            "amd/gaia",
            "v1-test",
            "v1-test",
            str(notes),
            prerelease,
        )
        assert not state["exists"]
        state["exists"] = True
        return "created"

    def release_json(*_args):
        events.append("release_json")
        assert state["exists"]
        return {
            "assets": [{"name": name, "state": "uploaded"} for name in state["assets"]]
        }

    def upload(_repo, _tag, path):
        events.append("upload")
        state["assets"].add(Path(path).name)
        return "uploaded"

    def publish(_repo, _tag, _notes, is_prerelease):
        events.append("publish")
        assert is_prerelease == prerelease
        assert state["assets"] == {asset.name}
        return "published"

    monkeypatch.setattr(release_retry, "_release_exists", lookup)
    monkeypatch.setattr(release_retry, "_create_draft", create)
    monkeypatch.setattr(release_retry, "_release_json", release_json)
    monkeypatch.setattr(release_retry, "_upload_asset", upload)
    monkeypatch.setattr(release_retry, "_publish", publish)

    release_retry.publish_release(
        repo="amd/gaia",
        tag="v1-test",
        title="v1-test",
        notes_file=str(notes),
        assets=[str(asset)],
        prerelease=prerelease,
    )

    assert events == [
        "lookup",
        "lookup",
        "create",
        "release_json",
        "upload",
        "release_json",
        "release_json",
        "publish",
    ]


def test_publish_does_not_recreate_draft_accepted_before_rate_limit(
    tmp_path, monkeypatch
):
    """GitHub can accept a draft even when gh reports a rate limit."""
    notes = tmp_path / "notes.md"
    notes.write_text("notes", encoding="utf-8")
    events = []
    state = {"exists": False}

    def lookup(*_args):
        events.append("lookup")
        return state["exists"]

    def create(*_args):
        events.append("create")
        state["exists"] = True
        raise release_retry.RateLimited("GitHub accepted the draft before limiting")

    monkeypatch.setattr(release_retry, "_release_exists", lookup)
    monkeypatch.setattr(release_retry, "_create_draft", create)
    monkeypatch.setattr(release_retry, "_release_json", lambda *_: {"assets": []})
    monkeypatch.setattr(release_retry, "_publish", lambda *_: events.append("publish"))
    monkeypatch.setattr(release_retry, "_retry_delay", lambda *_: 1)
    monkeypatch.setattr(
        release_retry.time, "sleep", lambda sec: events.append(("sleep", sec))
    )

    release_retry.publish_release(
        repo="amd/gaia",
        tag="v1-test",
        title="v1-test",
        notes_file=str(notes),
        assets=[],
        prerelease=False,
    )

    assert events == ["lookup", "lookup", "create", ("sleep", 1), "lookup", "publish"]


def test_publish_retries_create_only_after_confirming_draft_still_missing(
    tmp_path, monkeypatch
):
    notes = tmp_path / "notes.md"
    notes.write_text("notes", encoding="utf-8")
    events = []
    state = {"exists": False, "attempts": 0}

    def create(*_args):
        state["attempts"] += 1
        events.append("create")
        if state["attempts"] == 1:
            raise release_retry.RateLimited("secondary rate limit")
        state["exists"] = True
        return "created"

    monkeypatch.setattr(
        release_retry,
        "_release_exists",
        lambda *_: events.append("lookup") or state["exists"],
    )
    monkeypatch.setattr(release_retry, "_create_draft", create)
    monkeypatch.setattr(release_retry, "_release_json", lambda *_: {"assets": []})
    monkeypatch.setattr(release_retry, "_publish", lambda *_: events.append("publish"))
    monkeypatch.setattr(release_retry, "_retry_delay", lambda *_: 1)
    monkeypatch.setattr(
        release_retry.time, "sleep", lambda sec: events.append(("sleep", sec))
    )

    release_retry.publish_release(
        repo="amd/gaia",
        tag="v1-test",
        title="v1-test",
        notes_file=str(notes),
        assets=[],
        prerelease=False,
    )

    assert events == [
        "lookup",
        "lookup",
        "create",
        ("sleep", 1),
        "lookup",
        "create",
        "publish",
    ]


def test_publish_recovers_when_another_process_created_the_draft(tmp_path, monkeypatch):
    notes = tmp_path / "notes.md"
    notes.write_text("notes", encoding="utf-8")
    events = []
    monkeypatch.setattr(release_retry, "_release_exists", lambda *_: False)
    monkeypatch.setattr(
        release_retry,
        "_create_draft",
        lambda *_: events.append("create")
        or (_ for _ in ()).throw(release_retry.ReleaseError("release already exists")),
    )
    monkeypatch.setattr(release_retry, "_release_json", lambda *_: {"assets": []})
    monkeypatch.setattr(release_retry, "_publish", lambda *_: events.append("publish"))

    release_retry.publish_release(
        repo="amd/gaia",
        tag="v1-test",
        title="v1-test",
        notes_file=str(notes),
        assets=[],
        prerelease=False,
    )
    assert events == ["create", "publish"]


@pytest.mark.parametrize(
    "message",
    [
        "repository not found",
        "not found: missing permissions",
        "HTTP 404: bad endpoint",
    ],
)
def test_release_lookup_does_not_treat_unrelated_errors_as_missing(
    monkeypatch, message
):
    def fail(*_args):
        raise release_retry.ReleaseError(message)

    monkeypatch.setattr(release_retry, "_run", fail)
    with pytest.raises(release_retry.ReleaseError, match="not found|404"):
        release_retry._release_exists("amd/gaia", "v1-test")


def test_release_lookup_handles_explicit_missing_release(monkeypatch):
    def fail(*_args):
        raise release_retry.ReleaseError(
            "gh release view v1-test failed: release not found"
        )

    monkeypatch.setattr(release_retry, "_run", fail)
    assert not release_retry._release_exists("amd/gaia", "v1-test")


def test_missing_release_does_not_create_if_deadline_has_expired(tmp_path, monkeypatch):
    notes = tmp_path / "notes.md"
    notes.write_text("notes", encoding="utf-8")
    monkeypatch.setattr(release_retry, "_release_exists", lambda *_: False)
    monkeypatch.setattr(
        release_retry,
        "_create_draft",
        lambda *_: pytest.fail("must not create a release after its deadline"),
    )
    with pytest.raises(
        release_retry.ReleaseError, match="creation exceeded retry deadline"
    ):
        release_retry.publish_release(
            repo="amd/gaia",
            tag="v1-test",
            title="v1-test",
            notes_file=str(notes),
            assets=[],
            prerelease=False,
            max_wait_seconds=0,
        )
