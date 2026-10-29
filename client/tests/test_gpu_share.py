from __future__ import annotations

import pytest

from atas.providers import gpu_share


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake(method, url, payload=None, timeout=60.0):
        seen.append((method, url, payload))
        if url.endswith("/api/v1/models"):
            return {"models": [
                {"key": "gemma", "loaded_instances": [{"id": "gemma"}, {"id": "gemma:2"}]},
                {"key": "idle", "loaded_instances": []},
            ]}
        return None
    monkeypatch.setattr(gpu_share, "_call", fake)
    monkeypatch.setenv("ATAS_LMSTUDIO_URL", "http://lm:1234")
    monkeypatch.setenv("ATAS_VIDEO_PROVIDER", "comfyui")
    monkeypatch.setenv("COMFYUI_URL", "http://comfy:8188")
    return seen


def test_geometry_releases_lmstudio_and_comfyui(calls) -> None:
    gpu_share.release_for("geometry")
    unloads = [payload["instance_id"] for method, url, payload in calls if url.endswith("/unload")]
    assert unloads == ["gemma", "gemma:2"]
    assert ("POST", "http://comfy:8188/free", {"unload_models": True, "free_memory": True}) in calls


def test_comfyui_stage_only_unloads_lmstudio(calls) -> None:
    gpu_share.release_for("comfyui")
    assert not any(url.endswith("/free") for _, url, _ in calls)
    assert any(url.endswith("/unload") for _, url, _ in calls)


def test_llm_stage_leaves_lmstudio_loaded(calls) -> None:
    gpu_share.release_for("llm")
    assert not any("lm:1234" in url for _, url, _ in calls)


def test_release_failures_do_not_raise(monkeypatch, capsys) -> None:
    def down(*a, **k):
        raise ConnectionError("refused")
    monkeypatch.setattr(gpu_share, "_call", down)
    monkeypatch.setenv("ATAS_LMSTUDIO_URL", "http://lm:1234")
    monkeypatch.setenv("ATAS_VIDEO_PROVIDER", "comfyui")
    gpu_share.release_for("geometry")
    assert "LM Studio unload failed" in capsys.readouterr().out


def test_nothing_configured_makes_no_calls(monkeypatch) -> None:
    monkeypatch.setattr(gpu_share, "_call", lambda *a, **k: pytest.fail("no call expected"))
    monkeypatch.delenv("ATAS_LMSTUDIO_URL", raising=False)
    monkeypatch.setenv("ATAS_VIDEO_PROVIDER", "wan")
    monkeypatch.delenv("ATAS_IMAGE_BACKEND", raising=False)
    gpu_share.release_for("geometry")


def test_local_chat_gets_a_long_timeout_and_logs_retries(monkeypatch, capsys) -> None:
    from atas.providers import llm
    seen = []

    def once(messages, *, timeout):
        seen.append(timeout)
        if len(seen) == 1:
            raise TimeoutError("timed out")
        return "ok"
    monkeypatch.setattr(llm, "_chat_once", once)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setenv("ATAS_CHAT_URL", "http://lm:1234/v1")
    assert llm.chat("hi") == "ok"
    assert seen == [900.0, 900.0]
    assert "attempt 1/" in capsys.readouterr().out
