"""Hand one GPU between LM Studio, ComfyUI and the geometry service.

The pipeline runs one GPU stage at a time, so before a stage starts the others
are asked to release their models. LM Studio loads its model again on the next
chat request (just-in-time loading); geometry jobs run in short-lived worker
processes and need no release. Releases are best effort: a geometry job waits
for free memory anyway, so a failed release only costs time.

``ATAS_LMSTUDIO_URL`` enables LM Studio unloading; ComfyUI is released when it
is the video provider or the image backend.
"""
from __future__ import annotations

import json
import os
import urllib.request

STAGES = ("llm", "comfyui", "geometry")


def _call(method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw.strip() else None


def _instance_ids(model: dict) -> list[str]:
    ids = []
    for instance in model.get("loaded_instances") or []:
        if isinstance(instance, str):
            ids.append(instance)
        elif isinstance(instance, dict):
            value = instance.get("id") or instance.get("instance_id") or instance.get("identifier")
            if value:
                ids.append(str(value))
    return ids


def unload_lmstudio(base: str) -> list[str]:
    """Unload every loaded LM Studio model instance; returns the ids unloaded."""
    base = base.rstrip("/")
    listing = _call("GET", f"{base}/api/v1/models") or {}
    unloaded = []
    for model in listing.get("models", []):
        for instance in _instance_ids(model):
            _call("POST", f"{base}/api/v1/models/unload", {"instance_id": instance})
            unloaded.append(instance)
    return unloaded


def free_comfyui(base: str) -> None:
    _call("POST", f"{base.rstrip('/')}/free", {"unload_models": True, "free_memory": True})


def _comfyui_url() -> str | None:
    used = (os.getenv("ATAS_VIDEO_PROVIDER", "").strip().lower() == "comfyui"
            or os.getenv("ATAS_IMAGE_BACKEND", "").strip().lower() == "comfyui")
    if not used:
        return None
    from atas.providers.comfyui.settings import ComfyUISettings
    return ComfyUISettings().url


def release_for(stage: str) -> None:
    """Release the GPU models of every service except the one ``stage`` is about to use."""
    if stage not in STAGES:
        raise ValueError(f"unknown GPU stage {stage!r}")
    lmstudio = os.getenv("ATAS_LMSTUDIO_URL", "").strip()
    if stage != "llm" and lmstudio:
        try:
            unload_lmstudio(lmstudio)
        except Exception as exc:  # noqa: BLE001 - best effort, see module docstring
            print(f"gpu: LM Studio unload failed before {stage}: {exc}", flush=True)
    comfyui = _comfyui_url() if stage != "comfyui" else None
    if comfyui:
        try:
            free_comfyui(comfyui)
        except Exception as exc:  # noqa: BLE001
            print(f"gpu: ComfyUI free failed before {stage}: {exc}", flush=True)
