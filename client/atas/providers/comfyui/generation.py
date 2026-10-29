"""Generate one shot with a ComfyUI video workflow (built for MiniMax H3 reference-to-video).

A submission is journaled under ``<output_dir>/comfyui/`` before it is queued,
keyed by the filled workflow (uploads are named by content hash). The prompt id
is chosen by the client, so a resumed run waits on, or re-queues, the same
prompt instead of paying for a second generation.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

from atas.providers.comfyui.client import (ComfyUIClient, ComfyUIError, ComfyUIUnavailable, execution_error,
                                          failure, output_files)
from atas.providers.comfyui.settings import ComfyUISettings, load_node_map, load_workflow
from atas.providers.comfyui.workflow import choose_duration, fill_workflow, h3_prompt
from atas.providers.wan.types import ReferenceRole
from atas.runner.recon_ledger import write_json_atomic

_STILL_ROLES = {ReferenceRole.FIRST_FRAME, ReferenceRole.CONTINUITY_MEMORY}


async def generate_shot(request, *, geometry_control_mode: str = "reference") -> Path:
    return await asyncio.to_thread(_generate, request, ComfyUISettings())


def _generate(request, settings: ComfyUISettings, client: ComfyUIClient | None = None) -> Path:
    node_map = load_node_map(settings)
    client = client or ComfyUIClient(settings.url, timeout=settings.request_timeout_seconds)
    references = request.condition.references
    videos = sorted((item for item in references if item.media_type == "video"), key=lambda item: item.ordinal)
    images = sorted((item for item in references if item.media_type == "image"), key=lambda item: item.ordinal)
    for item in videos:
        _check_video_length(item.path, settings)
    upload_name = lambda path: _upload_name(path, settings.upload_subfolder)
    workflow = fill_workflow(
        load_workflow(settings), node_map,
        prompt=h3_prompt(request.prompt, has_geometry=any(item.role is ReferenceRole.TARGET_GEOMETRY for item in videos),
                         has_still=any(item.role in _STILL_ROLES for item in images),
                         # An H3-format authored prompt already defines the geometry video's role.
                         grey_hint=settings.grey_region_hint and "subject_definitions" not in request.prompt,
                         image_tag=settings.image_tag,
                         video_tag=settings.video_tag),
        seed=request.shot.seed,
        duration=choose_duration(request.shot.duration, minimum=settings.min_duration_seconds,
                                 maximum=settings.max_duration_seconds, allowed=settings.allowed_durations),
        width=settings.width, height=settings.height, turbo=settings.turbo,
        videos=[upload_name(item.path) for item in videos],
        images=[upload_name(item.path) for item in images],
    )
    signature = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
    root = Path(request.output_dir) / "comfyui"
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / f"{signature[:16]}.json"
    state = json.loads(state_file.read_text()) if state_file.is_file() else {}
    write_json_atomic(root / "workflow.json", workflow)

    if state.get("status") == "done" and Path(state.get("result", "")).is_file():
        return Path(state["result"])
    from atas.providers.gpu_share import release_for
    release_for("comfyui")
    prompt_id = state.get("prompt_id")
    if prompt_id is None or (client.history(prompt_id) is None and not client.in_queue(prompt_id)):
        # Not known to the server (never queued, or lost in a restart): queue it under
        # the journaled id, written first so a crash mid-POST resumes the same prompt.
        prompt_id = prompt_id or str(uuid.uuid4())
        write_json_atomic(state_file, {"status": "queued", "signature": signature, "prompt_id": prompt_id})
        for item in videos + images:
            uploaded = client.upload(item.path, subfolder=settings.upload_subfolder)
            if uploaded != upload_name(item.path):
                raise ComfyUIError(f"ComfyUI stored {item.path.name} as {uploaded!r}, expected {upload_name(item.path)!r}")
        extra = ({"api_key_comfy_org": settings.comfy_org_api_key.get_secret_value()}
                 if settings.comfy_org_api_key else None)
        client.queue(workflow, prompt_id=prompt_id, extra_data=extra)

    try:
        return _collect(client, request, settings, node_map, prompt_id, root, state_file, signature)
    finally:
        # Release the models whether or not the clip came out: a failed job left
        # loaded starves the next stage of RAM and VRAM.
        if settings.free_after_download:
            try:
                client.free()
            except ComfyUIError:
                pass  # the geometry service waits for free VRAM; a missed release only delays it


def _collect(client, request, settings, node_map, prompt_id, root, state_file, signature) -> Path:
    entry = _wait(client, prompt_id, settings)
    error = execution_error(entry)
    if error:
        # The failed id stays in ComfyUI's history; a retry must queue a new one.
        state_file.rename(root / f"{signature[:16]}.{prompt_id}.failed.json")
        raise failure(prompt_id, error)
    files = output_files(entry, node_map.output.node)
    if not files:
        raise ComfyUIError(f"ComfyUI prompt {prompt_id} finished without output on node {node_map.output.node}")
    result = root / prompt_id / "result.mp4"
    downloaded = client.download(files[0], result.with_name("download.mp4"))
    _check_complete(downloaded)
    if settings.trim_to_shot:
        _trim(downloaded, result, float(request.shot.duration))
    else:
        downloaded.replace(result)
    write_json_atomic(state_file, {"status": "done", "signature": signature, "prompt_id": prompt_id,
                                   "result": str(result), "output": files[0]})
    return result


def _upload_name(path: Path, subfolder: str) -> str:
    path = Path(path)
    name = f"{hashlib.sha256(path.read_bytes()).hexdigest()[:20]}{path.suffix.lower()}"
    return f"{subfolder}/{name}" if subfolder else name


def _wait(client: ComfyUIClient, prompt_id: str, settings: ComfyUISettings) -> dict:
    deadline = time.monotonic() + settings.task_timeout_seconds
    while True:
        # ComfyUI adds a history entry only once the prompt has finished, ok or not.
        entry = client.history(prompt_id)
        if entry is not None:
            return entry
        if time.monotonic() > deadline:
            raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within {settings.task_timeout_seconds:.0f}s")
        time.sleep(settings.poll_interval_seconds)


def _video_seconds(path: Path) -> float:
    import cv2
    video = cv2.VideoCapture(str(path))
    try:
        frames, fps = video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()
    return frames / fps if fps > 0 else 0.0


def _check_video_length(path: Path, settings: ComfyUISettings) -> None:
    seconds = _video_seconds(path)
    if not settings.min_reference_video_seconds <= seconds <= settings.max_reference_video_seconds:
        raise ValueError(
            f"reference video {Path(path).name} is {seconds:.2f}s; the workflow accepts "
            f"{settings.min_reference_video_seconds:g}-{settings.max_reference_video_seconds:g}s")


def _trim(source: Path, destination: Path, seconds: float) -> None:
    """Cut to ``seconds`` (re-encoded, audio kept); a clip no longer than that is moved as is."""
    if _video_seconds(source) <= seconds + 0.5 / 24:
        source.replace(destination)
        return
    import subprocess
    from atas.film.assemble import _ffmpeg
    partial = destination.with_name(destination.stem + ".partial.mp4")
    proc = subprocess.run([_ffmpeg(), "-v", "error", "-y", "-i", str(source), "-t", f"{seconds:.3f}",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-c:a", "aac", str(partial)],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode or not partial.is_file():
        raise ComfyUIError("trimming the ComfyUI clip failed: " + proc.stderr[-1500:])
    partial.replace(destination)
    source.unlink()


def _check_complete(path: Path) -> None:
    import cv2
    video = cv2.VideoCapture(str(path))
    try:
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = video.read()
        if count > 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        last, _ = video.read()
    finally:
        video.release()
    if not ok or not last or count < 1:
        path.unlink(missing_ok=True)
        raise ComfyUIUnavailable(f"downloaded ComfyUI video {path} is incomplete; retry download, not generation")
