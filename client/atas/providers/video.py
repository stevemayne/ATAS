"""Video-generator selection: ``ATAS_VIDEO_PROVIDER`` = wan (default) | comfyui.

Request packing (references, prompt, condition) is provider-neutral and stays in
``atas.providers.wan.generation``; only reference slots and submission differ.
"""
from __future__ import annotations

import os

PROVIDERS = ("wan", "comfyui")


def provider() -> str:
    name = os.getenv("ATAS_VIDEO_PROVIDER", "wan").strip().lower() or "wan"
    if name not in PROVIDERS:
        raise ValueError(f"ATAS_VIDEO_PROVIDER={name!r}; expected one of {PROVIDERS}")
    return name


def character_slots(*, has_geometry: bool, has_still: bool) -> int:
    """Reference slots left for character sheets after the geometry video and still."""
    if provider() == "comfyui":
        # H3 numbers images and videos separately: the still and the character
        # sheets share the image inputs; the geometry render uses a video input.
        from atas.providers.comfyui.settings import load_node_map
        return max(0, len(load_node_map().images) - int(has_still))
    from atas.shot.packing import WAN30_REFERENCE_CAP, reserved_reference_slots
    return max(0, WAN30_REFERENCE_CAP - reserved_reference_slots(has_geometry=has_geometry, has_still=has_still))


def check_packing(shot_id: str, references) -> None:
    videos = sum(1 for item in references if item.media_type == "video")
    images = sum(1 for item in references if item.media_type == "image")
    if provider() == "comfyui":
        from atas.providers.comfyui.settings import load_node_map
        node_map = load_node_map()
        if videos > len(node_map.videos) or images > len(node_map.images):
            raise RuntimeError(
                f"{shot_id} packed {videos} videos / {images} images; the ComfyUI workflow has "
                f"{len(node_map.videos)} video and {len(node_map.images)} image inputs")
        return
    from atas.shot.packing import WAN30_REFERENCE_CAP
    if videos + images > WAN30_REFERENCE_CAP:
        raise RuntimeError(
            f"{shot_id} packed {videos + images} Wan references; cap is {WAN30_REFERENCE_CAP}")


def prompt_handle(reference) -> str:
    """The label the authored video prompt uses for one packed reference."""
    if provider() == "comfyui":
        kind = {"video": "Video", "image": "Picture"}.get(reference.media_type, "Audio")
        return f"<{kind} {reference.ordinal}>"
    return reference.prompt_handle


def handle_pattern() -> str:
    return r"<(?:Video|Picture|Audio) \d+>" if provider() == "comfyui" else r"@(?:图片|视频)\d+"


def prompt_protocol() -> str:
    """Video-prompt rules the prompt author adds for this provider ("" for Wan)."""
    if provider() == "comfyui":
        from atas.providers.comfyui.h3_prompting import protocol
        return protocol()
    return ""


def max_prompt_chars() -> int:
    # H3's full-reference format runs to several hundred words.
    return 9000 if provider() == "comfyui" else 4000


def build_request(shot, **kwargs):
    from atas.providers.wan import generation
    return generation.build_request(shot, **kwargs)


async def generate_shot(request, **kwargs):
    if provider() == "comfyui":
        from atas.providers.comfyui.generation import generate_shot as comfyui_generate
        return await comfyui_generate(request, **kwargs)
    from atas.providers.wan import generation
    return await generation.generate_shot(request, **kwargs)
