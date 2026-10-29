"""Fill an API-format ComfyUI workflow for one shot. Pure: no network."""
from __future__ import annotations

import copy
import re

from atas.providers.comfyui.settings import NodeInput, NodeMap

# Packing uses Wan's handles (@视频1, @图片1); H3 tags references as <Video 1>, <Picture 1>.
_VIDEO = re.compile(r"@?视频(\d+)")
_IMAGE = re.compile(r"@?图片(\d+)")


def h3_prompt(prompt: str, *, has_geometry: bool, has_still: bool, grey_hint: bool = True,
              image_tag: str = "<Picture {n}>", video_tag: str = "<Video {n}>") -> str:
    prompt = _VIDEO.sub(lambda match: video_tag.format(n=match[1]), prompt)
    prompt = _IMAGE.sub(lambda match: image_tag.format(n=match[1]), prompt)
    if has_geometry and grey_hint:
        video, still = video_tag.format(n=1), image_tag.format(n=1)
        continuing = f" continuing the opening still {still}" if has_still else ""
        prompt += (
            f"\n{video} is a camera and layout guide rendered from a point cloud: follow its camera "
            f"motion and where surfaces are, not its look. Flat grey areas in {video} are unknown "
            f"space, not grey objects; fill them with the surrounding surfaces and scenery{continuing}."
        )
    return prompt


def choose_duration(seconds: float, *, minimum: float, maximum: float, allowed: list[float]) -> float:
    if allowed:
        return min(allowed, key=lambda value: (abs(value - seconds), value))
    return float(max(minimum, min(maximum, seconds)))


def fill_workflow(
    template: dict,
    node_map: NodeMap,
    *,
    prompt: str,
    seed: int | None = None,
    duration: float | None = None,
    width: int | None = None,
    height: int | None = None,
    turbo: bool | None = None,
    videos: list[str] = (),
    images: list[str] = (),
) -> dict:
    """Return a copy of ``template`` with the shot's values set and unused loaders removed."""
    if len(videos) > len(node_map.videos) or len(images) > len(node_map.images):
        raise ValueError(
            f"{len(videos)} videos / {len(images)} images exceed the workflow's "
            f"{len(node_map.videos)} video and {len(node_map.images)} image inputs")
    workflow = copy.deepcopy(template)

    def put(target: NodeInput | None, value) -> None:
        if target is None or value is None:
            return
        if target.node not in workflow:
            raise ValueError(f"node map names node {target.node!r}, which is not in the workflow")
        workflow[target.node].setdefault("inputs", {})[target.input] = value

    put(node_map.prompt, prompt)
    put(node_map.seed, seed)
    put(node_map.duration, duration)
    put(node_map.width, width)
    put(node_map.height, height)
    put(node_map.turbo, turbo)
    for slot, name in zip(node_map.videos, videos):
        put(slot, name)
    for slot, name in zip(node_map.images, images):
        put(slot, name)
    unused = [slot.node for slot in node_map.videos[len(videos):] + node_map.images[len(images):]]
    _remove(workflow, unused, keep={node_map.generator_node, node_map.output.node})
    return workflow


def _remove(workflow: dict, nodes: list[str], *, keep: set[str]) -> None:
    """Delete ``nodes``; nodes fed by them are deleted too, except ``keep``, which lose the link."""
    pending = [node for node in nodes if node in workflow]
    while pending:
        node = pending.pop()
        if node in keep:
            raise ValueError(f"node {node!r} is both an unused loader and required")
        workflow.pop(node, None)
        for other_id, other in list(workflow.items()):
            inputs = other.get("inputs", {})
            linked = [key for key, value in inputs.items() if _links_to(value, node)]
            if not linked:
                continue
            if other_id in keep:
                for key in linked:
                    del inputs[key]
            elif other_id not in pending:
                pending.append(other_id)


def _links_to(value, node: str) -> bool:
    return isinstance(value, list) and len(value) == 2 and str(value[0]) == node and isinstance(value[1], int)
