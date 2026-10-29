"""Stills and character sheets with Flux.2 Klein image edit on ComfyUI.

The workflow is built per call: each reference image is scaled, encoded and
chained onto the conditioning with ``ReferenceLatent`` (as in ComfyUI's Klein
edit template), so 0..N references need no template surgery. With none it is
plain text-to-image.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Sequence

from atas.providers.comfyui.client import ComfyUIClient, ComfyUIError, execution_error, failure, output_files
from atas.providers.comfyui.settings import ComfyUISettings

_HANDLE = re.compile(r"\bImage\s+(\d+)\b")


def edit_prompt(prompt: str, tag: str) -> str:
    return _HANDLE.sub(lambda match: tag.format(n=match[1]), prompt)


def output_size(aspect: float | None, settings: ComfyUISettings) -> tuple[int, int]:
    if aspect is None:
        return settings.image_square, settings.image_square
    height = settings.image_height
    return max(16, round(height * aspect / 16) * 16), height


def build_workflow(settings: ComfyUISettings, *, prompt: str, seed: int, width: int, height: int,
                   images: Sequence[str]) -> dict:
    def node(cls: str, **inputs) -> dict:
        return {"class_type": cls, "inputs": inputs}

    wf = {
        "unet": node("UNETLoader", unet_name=settings.image_unet, weight_dtype="default"),
        "clip": node("CLIPLoader", clip_name=settings.image_clip, type="flux2", device="default"),
        "vae": node("VAELoader", vae_name=settings.image_vae),
        "text": node("CLIPTextEncode", clip=["clip", 0], text=prompt),
        "zero": node("ConditioningZeroOut", conditioning=["text", 0]),
    }
    positive, negative = ["text", 0], ["zero", 0]
    for index, name in enumerate(images, start=1):
        wf[f"load{index}"] = node("LoadImage", image=name)
        wf[f"scale{index}"] = node("ImageScaleToTotalPixels", image=[f"load{index}", 0], upscale_method="lanczos",
                                   megapixels=settings.image_reference_megapixels, resolution_steps=1)
        wf[f"encode{index}"] = node("VAEEncode", pixels=[f"scale{index}", 0], vae=["vae", 0])
        wf[f"pos{index}"] = node("ReferenceLatent", conditioning=positive, latent=[f"encode{index}", 0])
        wf[f"neg{index}"] = node("ReferenceLatent", conditioning=negative, latent=[f"encode{index}", 0])
        positive, negative = [f"pos{index}", 0], [f"neg{index}", 0]
    wf.update({
        "guider": node("CFGGuider", model=["unet", 0], positive=positive, negative=negative, cfg=1),
        "sampler": node("KSamplerSelect", sampler_name="euler"),
        "sigmas": node("Flux2Scheduler", steps=settings.image_steps, width=width, height=height),
        "noise": node("RandomNoise", noise_seed=seed),
        "latent": node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1),
        "sample": node("SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0], sampler=["sampler", 0],
                       sigmas=["sigmas", 0], latent_image=["latent", 0]),
        "decode": node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "save": node("SaveImage", images=["decode", 0], filename_prefix="atas/still"),
    })
    return wf


def generate_image(prompt: str, output: Path, *, references: Sequence[Path] = (), aspect: float | None = None,
                   settings: ComfyUISettings | None = None, client: ComfyUIClient | None = None) -> Path:
    settings = settings or ComfyUISettings()
    if len(references) > settings.image_max_references:
        raise ValueError(f"{len(references)} reference images; the image workflow takes "
                         f"{settings.image_max_references} (COMFYUI_IMAGE_MAX_REFERENCES)")
    client = client or ComfyUIClient(settings.url, timeout=settings.request_timeout_seconds)
    from atas.providers.gpu_share import release_for
    release_for("comfyui")
    names = [client.upload(Path(path), subfolder=settings.upload_subfolder) for path in references]
    prompt = edit_prompt(prompt, settings.image_edit_tag)
    # Same prompt and references give the same seed, so a re-render is reproducible.
    seed = int(hashlib.sha256("\n".join([prompt, *names]).encode()).hexdigest()[:12], 16)
    width, height = output_size(aspect, settings)
    prompt_id = client.queue(build_workflow(settings, prompt=prompt, seed=seed, width=width, height=height,
                                            images=names), prompt_id=str(uuid.uuid4()))
    try:
        return _collect(client, prompt_id, Path(output), settings)
    finally:
        if settings.free_after_download:
            try:
                client.free()
            except ComfyUIError:
                pass


def _collect(client: ComfyUIClient, prompt_id: str, output: Path, settings: ComfyUISettings) -> Path:
    deadline = time.monotonic() + settings.task_timeout_seconds
    while (entry := client.history(prompt_id)) is None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"ComfyUI image {prompt_id} did not finish within {settings.task_timeout_seconds:.0f}s")
        time.sleep(settings.poll_interval_seconds)
    error = execution_error(entry)
    if error:
        raise failure(prompt_id, error)
    files = output_files(entry, "save")
    if not files:
        raise ComfyUIError(f"ComfyUI image {prompt_id} finished without output")
    client.download(files[0], output)
    return output
