"""ComfyUI video-generator configuration (``COMFYUI_*`` in ``client/.env``).

The workflow is in ComfyUI's API format (*Save (API format)*); the default is
the MiniMax H3 reference-to-video template in ``client/comfyui/``. The node map
says which node inputs receive the shot's prompt, seed, duration and media
(see ``client/comfyui/h3_ref2va_node_map.json``). ``videos`` and ``images`` are
listed in the order they are connected to the generator, which is the order H3
numbers them (<Video 1>, <Picture 1>, ...). Unused loaders, and nodes fed only
by them, are removed from the submitted workflow.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from atas.config import ENV_PATH, ROOT


class NodeInput(BaseModel):
    node: str
    input: str


class OutputNode(BaseModel):
    node: str


class NodeMap(BaseModel):
    prompt: NodeInput
    seed: NodeInput | None = None
    duration: NodeInput | None = None
    width: NodeInput | None = None
    height: NodeInput | None = None
    turbo: NodeInput | None = None
    videos: list[NodeInput] = Field(default_factory=list)
    images: list[NodeInput] = Field(default_factory=list)
    output: OutputNode
    # The node that consumes the loaders; it loses an unused loader's link
    # instead of being removed with it. Defaults to the prompt node.
    generator: str | None = None

    @property
    def generator_node(self) -> str:
        return self.generator or self.prompt.node


class ComfyUISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COMFYUI_",
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = "http://127.0.0.1:8188"
    workflow: Path | None = Path("comfyui/h3_ref2va_api.json")
    node_map: Path | None = Path("comfyui/h3_ref2va_node_map.json")
    # Comfy.org key for hosted partner nodes (e.g. H3 Max); not needed for local weights.
    comfy_org_api_key: SecretStr | None = Field(default=None, repr=False)
    upload_subfolder: str = "atas"
    poll_interval_seconds: float = 3.0
    request_timeout_seconds: float = 120.0
    task_timeout_seconds: float = 3600.0
    # Release VRAM after each clip so the geometry service on the same GPU can load.
    free_after_download: bool = True
    # H3 reference videos: 2-15 s each.
    min_reference_video_seconds: float = 2.0
    max_reference_video_seconds: float = 15.0
    # Seconds sent to the duration input: the shot duration clamped, or the
    # nearest allowed value when the workflow takes a fixed set.
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 15.0
    allowed_durations: list[float] = Field(default_factory=list)
    # H3 rounds length up to a valid frame count; cut the clip back to the shot
    # so it stays in step with its geometry render.
    trim_to_shot: bool = True
    # Output size and the turbo (few-step LoRA) branch; unset keeps the workflow's
    # own values. H3 needs multiples of 32: 960x544 is ~0.5 MP, 1344x768 native.
    width: int | None = None
    height: int | None = None
    turbo: bool | None = None
    # How the prompt names references, in connection order.
    image_tag: str = "<Picture {n}>"
    video_tag: str = "<Video {n}>"
    # Tell the model that grey in the geometry video is unknown space, not grey scenery.
    grey_region_hint: bool = True

    # Stills and character sheets (ATAS_IMAGE_BACKEND=comfyui): Flux.2 Klein image edit.
    image_unet: str = "flux-2-klein-9b.safetensors"
    image_clip: str = "qwen_3_8b_fp8mixed.safetensors"
    image_vae: str = "full_encoder_small_decoder.safetensors"
    image_steps: int = 4
    image_reference_megapixels: float = 1.0
    image_max_references: int = 6
    # Output size: height for a requested aspect (width follows, multiple of 16),
    # and a square side when no aspect is requested.
    image_height: int = 768
    image_square: int = 1024
    # How the edit prompt names references; stills are authored with "Image N".
    image_edit_tag: str = "<image {n}>"

    def resolve(self, path: Path | None, name: str) -> Path:
        if path is None:
            raise RuntimeError(f"COMFYUI_{name.upper()} is not set; see client/.env.example")
        return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def _load_node_map(path: Path) -> NodeMap:
    return NodeMap.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_node_map(settings: ComfyUISettings | None = None) -> NodeMap:
    settings = settings or ComfyUISettings()
    return _load_node_map(settings.resolve(settings.node_map, "node_map"))


def load_workflow(settings: ComfyUISettings | None = None) -> dict:
    settings = settings or ComfyUISettings()
    return json.loads(settings.resolve(settings.workflow, "workflow").read_text(encoding="utf-8"))
