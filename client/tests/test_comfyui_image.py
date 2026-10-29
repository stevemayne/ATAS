from __future__ import annotations

from pathlib import Path

from PIL import Image

from atas.providers.comfyui import image
from atas.providers.comfyui.settings import ComfyUISettings


def settings() -> ComfyUISettings:
    return ComfyUISettings(_env_file=None)


def test_tags_stay_separate_for_h3_and_stills() -> None:
    assert settings().image_tag == "<Picture {n}>"
    assert settings().image_edit_tag == "<image {n}>"
    assert image.edit_prompt("Keep Image 1 framing; face from Image 2.", "<image {n}>") == \
        "Keep <image 1> framing; face from <image 2>."


def test_output_size() -> None:
    assert image.output_size(16 / 9, settings()) == (1360, 768)
    assert image.output_size(None, settings()) == (1024, 1024)


def test_references_chain_through_reference_latents() -> None:
    wf = image.build_workflow(settings(), prompt="p", seed=1, width=1360, height=768, images=["a.png", "b.png"])
    assert wf["pos2"]["inputs"]["conditioning"] == ["pos1", 0]
    assert wf["neg1"]["inputs"]["conditioning"] == ["zero", 0]
    assert wf["guider"]["inputs"]["positive"] == ["pos2", 0] and wf["guider"]["inputs"]["negative"] == ["neg2", 0]
    assert wf["load2"]["inputs"]["image"] == "b.png"
    assert wf["latent"]["inputs"]["width"] == 1360


def test_no_references_is_text_to_image() -> None:
    wf = image.build_workflow(settings(), prompt="p", seed=1, width=1024, height=1024, images=[])
    assert not any(node["class_type"] == "LoadImage" for node in wf.values())
    assert wf["guider"]["inputs"]["positive"] == ["text", 0]


class FakeComfy:
    def __init__(self):
        self.queued = []

    def upload(self, path, *, subfolder=""):
        return f"{subfolder}/{Path(path).name}"

    def queue(self, workflow, *, prompt_id, extra_data=None):
        self.queued.append(workflow)
        return prompt_id

    def history(self, prompt_id):
        return {"status": {"status_str": "success"}, "outputs": {"save": {"images": [{"filename": "s.png"}]}}}

    def download(self, file, destination):
        Image.new("RGB", (16, 9)).save(destination)
        return destination

    def free(self):
        pass


def test_generate_image_uploads_references_and_downloads(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ATAS_LMSTUDIO_URL", raising=False)
    ref = tmp_path / "guide.png"
    Image.new("RGB", (8, 8)).save(ref)
    fake = FakeComfy()
    out = image.generate_image("match Image 1", tmp_path / "still.png", references=[ref], aspect=16 / 9,
                               settings=settings(), client=fake)
    assert out.is_file()
    wf = fake.queued[0]
    assert wf["text"]["inputs"]["text"] == "match <image 1>"
    assert wf["load1"]["inputs"]["image"] == "atas/guide.png"
