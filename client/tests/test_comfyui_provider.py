from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from atas.providers import video
from atas.providers.comfyui import generation
from atas.providers.comfyui.client import ComfyUIError, output_files
from atas.providers.comfyui.settings import ComfyUISettings, NodeMap
from atas.providers.comfyui.workflow import choose_duration, fill_workflow, h3_prompt
from atas.providers.wan.types import MediaReference, ReferenceRole

TEMPLATE = {
    "3": {"class_type": "MinimaxH3Reference", "inputs": {
        "prompt": "", "seed": 0, "duration": 6,
        "video_1": ["10", 0], "image_1": ["11", 0], "image_2": ["12", 0], "image_3": ["14", 0]}},
    "10": {"class_type": "LoadVideo", "inputs": {"file": "x.mp4"}},
    "11": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "12": {"class_type": "LoadImage", "inputs": {"image": "b.png"}},
    "13": {"class_type": "LoadImage", "inputs": {"image": "c.png"}},
    "14": {"class_type": "ImageScale", "inputs": {"image": ["13", 0], "width": 512}},
    "20": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0]}},
}
NODE_MAP = {
    "generator": "3",
    "prompt": {"node": "3", "input": "prompt"},
    "seed": {"node": "3", "input": "seed"},
    "duration": {"node": "3", "input": "duration"},
    "videos": [{"node": "10", "input": "file"}],
    "images": [{"node": "11", "input": "image"}, {"node": "12", "input": "image"},
               {"node": "13", "input": "image"}],
    "output": {"node": "20"},
}


def test_h3_prompt_renames_wan_handles_and_explains_grey() -> None:
    value = h3_prompt("用@视频1跟随相机；@图片1是开场静帧；视频1中不可见的物体不要添加", has_geometry=True, has_still=True)
    assert "@" not in value and "视频" not in value and "图片" not in value
    assert value.startswith("用<Video 1>跟随相机；<Picture 1>是开场静帧；<Video 1>中")
    assert "grey areas in <Video 1> are unknown" in value
    assert "opening still <Picture 1>" in value
    assert "grey" not in h3_prompt("x", has_geometry=False, has_still=True)
    assert "<Picture 1>" not in h3_prompt("x", has_geometry=True, has_still=False)
    assert h3_prompt("@图片2", has_geometry=False, has_still=False, image_tag="<image {n}>") == "<image 2>"


def test_choose_duration() -> None:
    assert choose_duration(4.4, minimum=2, maximum=15, allowed=[]) == 4.4
    assert choose_duration(1, minimum=2, maximum=15, allowed=[]) == 2
    assert choose_duration(30, minimum=2, maximum=15, allowed=[]) == 15
    assert choose_duration(7.9, minimum=2, maximum=15, allowed=[6, 10]) == 6
    assert choose_duration(8.5, minimum=2, maximum=15, allowed=[6, 10]) == 10


def test_fill_workflow_sets_inputs_and_drops_unused_loaders() -> None:
    node_map = NodeMap.model_validate(NODE_MAP)
    filled = fill_workflow(TEMPLATE, node_map, prompt="go", seed=7, duration=4,
                           videos=["atas/v.mp4"], images=["atas/still.png"])
    generator = filled["3"]["inputs"]
    assert (generator["prompt"], generator["seed"], generator["duration"]) == ("go", 7, 4)
    assert filled["10"]["inputs"]["file"] == "atas/v.mp4"
    assert filled["11"]["inputs"]["image"] == "atas/still.png"
    # Image 2 and 3 are unused: their loaders, and the resize fed by one, are removed,
    # and the generator keeps no dangling link.
    assert {"12", "13", "14"}.isdisjoint(filled)
    assert "image_2" not in generator and "image_3" not in generator
    assert generator["image_1"] == ["11", 0]
    assert TEMPLATE["3"]["inputs"]["image_2"] == ["12", 0]  # template untouched
    with pytest.raises(ValueError):
        fill_workflow(TEMPLATE, node_map, prompt="go", images=["a", "b", "c", "d"])


def test_output_files_prefers_video() -> None:
    entry = {"outputs": {"20": {"images": [{"filename": "f.png"}, {"filename": "f.mp4", "type": "output"}]}}}
    assert output_files(entry, "20")[0]["filename"] == "f.mp4"


@pytest.fixture
def comfy_config(tmp_path, monkeypatch):
    (tmp_path / "workflow.json").write_text(json.dumps(TEMPLATE))
    (tmp_path / "node_map.json").write_text(json.dumps(NODE_MAP))
    monkeypatch.setenv("ATAS_VIDEO_PROVIDER", "comfyui")
    monkeypatch.setenv("COMFYUI_WORKFLOW", str(tmp_path / "workflow.json"))
    monkeypatch.setenv("COMFYUI_NODE_MAP", str(tmp_path / "node_map.json"))
    return tmp_path


def test_character_slots_follow_workflow_image_inputs(comfy_config) -> None:
    assert video.character_slots(has_geometry=True, has_still=True) == 2
    assert video.character_slots(has_geometry=False, has_still=False) == 3
    refs = [SimpleNamespace(media_type="image")] * 4
    with pytest.raises(RuntimeError):
        video.check_packing("s1", refs)


def write_video(path: Path, seconds: float, fps: int = 8) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for index in range(int(seconds * fps)):
        writer.write(np.full((32, 32, 3), index % 255, np.uint8))
    writer.release()
    return path


class FakeComfy:
    def __init__(self, clip: Path, *, fail: bool = False):
        self.clip, self.fail = clip, fail
        self.queued: list[tuple[str, dict]] = []
        self.finished: dict[str, dict] = {}
        self.freed = 0

    def upload(self, path, *, subfolder=""):
        return generation._upload_name(path, subfolder)

    def queue(self, workflow, *, prompt_id, extra_data=None):
        self.queued.append((prompt_id, workflow))
        status = ({"status_str": "error", "completed": False,
                   "messages": [["execution_error", {"node_id": "3", "exception_message": "boom"}]]}
                  if self.fail else {"status_str": "success", "completed": True})
        self.finished[prompt_id] = {"status": status, "outputs": {"20": {"images": [{"filename": "out.mp4"}]}}}
        return prompt_id

    def history(self, prompt_id):
        return self.finished.get(prompt_id)

    def in_queue(self, prompt_id):
        return False

    def download(self, file, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(self.clip, destination))

    def free(self):
        self.freed += 1


def make_request(tmp_path: Path):
    geometry = write_video(tmp_path / "geometry.mp4", 4)
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.zeros((8, 8, 3), np.uint8))
    refs = (MediaReference(geometry, ReferenceRole.TARGET_GEOMETRY, 1, "video", "geometry"),
            MediaReference(still, ReferenceRole.FIRST_FRAME, 1, "image", "still"))
    return SimpleNamespace(prompt="用@视频1；@图片1是开场静帧", shot=SimpleNamespace(seed=3, duration=4.0),
                           condition=SimpleNamespace(references=refs), output_dir=tmp_path / "shot")


def test_generate_submits_once_and_resumes_without_requeue(comfy_config, tmp_path) -> None:
    request = make_request(tmp_path)
    fake = FakeComfy(write_video(tmp_path / "clip.mp4", 2))
    first = generation._generate(request, ComfyUISettings(_env_file=None), fake)
    assert first.is_file() and first.parent.parent == request.output_dir / "comfyui"
    assert len(fake.queued) == 1 and fake.freed == 1
    submitted = fake.queued[0][1]
    assert submitted["3"]["inputs"]["seed"] == 3 and submitted["3"]["inputs"]["duration"] == 4
    assert "<Video 1>" in submitted["3"]["inputs"]["prompt"]
    assert {"12", "13", "14"}.isdisjoint(submitted)
    assert generation._generate(request, ComfyUISettings(_env_file=None), fake) == first
    assert len(fake.queued) == 1

    # Result file lost after the prompt finished: download again, never re-queue.
    first.unlink()
    assert generation._generate(request, ComfyUISettings(_env_file=None), fake).is_file()
    assert len(fake.queued) == 1


def test_crash_after_journal_requeues_same_prompt_id(comfy_config, tmp_path) -> None:
    request = make_request(tmp_path)
    fake = FakeComfy(write_video(tmp_path / "clip.mp4", 2))
    original_queue = fake.queue

    def crash(workflow, *, prompt_id, extra_data=None):
        raise ConnectionError("lost before ComfyUI saw it")
    fake.queue = crash
    with pytest.raises(ConnectionError):
        generation._generate(request, ComfyUISettings(_env_file=None), fake)
    journaled = [json.loads(p.read_text()) for p in (request.output_dir / "comfyui").glob("*.json")
                 if p.name != "workflow.json"]
    assert len(journaled) == 1 and journaled[0]["status"] == "queued"
    fake.queue = original_queue
    generation._generate(request, ComfyUISettings(_env_file=None), fake)
    assert [pid for pid, _ in fake.queued] == [journaled[0]["prompt_id"]]


def test_failed_prompt_is_retired_and_next_attempt_queues_new_id(comfy_config, tmp_path) -> None:
    request = make_request(tmp_path)
    fake = FakeComfy(write_video(tmp_path / "clip.mp4", 2), fail=True)
    with pytest.raises(ComfyUIError, match="boom"):
        generation._generate(request, ComfyUISettings(_env_file=None), fake)
    fake.fail = False
    generation._generate(request, ComfyUISettings(_env_file=None), fake)
    assert len({pid for pid, _ in fake.queued}) == 2


def test_rejects_reference_video_outside_h3_limits(comfy_config, tmp_path) -> None:
    request = make_request(tmp_path)
    write_video(request.condition.references[0].path, 1)
    with pytest.raises(ValueError, match="accepts 2-15s"):
        generation._generate(request, ComfyUISettings(_env_file=None), FakeComfy(tmp_path / "unused.mp4"))


def test_shipped_h3_template_matches_its_node_map() -> None:
    settings = ComfyUISettings(_env_file=None)
    template = json.loads(settings.resolve(settings.workflow, "workflow").read_text())
    node_map = NodeMap.model_validate_json(settings.resolve(settings.node_map, "node_map").read_text())
    assert len(node_map.images) == 9 and len(node_map.videos) == 1
    filled = fill_workflow(template, node_map, prompt="p", seed=5, duration=4.0,
                           videos=["atas/g.mp4"], images=["atas/a.png", "atas/b.png"])
    generator = filled[node_map.generator_node]["inputs"]
    assert generator["prompt"] == "p" and filled["129"]["inputs"]["noise_seed"] == 5
    assert filled["132"]["inputs"]["value"] == 4.0
    assert generator["ref_videos.ref_video_0"] == ["161", 0]
    assert [key for key in generator if key.startswith("ref_images.")] == [
        "ref_images.ref_image_0", "ref_images.ref_image_1"]
    assert not any(node["class_type"] == "LoadImage" and not node["inputs"]["image"] for node in filled.values())
    # Without a geometry render the video loader and its frame splitter both go.
    no_video = fill_workflow(template, node_map, prompt="p", images=["atas/a.png"])
    assert {"160", "161"}.isdisjoint(no_video)
    assert "ref_videos.ref_video_0" not in no_video[node_map.generator_node]["inputs"]


def test_long_clip_is_trimmed_to_shot(comfy_config, tmp_path) -> None:
    request = make_request(tmp_path)
    fake = FakeComfy(write_video(tmp_path / "clip.mp4", 6))
    result = generation._generate(request, ComfyUISettings(_env_file=None), fake)
    assert abs(generation._video_seconds(result) - 4.0) < 0.2
    assert not (result.parent / "download.mp4").exists()


def test_size_and_turbo_settings_reach_the_template() -> None:
    settings = ComfyUISettings(_env_file=None)
    template = json.loads(settings.resolve(settings.workflow, "workflow").read_text())
    node_map = NodeMap.model_validate_json(settings.resolve(settings.node_map, "node_map").read_text())
    filled = fill_workflow(template, node_map, prompt="p", width=960, height=544, turbo=True, videos=["v.mp4"])
    assert (filled["136"]["inputs"]["width"], filled["136"]["inputs"]["height"]) == (960, 544)
    assert filled["146"]["inputs"]["value"] is True
    untouched = fill_workflow(template, node_map, prompt="p", videos=["v.mp4"])
    assert untouched["136"]["inputs"]["width"] == 1344 and untouched["146"]["inputs"]["value"] is False


def test_memory_failure_is_retryable_and_still_frees(comfy_config, tmp_path) -> None:
    from atas.providers.comfyui.client import ComfyUIUnavailable
    request = make_request(tmp_path)
    fake = FakeComfy(write_video(tmp_path / "clip.mp4", 2), fail=True)
    original = fake.queue

    def queue(workflow, *, prompt_id, extra_data=None):
        result = original(workflow, prompt_id=prompt_id)
        fake.finished[prompt_id]["status"]["messages"][0][1]["exception_message"] = "hostbuf_file_reader_read failed"
        return result
    fake.queue = queue
    with pytest.raises(ComfyUIUnavailable):
        generation._generate(request, ComfyUISettings(_env_file=None), fake)
    assert fake.freed == 1


def test_lmstudio_load_failure_is_transient(monkeypatch) -> None:
    from atas.providers.llm import LLMError, transient
    error = LLMError('model HTTP 400: {"error": {"message": "Failed to load model \\"gemma\\"."}}')
    monkeypatch.delenv("ATAS_CHAT_URL", raising=False)
    assert not transient(error)
    monkeypatch.setenv("ATAS_CHAT_URL", "http://lm:1234/v1")
    assert transient(error)
