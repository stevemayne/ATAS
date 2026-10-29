from __future__ import annotations

from pathlib import Path

import pytest

from atas.providers import video
from atas.providers.wan.types import MediaReference, ReferenceRole
from atas.shot.prompt_author import validate, video_manifest

REFS = (MediaReference(Path("g.mp4"), ReferenceRole.TARGET_GEOMETRY, 1, "video", "geometry"),
        MediaReference(Path("s.png"), ReferenceRole.FIRST_FRAME, 1, "image", "still"),
        MediaReference(Path("c.png"), ReferenceRole.CHARACTER, 2, "image", "Chen Mo", character_mapping=("chen_mo",)))


def response(prompt: str) -> dict:
    return {"status": "ready", "conflicts": [], "first_frame_prompt": "", "video_prompt": prompt,
            "character_decisions": [{"label": "chen_mo", "treatment": "walks in"}]}


@pytest.fixture
def h3(monkeypatch):
    monkeypatch.setenv("ATAS_VIDEO_PROVIDER", "comfyui")
    monkeypatch.setenv("ATAS_H3_OFFICIAL_GUIDE", "0")


def test_h3_manifest_uses_h3_labels(h3) -> None:
    assert [row["handle"] for row in video_manifest(REFS)] == ["<Video 1>", "<Picture 1>", "<Picture 2>"]


def test_h3_validation_requires_exactly_the_packed_labels(h3) -> None:
    manifest = video_manifest(REFS)
    ok = ("subject_definitions: <Subject 1> is Chen Mo, from <Picture 2>. <Picture 1> is the first frame of "
          "[Shot 1]. <Video 1> gives camera movement and layout.")
    validate(response(ok), ["chen_mo"], manifest, False)
    with pytest.raises(ValueError, match="handles differ"):
        validate(response(ok.replace("<Picture 2>", "<Picture 3>")), ["chen_mo"], manifest, False)
    with pytest.raises(ValueError, match="handles differ"):
        validate(response("@视频1 @图片1 @图片2"), ["chen_mo"], manifest, False)


def test_h3_allows_its_longer_format(h3) -> None:
    long = "<Video 1> <Picture 1> <Picture 2> " + "word " * 1500
    validate(response(long), ["chen_mo"], video_manifest(REFS), False)


def test_h3_protocol_carries_format_rules(h3) -> None:
    rules = video.prompt_protocol()
    for field in ("subject_definitions", "retention_analysis", "overall_soundscape", "[Shot 1]"):
        assert field in rules


def test_wan_is_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("ATAS_VIDEO_PROVIDER", "wan")
    assert [row["handle"] for row in video_manifest(REFS)] == ["@视频1", "@图片1", "@图片2"]
    assert video.prompt_protocol() == ""


def test_recapture_frame_must_not_be_anchored_as_first_frame(h3) -> None:
    refs = (REFS[0], MediaReference(Path("e.png"), ReferenceRole.EVENT_MASTER, 1, "image", "event"), REFS[2])
    manifest = video_manifest(refs)
    bad = "<Picture 1> is the first frame of [Shot 1]. <Video 1> camera. <Picture 2> Chen Mo."
    with pytest.raises(ValueError, match="appearance evidence only"):
        validate(response(bad), ["chen_mo"], manifest, False)
    with pytest.raises(ValueError, match="appearance evidence only"):
        validate(response("The shot begins from <Picture 1>. <Video 1> <Picture 2>"), ["chen_mo"], manifest, False)
    sneaky = "[Shot 1] The shot begins from the composition established in <Video 1> and <Picture 1>. <Picture 2> Chen."
    with pytest.raises(ValueError, match="appearance evidence only"):
        validate(response(sneaky), ["chen_mo"], manifest, False)
    good = ("<Subject 2> is the room, appearance from <Picture 1> (partially_preserved). The opening composition "
            "follows <Video 1>. <Picture 2> Chen Mo.")
    validate(response(good), ["chen_mo"], manifest, False)


def test_h3_recapture_still_gets_event_master_role(h3, tmp_path) -> None:
    from atas.providers.wan.generation import build_request
    from atas.request import plan_from_payload
    from PIL import Image
    alice = tmp_path / "alice.png"; Image.new("RGB", (8, 8)).save(alice)
    plan = plan_from_payload("demo", {"title": "t", "character_reference_map": {"alice": str(alice)}, "shots": [
        {"id": "s1", "location_id": "room", "event_id": "e", "start": 0, "duration": 4, "prompt": "x",
         "entity_ids": ["alice"], "character_reference_map": {"alice": str(alice)}}]}, tmp_path / "plan.json")
    still = tmp_path / "event.png"; Image.new("RGB", (8, 8)).save(still)
    request, _, _ = build_request(plan.shots[0], output_dir=tmp_path, geometry_video=tmp_path / "g.mp4",
                                  continuity_still=None, first_frame=still, same_event_recapture=True)
    roles = [r.role for r in request.condition.references if r.media_type == "image"]
    assert roles[0] is ReferenceRole.EVENT_MASTER
