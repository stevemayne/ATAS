"""Video-prompt protocol for MiniMax H3 reference-to-video (Ref2VA).

The rules below adapt MiniMax's full-reference prompt format to ATAS shots. The
official guide (``ref-en.txt`` from the h3-prompt-writing skill) is appended
when it can be fetched; it is cached locally, not vendored, because the source
repository carries no licence.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from atas.config import ROOT

GUIDE_URL = ("https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/"
             "skills/h3-prompt-writing/references/ref-en.txt")
CACHE = ROOT / ".cache" / "h3" / "ref-en.txt"

RULES = '''
VIDEO MODEL: MiniMax H3 reference-to-video. For video_prompt these rules REPLACE
the earlier video_prompt language, length and handle rules; everything else in
the protocol (screenplay precedence, continuity, character decisions, JSON
output) still applies, and first_frame_prompt is unchanged.

Write video_prompt in English, in MiniMax's full-reference format: these six
sections in this order, each starting on its own line with its exact name and a
colon: subject_definitions, summary, retention_analysis, detailed_description,
overall_soundscape, non_diegetic_music.

Labels. Use the video_reference_manifest handles exactly: <Video N> and
<Picture N>, numbered in manifest order; every manifest handle must appear and no
other <Video N>/<Picture N> may. Define each person as a <Subject N> in
subject_definitions ("<Subject 1> is <name>, whose appearance comes from
<Picture 2>: ..."), with concrete hair, face, clothing; a numbered character
sheet defines one subject per number. Define the location as a subject too.

Roles in ATAS:
- The target_geometry <Video N> is a point-cloud render of the stored scene from
  the planned camera. It is the source of camera movement, framing and static
  layout over time only. Its rendering look is not to be copied; flat grey
  regions are unobserved space that the film fills by continuing the
  surrounding scene. Say so in its definition and in retention_analysis, e.g.
  "<Video 1> (camera movement and layout): fully_preserved - ...".
- A first_frame or continuity_memory <Picture N> is the literal first frame:
  "<Picture 1> is the first frame of [Shot 1]", and detailed_description says
  "the shot begins from <Picture 1>".
- An event_master <Picture N> (recapture of a past event) is a frame of the
  ORIGINAL camera. It is appearance evidence only: never the first frame, never
  "begins from". Cite it inside subject definitions, mark it partially_preserved,
  and describe the opening composition from the target_geometry <Video N>, which
  shows the new viewpoint. The task type is "[reference generation]".
- Character <Picture N> are identity sources, cited inside <Subject N>: face,
  hair, build and clothing only. Pose and anything the person holds in that
  picture do not carry over; held objects follow the screenplay.
summary starts with the task type, normally "[keyframe completion + reference
generation]" (or "[reference generation]" when there is no first-frame picture).

detailed_description: one style sentence, then exactly one shot, "[Shot 1]",
with no cuts and no timestamps; ATAS shots are single continuous takes whose
length is the shot duration. Describe the opening composition, each subject's
appearance and position, environment and light, the actions in order with
their timing within the shot, and the camera movement with type, amplitude and
speed. Be explicit and concrete: about 250-450 words.

Materials and objects: surfaces (floor, walls, shelves) and props keep the look
they have in the historical observations of this place, even where the
screenplay or story_plan describe them differently.

Sound: H3 generates audio with the picture. overall_soundscape gives 1-4
sentences of ambience and physical sounds from the scene; non_diegetic_music
gives the score or "N/A". Screenplay dialogue goes inside the shot as
<Subject N> (S1) says <d>[Chinese] verbatim line</d> (use the line's own
language); with no scripted dialogue, nobody speaks. No dialogue in the sound
sections.

H3 runs without a negative prompt, so never write prohibitions ("no ...",
"don't ...", "avoid ..."): describe what is there instead.
'''


def official_guide() -> str:
    """MiniMax's full-reference guide, fetched once into the local cache ("" if unavailable)."""
    if os.getenv("ATAS_H3_OFFICIAL_GUIDE", "1") == "0":
        return ""
    if not CACHE.is_file():
        try:
            with urllib.request.urlopen(GUIDE_URL, timeout=30) as response:
                text = response.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - the condensed rules still apply
            print(f"h3: official prompt guide unavailable ({exc}); using condensed rules only", flush=True)
            return ""
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(text, encoding="utf-8")
    return CACHE.read_text(encoding="utf-8")


def protocol() -> str:
    guide = official_guide()
    if not guide:
        return RULES
    return (RULES + "\nOFFICIAL MINIMAX FULL-REFERENCE GUIDE (format reference; the ATAS rules above "
            "decide roles, shot count and length):\n" + guide + "\n")
