# Exploration notes: running ATAS with MiniMax H3 on ComfyUI

Notes from reading the code, 27–29 Oct 2026. Nothing here has been implemented or tested yet. File
references are relative to the repository root.

## 1. What ATAS does

ATAS ("Any Time, Any Space", [arXiv 2610.20571](https://arxiv.org/abs/2610.20571)) turns a plain-text
screenplay into a multi-shot film. Text-to-video models usually forget what a place looked like a few shots
ago. ATAS avoids this by keeping one persistent world memory for the whole film: a 3D model of each place,
plus a record of what happened there and when. Every shot is generated from that memory, and every
generated shot is checked before it is allowed to update it.

## 2. Architecture

The system has two processes that talk over HTTP.

### Client (`client/atas/`): no GPU needed

| Module | Role |
|---|---|
| `compiler/` | Uses an LLM to turn the screenplay into a list of shots, each with a scene, camera, time and event |
| `runner/pipeline.py` | Runs each shot through the loop below |
| `shot/` | Plans the camera, creates opening stills (gpt-image) and writes the video prompts |
| `providers/` | External services: Wan 3.0 on DashScope for video, and an OpenAI-compatible endpoint for the language, vision and image models |
| `review/` | Checks each clip with a vision model and retries if it fails |
| `world/` | The client's copy of the story state, and the HTTP client for the geometry service |
| `film/` | Joins the shots into `final.mp4` and writes `manifest.json` and `report.md` |

### Geometry service (`geometry/atas_geometry/`): NVIDIA GPU with about 20 GB free

| Module | Role |
|---|---|
| `reconstruct/` | Rebuilds each clip in 3D with Pi3X and finds moving people and objects with SAM3 |
| `register/` | Aligns a new clip to what is already stored about that place and validates the fit |
| `memory/` | Stores the world in three layers (below), keeping a version history |
| `render/` | Draws the stored scene from a requested camera path |
| `service/` | HTTP API on port 18300. Work is submitted as asynchronous jobs (`/v1/jobs`); the world is updated through `/v1/world-publication` |

The world memory has three layers (details in [design.md](design.md)):

- **L0, observations:** each accepted clip's per-frame depth, camera and moving-object masks. Never changed once committed.
- **L1, scene memory:** the fixed 3D points of each place, in one coordinate frame per place. Versions are only ever added.
- **L2, event memory:** what moved, indexed by event and time. It is kept separate from L1, which is what makes re-filming a past moment from a new angle possible.

### The per-shot loop

`StoryRunner._execute` in `client/atas/runner/pipeline.py` runs these steps for each shot:

1. **Bind** the shot to the earlier shot in the same scene or story, and to any past event it should re-film.
2. **Query memory.** The geometry service renders the stored scene from the planned camera path. Areas with no stored points are drawn grey.
3. **Opening still.** One of three sources:
   - the previous shot's last frame, when the shot continues from it;
   - a frame of the original event, when re-filming a past moment;
   - otherwise, a still painted by gpt-image.
4. **Generate** the clip with Wan 3.0. The geometry render is sent as a reference video.
5. **Review the film (`f`).** A vision model reviews the clip, with a limited number of repair attempts. The film is delivered once it passes.
6. **Reconstruct and register (`r`).** The clip is rebuilt in 3D, conditioned on earlier trusted frames of the same place. If this fails, the film is still delivered but does not update the world. The shot is never re-filmed for this reason.
7. **Publish (`a`).** The new world version is committed atomically, and only if its parent is still the current version.

Shots run strictly one after another, and each step finishes before the next starts. Reconstruction finishes
before the next shot begins.

Re-running the same command resumes without paying for generations again. `--memory none` turns the world
memory off, which is the baseline the paper compares against.

## 3. How the grey gaps get filled with Wan

The client never fills the grey areas deliberately. No prompt mentions grey, and there is no inpainting
step and no mask. They get filled because of how the inputs are split up:

1. **The geometry video only guides camera and layout.** Unobserved pixels are drawn in flat grey, value 127 (`geometry/atas_geometry/render/render.py:13`). Point size is capped so gaps are not hidden (`render/point_render_policy.py`). The default prompt calls Video 1 point-cloud geometry and occlusion, and tells Wan to follow its camera motion (`client/atas/providers/wan/generation.py:85`). It never asks Wan to copy its look, so a grey area is simply unconstrained.
2. **A complete opening still supplies the appearance.** For a new camera angle, gpt-image paints a full frame from two inputs: a reference image of the scene, and a composition guide (the render's first frame with the subjects' projected positions boxed). See `client/atas/shot/stills.py:28`. The prompt tells Wan to lock that still's lighting, time of day and character positions, but to take the camera from Video 1 (`generation.py:92`). For a continuation, the still is the previous shot's last frame, which is already complete.
3. **The model invents the rest as it generates.** Film review is told not to count newly revealed detail as an error (`client/atas/review/film_review.py:29`).
4. **The memory learns from the result.** Once a shot is admitted, the newly visible static surfaces are added to the scene's points, so the next visit renders real content there. Holes left by masked people and objects fill the same way (`client/atas/world/dynamic_keywords.py:89`).

**A risk in `video_edit` mode.** This experimental mode tells Wan to turn the point-cloud geometry into
finished-film appearance and not to add objects that are not visible in Video 1 (`generation.py:79`). That
leans much harder on Video 1, so large grey areas are more likely to come out flat or grey.

## 4. Using MiniMax H3 through ComfyUI

### Why it is not plug-and-play

Every part of the system except the video-generation step is model-agnostic. That step is tied to Wan:

- **Direct import.** `pipeline.py` imports `build_request` and `generate_shot` from `atas.providers.wan.generation` (lines 345 and 640). There is no generator interface to plug another model into.
- **Wan request format.** `providers/wan/client.py` (about 1,700 lines) builds DashScope requests in Wan's media format and enforces Wan's limits.
- **Wan-specific routing.** `providers/wan/router.py` spreads requests across Wan's two model tiers and recognises DashScope's errors by their message text.
- **Reference cap.** `client/atas/shot/packing.py:11` sets `WAN30_REFERENCE_CAP = 5`, which is Wan's limit.
- **Prompt wording.** The prompts in `generation.py` describe references the way Wan expects them.

### What the generator has to support

For each shot the generator must accept:

1. **The geometry render as a video.** This is essential: without it there is no camera control and no consistency when the story returns to a place.
2. **A complete opening still**, used as the first frame.
3. **Character reference images.**
4. **Seed control**, which the client uses to retry an event shot.

### Fit with ComfyUI's H3 node

ComfyUI has a built-in MiniMax H3 reference-to-video node. In the prompt, references are named by the order
they are connected: "Image 1", "Video 1", and so on. According to the docs it accepts up to three
reference videos, each 2–15 s and 15 s in total. Sources disagree on the number of image slots (5 or 9),
so check the installed version and set the reference cap to match.

Planned mapping:

- **Video 1:** the geometry render, described only as a camera and layout guide.
- **Image 1:** the opening still.
- **Images 2 and up:** character sheets.

### Adapter plan (`client/atas/providers/comfyui/`)

1. Build the H3 workflow in ComfyUI and export it with **Save (API format)**.
2. For each shot:
   - upload the geometry video, still and character images (`POST /upload/image`, which also accepts video);
   - fill in the prompt, seed, duration and file names using a node-ID mapping kept in config;
   - queue the job (`POST /prompt`) and wait on `/history/{prompt_id}` or the websocket;
   - download the output (`/view`) into the shot folder.
3. Store the `prompt_id` in the generation journal (as `providers/wan/journal.py` does for Wan), so a resumed run picks up the finished output instead of generating again.
4. Choose the provider by a setting instead of importing Wan directly in `pipeline.py`.
5. Make the reference cap and duration limits depend on the provider.
6. Reword the prompts for H3, for example "Video 1 is the camera and layout guide…". State explicitly that grey regions are unknown and should be continued from the opening still: Wan may not need this, but another model might copy the grey literally.
7. Validate on the `examples/` stories, comparing memory on against `--memory none`. Include a wide establishing shot of a place seen only once before, because its render will contain a lot of grey.

Keeping the node mapping in config means another ComfyUI video workflow could later be swapped in without code
changes.

### Open questions

- How closely does H3 follow a reference video's camera motion and layout, rather than treating it as loose style? Only testing will tell.
- Licence terms for running the H3 weights locally. One source mentions restrictions.
- Local open weights or the hosted H3 Max partner nodes? The partner nodes use credits but allow several jobs at once.

## 5. Running the geometry service remotely

This is supported. The client was written for a geometry service on another machine:

- **Uploads:** finished clips are sent to the service as file uploads (`client/atas/world/geometry_client.py:515`). No shared filesystem is needed.
- **Downloads:** renders come back through an authenticated `/v1/jobs/{id}/artifacts/...` route. The client never looks up file paths on the GPU machine (`geometry_client.py:617`), and `client/tests/test_remote_artifact_fetch.py` covers this.
- **Authentication:** requests carry a bearer token (`GEOMETRY_API_TOKEN`).
- **Waiting and resuming:** the client polls job status, waits through short outages of the service, and saves job IDs so a resumed run picks up where it stopped.

Configuration:

```
# geometry/.env on the GPU machine
GEOMETRY_HOST=0.0.0.0            # default 127.0.0.1 accepts local connections only
GEOMETRY_API_TOKEN=<long random string>

# client/.env on the client machine
GEOMETRY_SERVICE_URL=https://your-gpu-host
GEOMETRY_API_TOKEN=<same token>
```

Caveats:

- **Use HTTPS or an SSH tunnel.** The token travels as an ordinary header. The simplest secure option is `ssh -L 18300:127.0.0.1:18300 gpu-host`: the service stays bound to localhost and the client keeps `http://127.0.0.1:18300`. If the URL is supplied through the optional endpoint file instead (`GEOMETRY_ENDPOINT_FILE`), the client accepts only HTTPS or local HTTP. That file can also be updated mid-run, for example after a tunnel restarts.
- **The world lives on the GPU machine,** under `GEOMETRY_HOT_ROOT`, so keep that disk persistent. `GEOMETRY_ARCHIVE_ROOT` can keep a copy of committed worlds elsewhere.
- **Bandwidth is modest:** one clip up and one render down per shot, a few MB to tens of MB each.

## 6. Sharing one Windows GPU between ComfyUI and the geometry service

### Geometry side: already releases VRAM

- **Each job runs in its own process.** With the default `GEOMETRY_RESIDENT_MODELS=0`, every job is a separate worker that loads Pi3X and SAM3 and then exits, returning its VRAM (`geometry/atas_geometry/service/resident_models.py:47`). Leave this setting off.
- **Jobs wait for free memory.** Before starting, the service checks free memory with `nvidia-smi` (`service/gpu_pool.py:21`). A reconstruction needs 24 GB (`GEOMETRY_RECON_MIN_FREE_MIB`) and a render 15 GB (`GEOMETRY_GPU_MIN_FREE_MIB`). If the card is busy the job queues; if it waits too long, it fails with a retryable error.

### ComfyUI side: release models before each geometry job

ComfyUI keeps models in VRAM after a run. `POST /free` with
`{"unload_models": true, "free_memory": true}` releases them.

The pipeline is strictly sequential (render → generate and retries → reconstruct → next shot), so no lock is
needed:

- The ComfyUI adapter calls `/free` after downloading each finished clip.
- The geometry client could also call it before every render and reconstruction. That covers runs where ComfyUI was used by something else in the meantime.

The cost is reloading H3 for every generation, which is small next to generation time.

### Windows: use WSL2

The geometry service does not run on native Windows:

- **Process management:** `os.killpg` and Unix signals (`service/worker_protocol.py:104`).
- **File locking:** `fcntl` in the optional archive (`world/archive.py:4`).
- **Dependencies:** Triton (SAM3, `backbones/sam3/model/edt.py`), `flash_attn` and `xformers`. All are hard or impossible to install on Windows.
- **Launcher:** a bash script.

Run it under WSL2 (Ubuntu) with the NVIDIA WSL driver:

- **GPU sharing:** WSL2 uses the same GPU as Windows ComfyUI, and `nvidia-smi` inside WSL reports memory use for the whole card, so the free-memory check sees what ComfyUI holds.
- **Storage:** put `GEOMETRY_HOT_ROOT` on the Linux filesystem (for example `~/geometry-hot`), not under `/mnt/c`. Access to the Windows drive from WSL is slow.
- **Network:** forward the Windows port into WSL (or enable WSL mirrored networking) for remote access, and add HTTPS or an SSH tunnel.

### Windows driver setting

When an app asks for more VRAM than is free, the Windows NVIDIA driver can quietly spill into system RAM
instead of failing. Everything then runs many times slower, which looks like a hang. In the NVIDIA Control
Panel, set **CUDA – Sysmem Fallback Policy** to **Prefer No Sysmem Fallback**, so running out of memory
produces an error the service can retry.

## Sources

- [ComfyUI docs – MinimaxHailuo03ReferenceNode](https://docs.comfy.org/zh/built-in-nodes/MinimaxHailuo03ReferenceNode.md)
- [ComfyUI Wiki – MiniMax H3 in ComfyUI](https://comfyui-wiki.com/en/tutorial/advanced/video/minimax/minimax-h3)
- [ComfyUI Wiki – H3 Max partner nodes](https://comfyui-wiki.com/en/news/2026-09-05-minimax-h3-max-comfyui)
- [InstaSD – MiniMax H3 ComfyUI nodes, open weights](https://instasd.com/post/minimax-h3-comfyui-nodes-open-weights)
- [RuntimeWire – ComfyUI H3 workflows and licence restrictions](https://runtimewire.com/zh/article/comfyui-minimax-h3-workflows-us-license-restrictions)
- [Oxen.ai – MiniMax H3](https://docs.oxen.ai/inference-api/reference/models/minimax-h3)
- [EvoLink – Hailuo 3 API guide](https://evolink.ai/blog/hailuo-3-api-guide)
