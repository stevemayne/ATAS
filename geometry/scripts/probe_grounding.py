"""Explain a camera-grounding failure for one stored frame.

Prints, for each region (bbox + seed in 0..1 image coordinates), how many valid
depth pixels the seed neighbourhood has, how many of them carry the dynamic
(SAM3) mask, how much of the region is masked as sky, and the depth spread that
decides "ambiguous seed depth". Writes an overlay PNG of the dynamic/sky masks.

    python scripts/probe_grounding.py L0_DIR FRAME "label:x0,y0,x1,y1:u,v" ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from atas_geometry.memory.vista4d_layers import DEFAULT_MASK_DILATE, _mask, _read_exr_depth


def main(l0: Path, frame: int, regions: list[str]) -> None:
    depth = _read_exr_depth(sorted((l0 / "depths").glob("*.exr"))[frame])
    h, w = depth.shape
    dynamic_files = sorted((l0 / "dynamic_mask").glob("*.png"))
    sky_files = sorted((l0 / "sky_mask").glob("*.png"))
    dynamic = _mask(dynamic_files[frame], depth.shape, dilate=DEFAULT_MASK_DILATE) if frame < len(dynamic_files) else np.zeros_like(depth, bool)
    sky = _mask(sky_files[frame], depth.shape) if frame < len(sky_files) else np.zeros_like(depth, bool)
    valid = np.isfinite(depth) & (depth > 1e-5) & ~sky
    print(f"frame {frame}: {w}x{h}, dynamic masks {len(dynamic_files)}, sky masks {len(sky_files)}")
    print(f"  whole frame: valid {valid.mean():.1%}, dynamic {dynamic.mean():.1%}, sky {sky.mean():.1%}")
    ys, xs = np.mgrid[0:h, 0:w]
    uv = np.stack(((xs + .5) / w, (ys + .5) / h), axis=-1)
    for spec in regions:
        label, box, seed = spec.split(":")
        box, seed = np.array([float(v) for v in box.split(",")]), np.array([float(v) for v in seed.split(",")])
        inside = np.all((uv >= box[:2]) & (uv <= box[2:]), axis=-1)
        near = inside & (np.linalg.norm((uv - seed) / np.maximum(box[2:] - box[:2], .01), axis=-1) < .2)
        print(f"  {label}: region valid {valid[inside].mean():.1%} dynamic {dynamic[inside].mean():.1%} "
              f"sky {sky[inside].mean():.1%}")
        for name, keep in (("dynamic", near & valid & dynamic), ("static", near & valid & ~dynamic)):
            z = depth[keep]
            if len(z) < 16:
                print(f"    seed {name}: {len(z)} px (needs 16)")
                continue
            median = float(np.median(z))
            mad = float(np.median(np.abs(z - median)))
            print(f"    seed {name}: {len(z)} px, depth median {median:.3f}, mad/median {mad / median:.2f} "
                  f"(ambiguous above 0.15), p10-p90 {np.percentile(z, 10):.3f}-{np.percentile(z, 90):.3f}")
    overlay = np.zeros((h, w, 3), np.uint8)
    overlay[..., 0] = dynamic * 255
    overlay[..., 2] = sky * 255
    overlay[..., 1] = (~valid & ~sky) * 255
    out = Path(f"probe_frame{frame}.png")
    Image.fromarray(overlay).save(out)
    print(f"overlay (red=dynamic, blue=sky, green=no depth): {out.resolve()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
