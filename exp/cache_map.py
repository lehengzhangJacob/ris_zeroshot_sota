"""Map an image file (RefCOCO-family file_name) to its glod.vision cache npz.

The cache key is digest(dict(image_sha256=..., settings=...)) where settings is
the dict built in glod/vision.py VisionEncoder.__init__ for the released
configuration (SAM ViT-H, pps=8, HybridGL G2L).  Replicated here so offline
re-analysis does not have to instantiate SAM.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
CACHE = REPRO / "cache/vision"
CACHE_PPS16 = REPRO / "cache/vision_pps16"
IMAGES = REPRO / "data/refer/images/train2014"
MAP_FILE = Path("exp/out_extreme/cache_map.json")


def _settings(pps=8, min_region_area=800):
    sam = REPRO / "checkpoints/sam_vit_h_4b8939.pth"
    clip = REPRO / "checkpoints/ViT-B-16.pt"
    return dict(upstream_commit="f7eb19bb618353076f1da19485c1d9c6236a6f31",
                feature_version=1, preprocessing="official-hybridgl", fusion_mode="G2L",
                masking_block=9, sam_backbone="vit_h", points_per_side=pps,
                pred_iou_thresh=0.7, stability_score_thresh=0.7,
                min_mask_region_area=min_region_area,
                checkpoints=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                             for p in (sam, clip)])


def _image_path(file_name):
    for root in (IMAGES, REPRO / "data/refer/images"):
        p = root / file_name
        if p.is_file():
            return p
    raise FileNotFoundError(file_name)


@functools.lru_cache(maxsize=1)
def _map():
    if MAP_FILE.is_file():
        return json.loads(MAP_FILE.read_text())
    return {}


def cache_file(file_name, pps=8):
    m = _map()
    key = f"{pps}:{file_name}"
    if key in m:
        return Path(m[key])
    from glod.io import digest, file_sha256  # noqa: E402
    h = file_sha256(_image_path(file_name))
    d = digest(dict(image_sha256=h, settings=_settings(pps)))
    root = CACHE if pps == 8 else CACHE_PPS16
    path = root / (d + ".npz")
    return path


def build_map(file_names, pps=8, save=True):
    m = dict(_map())
    for fn in file_names:
        m[f"{pps}:{fn}"] = str(cache_file(fn, pps))
    if save:
        MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        MAP_FILE.write_text(json.dumps(m, indent=0, sort_keys=True))
        _map.cache_clear()
    return m


if __name__ == "__main__":
    import sys
    # validation: gRefCOCO record 0 has both file_name and cache_file
    rec = json.loads(open("exp/out_extreme/records_grefcoco.jsonl").readline())
    got = cache_file(rec["file_name"])
    print("derived:", got)
    print("record :", rec["cache_file"])
    print("MATCH" if str(got) == rec["cache_file"] else "MISMATCH")
    sys.exit(0)
