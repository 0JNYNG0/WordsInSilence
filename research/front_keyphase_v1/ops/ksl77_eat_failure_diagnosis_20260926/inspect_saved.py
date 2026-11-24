"""Read-only KSL eat diagnosis: saved coordinates + source pixels, no extractor/model.

The focused rows follow the six-frame-per-clip review. They are diagnostic,
not a new independent evaluation or full-video visual annotation.
"""
from pathlib import Path
import hashlib
import json

import av
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[1]
PLAN = WEB / "ops/ksl77_development_probe_20260926/FROZEN_PLAN.json"
FOCUS = {"KSL77_09_16": [14, 15, 16, 17, 18],
         "KSL77_17_16": [4, 5, 6, 7, 8],
         "KSL77_01_16": [25, 26, 27],
         "KSL77_18_16": [13, 14, 15]}
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
         (15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_new(p, value):
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    sources = {r["id"]: r for r in plan["rows"]}
    index = json.loads((HERE / "SAMPLE_INDEX.json").read_text(encoding="utf-8"))
    out = []
    (HERE / "focus").mkdir(exist_ok=True)
    for r in index["rows"]:
        clip = r["id"]
        source = sources[clip]["path"]
        obs = WEB / "runs/ksl77_development_probe_v1" / clip / "observation.npz"
        assert sha(source) == r["source_sha256"]
        assert sha(obs) == r["observation_sha256"]
        with np.load(obs, allow_pickle=False) as z:
            xy, mask, ts = z["xy_px"], z["usable"], z["timestamp_us"]
            lo, hi = r["active_span"]
            hand = np.stack([mask[:,33:54].any(1), mask[:,54:75].any(1)], 1)
            any_hand = hand.any(1)
            assert np.flatnonzero(any_hand)[[0,-1]].tolist() == [lo,hi]
            missing = np.flatnonzero(~any_hand[lo:hi+1]) + lo
            row = {"id": clip, "prediction": r["prediction"], "agreement": r["agreement"],
                   "active_span_inclusive": [lo, hi], "active_frames": hi-lo+1,
                   "any_hand_missing_rows": missing.tolist(),
                   "left_only_rows": (np.flatnonzero(hand[:,0] & ~hand[:,1])).tolist(),
                   "right_only_rows": (np.flatnonzero(hand[:,1] & ~hand[:,0])).tolist(),
                   "both_rows": (np.flatnonzero(hand.all(1))).tolist(),
                   "sampled_rows": r["sampled_rows"],
                   "note": "usable masks are detector outputs, not ground truth"}
            if clip in FOCUS:
                wanted = FOCUS[clip]
                frames = {}
                with av.open(source) as c:
                    origin = None
                    last = -1
                    i = 0
                    for f in c.decode(video=0):
                        if f.pts is None:
                            continue
                        if origin is None:
                            origin = f.pts
                        us = round(float((f.pts-origin)*f.time_base)*1e6)
                        if us <= last:
                            continue
                        last = us
                        if i in wanted:
                            assert us == int(ts[i]), (clip, i)
                            frames[i] = f.to_image().convert("RGB")
                        i += 1
                assert len(frames) == len(wanted)
                first = next(iter(frames.values()))
                w, h = first.size
                crop = (int(w*.2), int(h*.1), int(w*.8), int(h*.95))
                size = (360, round(360*(crop[3]-crop[1])/(crop[2]-crop[0])))
                sheet = Image.new("RGB", (size[0]*len(wanted), (size[1]+25)*2), "white")
                for j, k in enumerate(wanted):
                    for annotated in (False, True):
                        frame = frames[k].copy()
                        d = ImageDraw.Draw(frame)
                        if annotated:
                            for a,b in [(11,13),(13,15),(12,14),(14,16)]:
                                if mask[k,a] and mask[k,b]:
                                    d.line([tuple(xy[k,a]),tuple(xy[k,b])], fill="lime", width=3)
                            for start, color in [(33,"cyan"),(54,"orange")]:
                                for a,b in EDGES:
                                    a,b=a+start,b+start
                                    if mask[k,a] and mask[k,b]:
                                        d.line([tuple(xy[k,a]),tuple(xy[k,b])], fill=color,width=3)
                                if mask[k,start]:
                                    x,y=xy[k,start]
                                    d.ellipse((x-5,y-5,x+5,y+5),fill=color)
                        yoff = int(annotated)*(size[1]+25)
                        sheet.paste(frame.crop(crop).resize(size), (j*size[0],yoff+25))
                        ImageDraw.Draw(sheet).text((j*size[0]+4,yoff+4),
                          f"row {k} {'L=cyan R=orange' if annotated else 'unmarked'}",fill="black")
                dest = HERE / "focus" / (clip+".png")
                assert not dest.exists(), "reuse finished render"
                sheet.save(dest)
                row["focus_rows"] = wanted
                row["focus_sheet"] = str(dest)
            out.append(row)
    save_new(HERE / "MEASUREMENTS.json", {"scope":"16 new eat clips; no new inference",
        "index_sha256":sha(HERE/'SAMPLE_INDEX.json'), "rows":out})
    print("Saved measurements for", len(out), "clips; focus sheets",len(FOCUS))


if __name__ == "__main__":
    main()
