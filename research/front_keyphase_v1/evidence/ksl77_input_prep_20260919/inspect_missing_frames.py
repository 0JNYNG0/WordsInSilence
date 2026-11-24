"""Inspect already-review-exposed prefix00 missingness; do not derive action labels."""
from pathlib import Path
import json

import av
import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
report = {"purpose": "targeted source-quality triage after new missing-hand evidence",
          "annotations_created": False, "feature_extraction_rerun": False, "clips": []}
for clip in result["clips"]:
    clip_id = clip["clip_id"]
    selected = [0, 9, 11, 20, 39, 54, 56, 59, 60, 68, 77, 84] if clip_id.endswith("_16") else [0, 9, 13, 16, 17, 22, 40, 50, 56, 60, 70, 88]
    obs_path = Path(clip["observation_path"])
    side = json.loads(obs_path.with_suffix(".sidecar.json").read_text(encoding="utf-8"))
    with np.load(obs_path, allow_pickle=False) as z:
        xy, seen, usable, scores, ts = (z[k] for k in ("xy_px", "observed", "usable", "hand_score", "timestamp_us"))
    per_hand = {}
    for name, s in (("left", slice(33, 54)), ("right", slice(54, 75))):
        per_hand[name] = {"no_assigned_observation_frames": np.flatnonzero(~seen[:, s].any(axis=1)).tolist(),
                          "observed_but_not_all_usable_frames": np.flatnonzero(seen[:, s].any(axis=1) & ~usable[:, s].all(axis=1)).tolist()}
    images = {}
    with av.open(side["source_path"]) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in selected:
                images[i] = frame.to_image().resize((480, 270))
    assert set(images) == set(selected)
    sheet = Image.new("RGB", (1920, 930), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    shown = []
    for n, i in enumerate(selected):
        x, y = (n % 4) * 480, (n // 4) * 310
        draw.text((x + 4, y + 3), f"{clip_id} frame {i}  {ts[i]/1e6:.3f}s", fill="black")
        sheet.paste(images[i], (x, y + 20))
        counts = [int(usable[i, s].sum()) for s in (slice(33, 54), slice(54, 75))]
        observed = [int(seen[i, s].sum()) for s in (slice(33, 54), slice(54, 75))]
        score = [float(v) if np.isfinite(v) else None for v in scores[i]]
        draw.text((x + 4, y + 292), f"usable L/R {counts[0]}/21 {counts[1]}/21; observed {observed}", fill="black")
        shown.append({"frame": i, "time_s": float(ts[i]/1e6), "usable_joint_counts_left_right": counts,
                      "observed_joint_counts_left_right": observed, "hand_scores": score})
    sheet.save(OUT / (clip_id + "_missingness.jpg"), quality=93)
    report["clips"].append({"clip_id": clip_id, "all_frame_observation_missingness": per_hand, "displayed_frames": shown})
(OUT / "missingness_frame_evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"status": "FRAME_EVIDENCE_SAVED", "clips": len(report["clips"]), "new_feature_extraction_runs": 0}))
