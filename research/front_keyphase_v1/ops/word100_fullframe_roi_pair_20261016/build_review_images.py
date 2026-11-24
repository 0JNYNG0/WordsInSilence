"""Read-only visual audit of frozen paired observations. No model execution.

The full original image extent is preserved in every panel. Shading communicates
model exclusion; it is not an extra crop. Image selection is deterministic and
includes the two clips with largest net palm5 loss, not only favorable
examples. Point counts are observation evidence, not coordinate ground truth.
"""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
OUT = U / "review_images"
PALM = np.array([0, 5, 9, 13, 17])
BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24]
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
              (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
              (0, 17), (17, 18), (18, 19), (19, 20)]
COLORS = ["#00ffff", "#ff55df"]
FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 21)
SMALL = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)
POINT_FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 25)


def read(p):
    return json.loads(Path(p).read_text("utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(cid, mode):
    p = U / "outputs" / cid / mode / "model_input.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, p


def hand_stats(ob, frame):
    stats = {}
    for side, label in enumerate(("LEFT", "RIGHT")):
        sl = slice(33 + side * 21, 54 + side * 21)
        v = ob["usable"][frame, sl]
        stats[label] = {"observed_points": int(ob["observed"][frame, sl].sum()),
                        "usable_points": int(v.sum()), "palm5": bool(v[PALM].all())}
    stats["body_usable_9"] = int(ob["usable"][frame, BODY].sum())
    return stats


def stats_text(ob, frame):
    a = hand_stats(ob, frame)
    return (f'Body {a["body_usable_9"]}/9 | LEFT {a["LEFT"]["usable_points"]}/21 '
            f'P5={int(a["LEFT"]["palm5"])} | RIGHT {a["RIGHT"]["usable_points"]}/21 '
            f'P5={int(a["RIGHT"]["palm5"])}')


def shade_model_excluded(im, box, physical):
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    w, h = im.size
    alpha = 210 if physical else 145
    for rect in [(0, 0, w, y0), (0, y1, w, h), (0, y0, x0, y1), (x1, y0, w, y1)]:
        if rect[2] > rect[0] and rect[3] > rect[1]:
            d.rectangle(rect, fill=(105, 105, 105, alpha))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def panel(im, ob, frame, box, physical):
    im = shade_model_excluded(im, box, physical)
    d = ImageDraw.Draw(im)
    xy, valid = ob["xy_px"][frame], ob["usable"][frame]
    # Orange body squares; hand circles/edges distinguish anatomical sides.
    for a, b in BODY_EDGES:
        if valid[a] and valid[b]:
            d.line([tuple(xy[a]), tuple(xy[b])], fill="#ffbd43", width=6)
    for j in BODY:
        if valid[j]:
            x, y = map(float, xy[j])
            d.rectangle((x-7, y-7, x+7, y+7), fill="#ffbd43", outline="black", width=2)
    for side in range(2):
        base = 33 + side * 21
        color = COLORS[side]
        for a, b in HAND_EDGES:
            if valid[base+a] and valid[base+b]:
                d.line([tuple(xy[base+a]), tuple(xy[base+b])], fill=color, width=5)
        for j in range(21):
            i = base + j
            if valid[i]:
                x, y = map(float, xy[i])
                d.ellipse((x-6, y-6, x+6, y+6), fill=color, outline="black", width=2)
        if valid[base]:
            x, y = map(float, xy[base])
            d.text((x+13, y-32), ["LEFT", "RIGHT"][side], font=POINT_FONT,
                   fill=color, stroke_width=2, stroke_fill="black")
    w, h = im.size
    d.rectangle((2, 2, w-3, h-3), outline="#50e57c" if not physical else "#999999", width=6)
    x0, y0, x1, y1 = box
    d.rectangle((x0+1, y0+1, x1-2, y1-2), outline="#ffe800", width=6)
    if physical:
        d.rectangle((x0+8, y0+8, x1-9, y1-9), outline="#ff7a32", width=5)
    return im


def decode(video, indices):
    need = set(indices)
    frames = {}
    with av.open(video) as c:
        for f, vf in enumerate(c.decode(video=0)):
            if f in need:
                frames[f] = vf.to_image().convert("RGB")
            if f >= max(need):
                break
    assert set(frames) == need
    return frames


def main():
    assert not (U / "VISUAL_REVIEW_INDEX.json").exists(), "Reuse completed visuals"
    plan = read(U / "PLAN.json")
    cmp = read(U / "COMPARISON.json")
    items = {r["clip_id"]: r for r in plan["items"]}
    targets = [
        ("NIA_SL_WORD0651_REAL03_F", 57, "bottom", "fixed_sobok"),
        ("NIA_SL_WORD0766_REAL03_F", 46, "bottom", "fixed_reception"),
        ("NIA_SL_WORD2721_REAL02_F", 43, "bottom", "fixed_eighty_nine"),
        ("NIA_SL_WORD1139_REAL01_F", 56, "bottom", "fixed_believe_crop_frame"),
        ("NIA_SL_WORD1139_REAL01_F", 71, "bottom", "fixed_believe_native_frame"),
        ("NIA_SL_WORD0106_REAL01_F", 52, "bottom", "fixed_cold_coffee"),
    ]
    adverse = sorted(cmp["rows"], key=lambda r: (
        -sum(r["pairs"]["bottom"][s]["palm5"]["sensor_only"]
             - r["pairs"]["bottom"][s]["palm5"]["full_only"] for s in ("left", "right")),
        r["clip_id"]))[:2]
    adverse_selection = []
    for row in adverse:
        cid = row["clip_id"]
        full, _ = load(cid, "full_then_bottom_mask")
        sensor, _ = load(cid, "bottom_crop_then_mask")
        events = []
        for side in range(2):
            slots = 33 + side*21 + PALM
            a = full["usable"][:, slots].all(1)
            b = sensor["usable"][:, slots].all(1)
            events.extend((int(f), side) for f in np.flatnonzero(b & ~a))
        if not events:
            continue
        frame, side = min(events)
        targets.append((cid, frame, "bottom", "adverse_sensor_only_palm5"))
        adverse_selection.append({"clip_id": cid, "frame": frame, "side": ["LEFT", "RIGHT"][side],
            "sensor_only_palm5_framesides": sum(row["pairs"]["bottom"][s]["palm5"]["sensor_only"] for s in ("left", "right")),
            "net_full_loss_palm5_framesides": sum(row["pairs"]["bottom"][s]["palm5"]["sensor_only"]
                - row["pairs"]["bottom"][s]["palm5"]["full_only"] for s in ("left", "right"))})
    targets.extend((cid, f, "legacy", "legacy_fixed_problem") for cid, f in [
        ("NIA_SL_WORD0651_REAL03_F", 57), ("NIA_SL_WORD0766_REAL03_F", 46),
        ("NIA_SL_WORD2721_REAL02_F", 43)])
    OUT.mkdir(exist_ok=True)
    index = []
    for cid, target, pair, reason in targets:
        it = items[cid]
        names = ["full_then_"+pair+"_mask", pair+"_crop_then_mask"]
        obs_paths = [load(cid, name) for name in names]
        a, b = [r[0] for r in obs_paths]
        assert np.array_equal(a["frame_id"], b["frame_id"])
        assert np.array_equal(a["timestamp_us"], b["timestamp_us"])
        assert np.array_equal(a["frame_id"], np.arange(len(a["frame_id"])))
        indices = [f for f in (target-1, target, target+1) if 0 <= f < len(a["frame_id"])]
        box = it["bottom_box"] if pair == "bottom" else it["box"]
        frames = decode(it["video_path"], indices)
        assert all(list(im.size) == it["original_size"] for im in frames.values())
        sheet = Image.new("RGB", (1980, 150+len(indices)*440), "#f7f9fc")
        d = ImageDraw.Draw(sheet)
        d.text((18, 12), f'{it["label"]} / {cid} / {pair} ROI / target frame {target} / {reason}', font=FONT, fill="#102138")
        d.text((18, 45), "Orange squares=body | Cyan=anatomical LEFT | Magenta=RIGHT | Yellow=model ROI | Green/orange=detector extent", font=SMALL, fill="#102138")
        d.text((18, 72), "Gray pixels: excluded from model. FULL detector saw them; SENSOR detector never received them. Original extent preserved; no zoom.", font=SMALL, fill="#102138")
        d.text((18, 99), "Only usable model-input points are drawn. P5=all 5 palm anchors usable; counts do not certify correctness. Shading is for display only.", font=SMALL, fill="#102138")
        frame_rows = []
        for row, f in enumerate(indices):
            y = 150+row*440
            ims = [frames[f], panel(frames[f], a, f, box, False), panel(frames[f], b, f, box, True)]
            titles = ["Source / original camera image", "FULL detector -> ROI mask", "Physical SENSOR crop -> ROI mask"]
            for col, im in enumerate(ims):
                x = col*660
                d.text((x+12, y), f'{titles[col]} | frame {f}', font=SMALL, fill="#102138")
                im = im.copy()
                im.thumbnail((650, 366), Image.Resampling.LANCZOS)
                sheet.paste(im, (x+5+(650-im.width)//2, y+30+(366-im.height)//2))
                if col:
                    d.text((x+8, y+403), stats_text([a, b][col-1], f), font=SMALL, fill="#102138")
                else:
                    d.text((x+8, y+403), f't={int(a["timestamp_us"][f])/1e6:.3f}s | ROI={box}', font=SMALL, fill="#102138")
            frame_rows.append({"frame": f, "timestamp_us": int(a["timestamp_us"][f]),
                               names[0]: hand_stats(a, f), names[1]: hand_stats(b, f)})
        name = f'{pair}_{cid}_f{target}_{reason}.png'
        path = OUT / name
        assert not path.exists()
        sheet.save(path)
        index.append({"clip_id": cid, "label": it["label"], "target_frame": target,
            "pair": pair, "reason": reason, "roi": box, "source_video": it["video_path"],
            "source_sha256_expected": it["source_sha256"], "image": str(path), "image_sha256": sha(path),
            "observation_pins": {str(p): sha(p) for _, p in obs_paths}, "frames": frame_rows})
    result = {"status": "RENDERED_PENDING_HUMAN_VISUAL_REVIEW", "images": len(index),
        "selection": "Six fixed issue frames; two clips ranked by bottom net full loss palm5 framesides sensor_only minus full_only (descending, clip_id tie-break), earliest adverse frame; three legacy issue frames",
        "adverse_selection": adverse_selection, "plan_sha256": sha(U / "PLAN.json"),
        "comparison_sha256": sha(U / "COMPARISON.json"), "script_sha256": sha(__file__),
        "new_detector_runs": 0, "classifier_inference": 0, "training": 0, "rows": index}
    with (U / "VISUAL_REVIEW_INDEX.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"images": len(index), "adverse_selection": adverse_selection}, ensure_ascii=False))


if __name__ == "__main__":
    main()
