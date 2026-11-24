"""Render cached observations over original video extent; no detector/model runs.

Fixed prior issue frames, fixed endpoint audits, and deterministically selected
new-crop palm5 regressions are all included. This is visual evidence, not joint
ground truth. Only this unit's visual_review/ and VISUAL_REVIEW.json are written.
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
OUT = U / "visual_review"
PALM = np.array([0, 5, 9, 13, 17])
BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24]
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
              (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
              (0, 17), (17, 18), (18, 19), (19, 20)]
COLORS = ["#00ff99", "#e45bff"]
FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 22)
SMALL = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)
POINT_FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 26)


def read(p):
    return json.loads(Path(p).read_text("utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def hand_stats(ob, frame):
    result = {}
    for side, label in enumerate(("LEFT", "RIGHT")):
        sl = slice(33 + side * 21, 54 + side * 21)
        v = ob["usable"][frame, sl]
        result[label] = {"observed_points": int(ob["observed"][frame, sl].sum()),
                         "usable_points": int(v.sum()), "palm5": bool(v[PALM].all())}
    result["body_usable_9"] = int(ob["usable"][frame, BODY].sum())
    return result


def stats_text(ob, frame):
    s = hand_stats(ob, frame)
    return (f'Body {s["body_usable_9"]}/9 | L {s["LEFT"]["usable_points"]}/21 '
            f'P5={int(s["LEFT"]["palm5"])} | R {s["RIGHT"]["usable_points"]}/21 '
            f'P5={int(s["RIGHT"]["palm5"])}')


def panel(source, ob, frame, box, sensor):
    # Whole original image remains geometrically visible, including excluded RGB.
    im = source.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    w, h = im.size
    for r in [(0, 0, w, y0), (0, y1, w, h), (0, y0, x0, y1), (x1, y0, w, y1)]:
        if r[2] > r[0] and r[3] > r[1]:
            ld.rectangle(r, fill=(105, 105, 105, 205 if sensor else 145))
    im = Image.alpha_composite(im, layer).convert("RGB")
    d = ImageDraw.Draw(im)
    xy, valid = ob["xy_px"][frame], ob["usable"][frame]
    for a, b in BODY_EDGES:
        if valid[a] and valid[b]:
            d.line([tuple(xy[a]), tuple(xy[b])], fill="#ffc44c", width=6)
    for j in BODY:
        if valid[j]:
            x, y = map(float, xy[j])
            d.rectangle((x-7, y-7, x+7, y+7), fill="#ffc44c", outline="black", width=2)
    for side, color in enumerate(COLORS):
        base = 33 + side * 21
        for a, b in HAND_EDGES:
            if valid[base+a] and valid[base+b]:
                d.line([tuple(xy[base+a]), tuple(xy[base+b])], fill=color, width=5)
        for j in range(21):
            if valid[base+j]:
                x, y = map(float, xy[base+j])
                d.ellipse((x-6, y-6, x+6, y+6), fill=color, outline="black", width=2)
        if valid[base]:
            x, y = map(float, xy[base])
            d.text((x+13, y-32), ["LEFT", "RIGHT"][side], font=POINT_FONT,
                   fill=color, stroke_width=2, stroke_fill="black")
    d.rectangle((2, 2, w-3, h-3), outline="#a5bbc6", width=5)
    d.rectangle((x0+1, y0+1, x1-2, y1-2), outline="#ffe800", width=6)
    if sensor:
        d.rectangle((x0+8, y0+8, x1-9, y1-9), outline="#ff7a32", width=5)
    return im


def decode(video, indices, expected_times):
    need, frames = set(indices), {}
    with av.open(video) as container:
        for f, vf in enumerate(container.decode(video=0)):
            if f in need:
                timestamp_us = int(round(float(vf.pts * vf.time_base) * 1_000_000))
                assert abs(timestamp_us-int(expected_times[f])) <= 1, (video, f, timestamp_us, expected_times[f])
                frames[f] = vf.to_image().convert("RGB")
            if f >= max(need):
                break
    assert set(frames) == need
    return frames


def main():
    assert not (U / "VISUAL_REVIEW.json").exists(), "Reuse completed review"
    plan, comparison = read(U / "PLAN.json"), read(U / "COMPARISON.json")
    items = {it["clip_id"]: it for it in plan["items"]}
    observations, paths = {}, {}
    for cid, it in items.items():
        paths[cid] = dict(it["old_models"])
        paths[cid].update({m: str(U / "outputs" / cid / m / "model_input.npz")
                           for m in ("wrist_full", "wrist_sensor")})
        observations[cid] = {m: load(p) for m, p in paths[cid].items()}
        reference = observations[cid]["wrist_full"]
        for ob in observations[cid].values():
            assert np.array_equal(ob["frame_id"], reference["frame_id"])
            assert np.array_equal(ob["timestamp_us"], reference["timestamp_us"])
        assert np.array_equal(reference["frame_id"], np.arange(len(reference["frame_id"])))

    targets = []
    fixed = [("NIA_SL_WORD0106_REAL01_F", 52), ("NIA_SL_WORD0106_REAL01_F", 53),
             ("NIA_SL_WORD1139_REAL01_F", 56), ("NIA_SL_WORD1139_REAL01_F", 71),
             ("NIA_SL_WORD1545_REAL01_F", 36), ("NIA_SL_WORD0651_REAL03_F", 57),
             ("NIA_SL_WORD0766_REAL03_F", 46), ("NIA_SL_WORD2721_REAL02_F", 43)]
    for cid, frame in fixed:
        n = len(observations[cid]["wrist_full"]["frame_id"])
        targets.append(dict(clip_id=cid, target_frame=frame,
                            frames=[f for f in (frame-1, frame, frame+1) if 0 <= f < n],
                            reason="fixed_prior_problem"))

    # Explicit trace-driven sustained-loss checks, supplementing ranked first loss.
    for cid, frame in [("NIA_SL_WORD1545_REAL01_F", 55),
                       ("NIA_SL_WORD2348_REAL02_F", 49),
                       ("NIA_SL_WORD2348_REAL02_F", 68)]:
        targets.append(dict(clip_id=cid, target_frame=frame,
                            frames=[frame-1, frame, frame+1],
                            reason="trace_sustained_new_sensor_loss"))

    # Selection rule uses all20 clips, not labels or manually favorable outcomes.
    ranking = []
    for row in comparison["rows"]:
        pair = row["pairs"]["sensor_boundary"]
        loss = sum(pair[s]["palm5"]["loss"] for s in ("left", "right"))
        gain = sum(pair[s]["palm5"]["gain"] for s in ("left", "right"))
        if loss:
            ranking.append(dict(clip_id=row["clip_id"], gross_loss=loss, gain=gain, net_loss=loss-gain))
    ranking.sort(key=lambda r: (-r["net_loss"], -r["gross_loss"], r["clip_id"]))
    adverse_selection = []
    for rank in ranking[:3]:
        cid = rank["clip_id"]
        old, new = (observations[cid][m] for m in ("old_sensor", "wrist_sensor"))
        events = []
        for side in range(2):
            slots = 33 + side * 21 + PALM
            events.extend((int(f), side) for f in np.flatnonzero(
                old["usable"][:, slots].all(1) & ~new["usable"][:, slots].all(1)))
        frame, side = min(events)
        adverse_selection.append(dict(rank, frame=frame, side=["LEFT", "RIGHT"][side]))
        n = len(new["frame_id"])
        targets.append(dict(clip_id=cid, target_frame=frame,
                            frames=[f for f in (frame-1, frame, frame+1) if 0 <= f < n],
                            reason="ranked_new_sensor_palm5_loss"))

    for cid, endpoint in [("NIA_SL_WORD1133_REAL01_F", "first5"),
                          ("NIA_SL_WORD1844_REAL02_F", "last5"),
                          ("NIA_SL_WORD0106_REAL01_F", "first_last"),
                          ("NIA_SL_WORD1545_REAL01_F", "first_last")]:
        n = len(observations[cid]["wrist_full"]["frame_id"])
        frames = list(range(5)) if endpoint == "first5" else list(range(n-5, n)) if endpoint == "last5" else [0, n-1]
        targets.append(dict(clip_id=cid, target_frame=frames[0], frames=frames,
                            reason="endpoint_"+endpoint))

    OUT.mkdir(exist_ok=True)
    index, verified_sources = [], {}
    for target in targets:
        cid, indices = target["clip_id"], target["frames"]
        it, obs = items[cid], observations[cid]
        if cid not in verified_sources:
            verified_sources[cid] = sha(it["video_path"])
            assert verified_sources[cid] == it["source_sha256"]
        frames = decode(it["video_path"], indices, obs["wrist_full"]["timestamp_us"])
        assert all(list(im.size) == it["original_size"] for im in frames.values())
        sheet = Image.new("RGB", (2640, 160 + len(indices) * 445), "#f7f9fc")
        d = ImageDraw.Draw(sheet)
        d.text((18, 10), f'{it["label"]} | {cid} | {target["reason"]} | old y<{it["bottom_box"][3]} / wrist y<{it["wrist_box"][3]}', font=FONT, fill="#102138")
        d.text((18, 46), "Original 1920x1080 extent in every panel. Green=anatomical LEFT; Purple=RIGHT; Gold squares=body. Yellow=model ROI; Orange=sensor crop.", font=SMALL, fill="#102138")
        d.text((18, 73), "Gray region is excluded from model. FULL detector saw it; SENSOR detector never received it. Shading is display only; no zoom or extra crop.", font=SMALL, fill="#102138")
        d.text((18, 100), "Drawn points are usable model inputs. P5=all 5 palm anchors. Counts do not certify correctness; missing points are not automatically outside-screen events.", font=SMALL, fill="#102138")
        d.text((18, 127), "Source footer reports OLD FULL counts for numeric reference only. Expanded-boundary FULL uses the same cached whole-image detector as OLD FULL.", font=SMALL, fill="#102138")
        frame_rows = []
        for row_number, frame in enumerate(indices):
            y = 160 + row_number * 445
            modes = [None, "old_sensor", "wrist_full", "wrist_sensor"]
            titles = ["Original RGB", "OLD sensor crop", "WRIST full detector + mask", "WRIST sensor crop + mask"]
            for col, (mode, title) in enumerate(zip(modes, titles)):
                x = col * 660
                d.text((x+12, y), f'{title} | frame {frame}', font=SMALL, fill="#102138")
                if mode is None:
                    im = frames[frame].copy()
                else:
                    box = it["bottom_box"] if mode == "old_sensor" else it["wrist_box"]
                    im = panel(frames[frame], obs[mode], frame, box, mode != "wrist_full")
                im.thumbnail((650, 366), Image.Resampling.LANCZOS)
                sheet.paste(im, (x+5+(650-im.width)//2, y+30+(366-im.height)//2))
                text = stats_text(obs[mode], frame) if mode else "OLD FULL: " + stats_text(obs["old_full"], frame)
                d.text((x+8, y+401), text, font=SMALL, fill="#102138")
                d.text((x+8, y+423), f't={int(obs["wrist_full"]["timestamp_us"][frame])/1e6:.6f}s', font=SMALL, fill="#102138")
            frame_rows.append(dict(frame=frame, timestamp_us=int(obs["wrist_full"]["timestamp_us"][frame]),
                                   modes={m: hand_stats(ob, frame) for m, ob in obs.items()}))
        path = OUT / f'{cid}_f{target["target_frame"]}_{target["reason"]}.png'
        assert not path.exists()
        sheet.save(path)
        index.append(dict(target, label=it["label"], old_box=it["bottom_box"], wrist_box=it["wrist_box"],
                          image=str(path), image_sha256=sha(path), source_video=it["video_path"],
                          source_sha256_verified=verified_sources[cid],
                          observation_pins={str(p): sha(p) for p in paths[cid].values()}, frame_rows=frame_rows))
    result = dict(status="RENDERED_PENDING_VISUAL_INSPECTION", reviewer="wrist_visual_review",
                  images=len(index), fixed_problem_frames=len(fixed), adverse_selection=adverse_selection,
                  adverse_ranking_all=ranking,
                  selection="Eight prior fixed problem targets +/-1; trace-requested sustained-loss swimming55/vanish49/68 +/-1; three loss-bearing clips sorted by net new-sensor palm5 loss descending, gross loss descending, clip_id; earliest loss event +/-1; fixed endpoint windows for well-done, Hongseong, coldcoffee, swimming.",
                  layout="Original whole RGB extent in every panel; old sensor, wrist full, wrist sensor usable joints. Excluded pixels shaded for explanation only.",
                  plan_sha256=sha(U / "PLAN.json"), comparison_sha256=sha(U / "COMPARISON.json"), script_sha256=sha(__file__),
                  new_detector_runs=0, classifier_inference=0, training=0, rows=index,
                  findings=[], limitations=["No joint ground truth or annotation of all differing frames.",
                  "Endpoint windows are not automatically resting posture, and interior frames are not labelled active gestures.",
                  "Original source pixels below the sensor boundary are displayed for review but were unavailable to that detector.",
                  "Observed point counts and visual review do not measure recognition accuracy."])
    with (U / "VISUAL_REVIEW.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"images": len(index), "adverse_selection": adverse_selection}, ensure_ascii=False))


def raw_audit():
    """Supplement the model-input views with cached raw candidate geometry."""
    result = read(U / "VISUAL_REVIEW.json")
    assert "raw_audit_rows" not in result
    items = {it["clip_id"]: it for it in read(U / "PLAN.json")["items"]}
    rows = []
    for cid, indices in [("NIA_SL_WORD1545_REAL01_F", [55]),
                         ("NIA_SL_WORD2348_REAL02_F", [49, 68])]:
        it = items[cid]
        raw_paths = [Path(it["old_sensor_raw"]), U / "sensor_crop" / cid / "raw.npz"]
        raws = [load(p) for p in raw_paths]
        frames = decode(it["video_path"], indices, raws[1]["timestamp_us"])
        sheet = Image.new("RGB", (1980, 135+len(indices)*450), "#f7f9fc")
        d = ImageDraw.Draw(sheet)
        d.text((15, 10), f'{it["label"]} | {cid} | RAW candidates before assignment / no new detection', font=FONT, fill="#102138")
        d.text((15, 46), "Whole original extent. C0=cyan, C1=pink are candidate indices, NOT anatomical sides. Raw label is a detector output, not truth.", font=SMALL, fill="#102138")
        d.text((15, 73), "Yellow horizontal line=cropped RGB boundary. Raw finite hand points shown even if assignment later rejects them.", font=SMALL, fill="#102138")
        d.text((15, 100), "Compare each candidate to the source hand: two candidates over one hand do not supply evidence of the other hand.", font=SMALL, fill="#102138")
        evidence = []
        for row_number, frame in enumerate(indices):
            y = 135+row_number*450
            event = dict(frame=frame, timestamp_us=int(raws[1]["timestamp_us"][frame]), conditions={})
            for col in range(3):
                x = col*660
                im = frames[frame].copy()
                title = "Original RGB"
                summary = ""
                if col:
                    raw = raws[col-1]
                    title = ["OLD sensor RAW", "WRIST sensor RAW"][col-1]
                    draw = ImageDraw.Draw(im)
                    box = it["bottom_box"] if col == 1 else it["wrist_box"]
                    draw.line((0, box[3], im.width, box[3]), fill="#ffe800", width=6)
                    candidates = []
                    for candidate in range(int(raw["raw_count"][frame])):
                        xy = raw["hand_xy_px"][frame, candidate]
                        color = ["#00ffff", "#ff55df"][candidate]
                        valid = np.isfinite(xy).all(1)
                        for a, b in HAND_EDGES:
                            if valid[a] and valid[b]:
                                draw.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=5)
                        for j in np.flatnonzero(valid):
                            px, py = map(float, xy[j])
                            draw.ellipse((px-6, py-6, px+6, py+6), fill=color, outline="black", width=2)
                        if valid[0]:
                            px, py = map(float, xy[0])
                            draw.text((px+10, py-35+candidate*35), f'C{candidate}', font=POINT_FONT,
                                      fill=color, stroke_width=2, stroke_fill="black")
                        candidates.append(dict(index=candidate, label=str(raw["raw_label_name"][frame,candidate]),
                                               score=float(raw["raw_score"][frame,candidate]),
                                               wrist_xy=[float(v) for v in xy[0]]))
                    summary = " | ".join(f'C{c["index"]} {c["label"]} {c["score"]:.3f}' for c in candidates)
                    event["conditions"][["old_sensor", "wrist_sensor"][col-1]] = dict(
                        raw_count=int(raw["raw_count"][frame]), candidates=candidates)
                d.text((x+12,y), f'{title} | frame {frame}', font=SMALL, fill="#102138")
                im.thumbnail((650,366), Image.Resampling.LANCZOS)
                sheet.paste(im,(x+5,y+30))
                d.text((x+8,y+401), summary, font=SMALL, fill="#102138")
                d.text((x+8,y+426), f't={event["timestamp_us"]/1e6:.6f}s', font=SMALL, fill="#102138")
            evidence.append(event)
        path = OUT / f'{cid}_raw_candidate_audit.png'
        assert not path.exists()
        sheet.save(path)
        rows.append(dict(clip_id=cid,label=it["label"],image=str(path),image_sha256=sha(path),
                         raw_pins={str(p):sha(p) for p in raw_paths},frames=evidence))
    result["raw_audit_rows"] = rows
    result["images"] = len(result["rows"])+len(rows)
    result["script_sha256"] = sha(__file__)
    (U / "VISUAL_REVIEW.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"raw_audit_images":len(rows),"total_images":result["images"]}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--raw-audit"]:
        raw_audit()
    else:
        main()
