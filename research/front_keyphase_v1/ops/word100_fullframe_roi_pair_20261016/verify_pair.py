"""Independent read-only audit of frozen observation pairs (no model/detector run).

Expected masks, features and aggregates are calculated here without importing
the producer's comparison functions. Frozen joint-v3 is replayed only to verify
that every raw hand retained by a condition came from the identical policy.
"""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
U = Path(__file__).resolve().parent
LAB = U.parents[1]
J = U.with_name("word100_hand_assignment_joint_20261016")
sys.path[:0] = [str(LAB / "src"), str(U)]
from wis_lab.core.observation import Observation
from roi_adapter import apply_roi, build_roi_channels


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert np.array_equal(a, b, equal_nan=True), "array mismatch"


def pincheck(pins):
    for p, h in pins.items():
        assert sha(p) == h, p


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def mask_reference(source, box, origin):
    original = source.xy_px.astype(np.float64) + np.asarray(origin, np.float64)
    inside = np.isfinite(original).all(-1)
    inside &= (original[..., 0] >= box[0]) & (original[..., 0] < box[2])
    inside &= (original[..., 1] >= box[1]) & (original[..., 1] < box[3])
    observed = source.observed & inside
    usable = source.usable & observed
    xy = original.astype(np.float32)
    xy[~observed] = np.nan
    conf = source.conf.copy()
    conf[~observed] = np.nan
    score = source.hand_score.copy()
    for s in range(2):
        score[~observed[:, 33 + 21 * s:54 + 21 * s].any(1), s] = np.nan
    return xy, observed, usable, conf, score, inside


def independent_channels(ob):
    # Direct MP75 -> common51 mapping, avoiding producer's to_native_like and
    # build_channels. Missing points are excluded before computing all scales.
    ids = [0, 11, 12, 13, 14, 15, 16, 23, 24] + list(range(33, 75))
    xy = ob.xy_px[:, ids].copy()
    valid = ob.usable[:, ids]
    xy[~valid] = 0
    t = len(xy)
    shoulders = valid[:, 1] & valid[:, 2]
    assert shoulders.any()
    scale = float(np.median(np.linalg.norm(xy[shoulders, 1] - xy[shoulders, 2], axis=1)))
    assert np.isfinite(scale) and scale > 1e-6
    origin = (xy[:, 1] + xy[:, 2]) / 2
    gx = np.zeros((t, 51, 2), np.float32)
    gm = (valid & shoulders[:, None]).astype(np.float32)
    for f in range(t):
        for p in range(51):
            if gm[f, p]:
                gx[f, p] = (xy[f, p] - origin[f]) / scale
    lx = np.zeros((t, 42, 2), np.float32)
    lm = np.zeros((t, 42), np.float32)
    hscales = []
    for s, base in enumerate((9, 30)):
        lengths = []
        for f in range(t):
            if valid[f, base]:
                for mcp in (5, 9, 13, 17):
                    if valid[f, base + mcp]:
                        lengths.append(float(np.linalg.norm(xy[f, base + mcp] - xy[f, base])))
        hs = float(np.median(lengths)) if lengths else float("nan")
        hs = hs if hs > 1e-6 else float("nan")
        hscales.append(hs)
        if not np.isfinite(hs):
            continue
        for f in range(t):
            if valid[f, base]:
                for p in range(21):
                    if valid[f, base + p]:
                        lx[f, s * 21 + p] = (xy[f, base + p] - xy[f, base]) / hs
                        lm[f, s * 21 + p] = 1
    dt = np.zeros((t, 1), np.float32)
    dt[1:, 0] = np.diff(ob.timestamp_us).astype(np.float32) / 1e6
    x = np.concatenate([gx.reshape(t, -1), lx.reshape(t, -1), gm, lm, dt], axis=1)
    return x, scale, hscales


def flags(ob, side):
    u = ob.usable[:, 33 + side * 21:54 + side * 21]
    return dict(any=u.any(1), palm5=u[:, [0, 5, 9, 13, 17]].all(1), all21=u.all(1))


def metrics(ob):
    out = dict(frames=ob.n_frames, usable_body_points=int(ob.usable[:, :33].sum()),
               shoulders_both_frames=int(ob.usable[:, [11, 12]].all(1).sum()))
    for side, label in enumerate(("left", "right")):
        fs = flags(ob, side)
        row = {k: int(v.sum()) for k, v in fs.items()}
        row["usable_points"] = int(ob.usable[:, 33 + side * 21:54 + side * 21].sum())
        gaps = np.split(np.flatnonzero(~fs["any"]), np.flatnonzero(np.diff(np.flatnonzero(~fs["any"])) != 1) + 1)
        row["longest_no_hand_frames"] = max((len(x) for x in gaps), default=0)
        out[label] = row
    return out


def pair_reference(a, b):
    out = {}
    events = []
    for side, label in enumerate(("left", "right")):
        fa, fb = flags(a, side), flags(b, side)
        out[label] = {}
        for metric in fa:
            count = Counter(zip(fa[metric].tolist(), fb[metric].tolist()))
            out[label][metric] = {"both": count[(True, True)], "full_only": count[(True, False)],
                                  "sensor_only": count[(False, True)], "neither": count[(False, False)]}
        prev_on = -1
        for f in np.flatnonzero(fa["palm5"]):
            f = int(f)
            gap = f - prev_on - 1
            if gap >= 3:
                stop = min(f + 6, a.n_frames)
                sensor_on = np.flatnonzero(fb["palm5"][f:stop])
                events.append(dict(side=label, frame=f, timestamp_us=int(a.timestamp_us[f]),
                    preceding_reference_missing_frames=gap, window_frames=stop-f,
                    full_palm5_frames=int(fa["palm5"][f:stop].sum()),
                    sensor_palm5_frames=int(fb["palm5"][f:stop].sum()),
                    sensor_has_palm5_at_reference_entry=bool(fb["palm5"][f]),
                    sensor_first_palm5_offset_within_window=int(sensor_on[0]) if len(sensor_on) else None))
            prev_on = f
    out["reference_entry_events"] = events
    return out


def replay_check(raw_dir, candidate_path, assignment_path, cls, contract, expected_size):
    raw = npz(Path(raw_dir) / "raw.npz")
    baseline = Observation.load(str(Path(raw_dir) / "baseline.npz"))
    candidate = Observation.load(str(candidate_path))
    stored = npz(assignment_path)["sides"]
    nframes = baseline.n_frames
    assert nframes == candidate.n_frames
    equal(raw["frame_id"], baseline.frame_id)
    equal(raw["timestamp_us"], baseline.timestamp_us)
    equal(candidate.timestamp_us, baseline.timestamp_us)
    assert (np.diff(raw["detector_timestamp_ms"]) > 0).all()
    assert (raw["width"] == expected_size[0]).all() and (raw["height"] == expected_size[1]).all()
    for src, dst in (("pose_xy_px", "xy_px"), ("pose_usable", "usable"), ("pose_observed", "observed"), ("pose_conf", "conf")):
        equal(raw[src], getattr(baseline, dst)[:, :33])
        equal(getattr(candidate, dst)[:, :33], getattr(baseline, dst)[:, :33])
    expected_xy = np.full((nframes, 42, 2), np.nan, np.float32)
    expected_observed = np.zeros((nframes, 42), bool)
    expected_usable = expected_observed.copy()
    expected_score = np.full((nframes, 2), np.nan, np.float32)
    sides = np.full((nframes, 2), -1, np.int8)
    ass = cls()
    for f in range(nframes):
        n = int(raw["raw_count"][f])
        w, h = int(raw["width"][f]), int(raw["height"][f])
        assigned, _ = ass.assign(raw["hand_xy_px"][f, :n], raw["raw_label"][f, :n], raw["raw_score"][f, :n],
            raw["pose_xy_px"][f], raw["pose_usable"][f], int(raw["timestamp_us"][f]), w, h)
        for i, side in assigned.items():
            if side is None:
                continue
            sides[f, i] = side
            pts = raw["hand_xy_px"][f, i].astype(np.float32)
            sc = raw["raw_score"][f, i]
            sl = slice(side * 21, side * 21 + 21)
            expected_xy[f, sl] = pts
            expected_observed[f, sl] = True
            mx, my = w * contract["usable_rule"]["bounds_margin_ratio"], h * contract["usable_rule"]["bounds_margin_ratio"]
            expected_usable[f, sl] = np.isfinite(pts).all(1) & (pts[:, 0] >= -mx) & (pts[:, 0] <= w + mx) & (pts[:, 1] >= -my) & (pts[:, 1] <= h + my) & (sc >= contract["usable_rule"]["hand_score_threshold"])
            expected_score[f, side] = sc
    equal(sides, stored)
    equal(expected_xy, candidate.xy_px[:, 33:])
    equal(expected_observed, candidate.observed[:, 33:])
    equal(expected_usable, candidate.usable[:, 33:])
    equal(expected_score, candidate.hand_score)
    return nframes, int(raw["raw_count"].sum())


def main():
    outpath = U / "INDEPENDENT_VERIFICATION.json"
    assert not outpath.exists(), "Reuse independent completed report"
    started = time.perf_counter()
    plan = read(U / "PLAN.json")
    capture = read(U / "CAPTURE_RESULT.json")
    result = read(U / "COMPARISON.json")
    context = read(U / "COMPARISON_CONTEXT.json")
    pincheck(plan["pins"])
    pincheck(context["code"])
    assert context["plan_sha256"] == sha(U / "PLAN.json")
    assert context["capture_sha256"] == sha(U / "CAPTURE_RESULT.json")
    assert len(plan["items"]) == 20 and len(capture["rows"]) == 20
    assert len({r["detector_instance"] for r in capture["rows"]}) == 20
    for row in capture["rows"]:
        pincheck(row["outputs"])
        assert row["plan_sha256"] == sha(U / "PLAN.json")
    spec = importlib.util.spec_from_file_location("independent_fixed_joint_v3", J / "assignment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = read(LAB / "config/contract.json")
    conditions = frames = raw_hands = masked_points = 0
    aggregates = {"legacy": Counter(), "bottom": Counter()}
    entries = {"legacy": [], "bottom": []}
    auditrows = []
    for item, saved in zip(plan["items"], result["rows"]):
        cid = item["clip_id"]
        assert cid == saved["clip_id"]
        assert item["source_role"] == "train_candidate" and item["signer_id"] in (1, 2, 3)
        assert item["bottom_box"] == [0, 0, item["original_size"][0], item["box"][3]]
        assert item["box"][1] == 0 and 0 < item["box"][3] <= item["original_size"][1]
        pincheck(saved["outputs"])
        native = Observation.load(item["cached"]["native"]["observation"])
        oldcrop = Observation.load(item["cached"]["crop"]["observation"])
        sd = U / "sensor_crop" / cid
        newcrop = Observation.load(str(sd / "candidate.npz"))
        for mode, expected_size in (("native", item["original_size"]), ("crop", [item["box"][2]-item["box"][0], item["box"][3]-item["box"][1]])):
            cache = item["cached"][mode]
            nf, nr = replay_check(cache["raw_dir"], cache["observation"], cache["assignment"], module.HandAssigner, contract, expected_size)
            raw_hands += nr
        nf, nr = replay_check(sd, sd / "candidate.npz", sd / "assignment.npz", module.HandAssigner, contract, item["bottom_box"][2:])
        raw_hands += nr
        for ob in (oldcrop, newcrop):
            equal(ob.frame_id, native.frame_id)
            equal(ob.timestamp_us, native.timestamp_us)
        variants = {
            "full_then_legacy_mask": (native, item["box"], (0, 0)),
            "legacy_crop_then_mask": (oldcrop, item["box"], item["box"][:2]),
            "full_then_bottom_mask": (native, item["bottom_box"], (0, 0)),
            "bottom_crop_then_mask": (newcrop, item["bottom_box"], (0, 0)),
        }
        obs = {}
        for key, (source, box, origin) in variants.items():
            d = U / "outputs" / cid / key
            ob = Observation.load(str(d / "model_input.npz"))
            expected = mask_reference(source, box, origin)
            for attribute, exp in zip(("xy_px", "observed", "usable", "conf", "hand_score"), expected[:5]):
                equal(getattr(ob, attribute), exp)
            equal(ob.frame_id, source.frame_id)
            equal(ob.timestamp_us, source.timestamp_us)
            ch = npz(d / "channels.npz")
            x, bs, hs = independent_channels(ob)
            equal(x, ch["x"])
            equal(ch["frame_id"], source.frame_id)
            equal(ch["timestamp_us"], source.timestamp_us)
            assert ch["x"].shape == (source.n_frames, 280)
            assert metrics(ob) == saved["metrics"][key]
            # Arbitrary values outside the ROI cannot affect channels/scales.
            # Only detector/association *outputs* are mutated: this deliberately
            # does not pretend tracker context is independent of outside pixels.
            changed = copy.deepcopy(source)
            hidden = ~expected[5] & source.observed
            changed.xy_px[hidden] = np.array([-100000, -200000], np.float32) - np.array(origin, np.float32)
            changed.conf[hidden] = .123
            poisoned = build_roi_channels(apply_roi(changed, box, origin))
            equal(poisoned["x"], ch["x"])
            assert poisoned["body_scale"] == bs
            np.testing.assert_array_equal(np.array(list(poisoned["hand_scales"].values())), np.array(hs))
            masked_points += int(hidden.sum())
            frames += source.n_frames
            conditions += 1
            obs[key] = ob
        for kind in ("legacy", "bottom"):
            pair = pair_reference(obs["full_then_"+kind+"_mask"], obs[kind+"_crop_then_mask"])
            assert pair == saved["pairs"][kind]
            for side in ("left", "right"):
                for metric, counts in pair[side].items():
                    for flag, count in counts.items():
                        aggregates[kind][metric+"_"+flag] += count
            entries[kind].extend(dict(clip_id=cid, label=item["label"], **row) for row in pair["reference_entry_events"])
        auditrows.append(dict(clip_id=cid, frames=native.n_frames, conditions=4, passed=True))
        print(json.dumps(dict(verified_clip=cid, conditions=conditions), ensure_ascii=False), flush=True)
    assert {k: dict(v) for k, v in aggregates.items()} == result["aggregate"]
    assert entries == result["entry_events"]
    assert conditions == result["conditions"] == 80 and frames == result["condition_frames"]
    assert sum(r["frames"] for r in auditrows) == result["unique_video_frames"] == capture["frames"]
    assert result["training"] == result["classifier_inference"] == result["production_changes"] == 0
    pincheck(plan["pins"])
    final = dict(status="PASS", reviewer="roi_pair_verify", independent_of="producer mask/feature/metric formulae; frozen assignment replay intentionally uses same fixed policy",
        conditions=conditions, clips=20, condition_frames=frames, source_detector_conditions_replayed=60,
        raw_hands_replayed=raw_hands, hidden_source_points_mutated=masked_points,
        checks=["all pinned sources and frozen model unchanged", "same raw assignment replays in 60 conditions", "original frame IDs and timestamps exact", "strict all-75-point ROI masks exact", "no fabricated coordinate after mask", "all 80 input tensors independently recalculated exactly", "outside-coordinate perturbations cannot alter features or normalization", "partial hands and missing wrist local masks follow input contract", "sameframe gains/losses and entry proxy events independently recomputed", "new bottom crops have correct sensor bounds and unique detector instances"],
        limitations=["No hand-coordinate ground truth; observation counts are not recognition accuracy", "Entry events use fullframe detected palm visibility proxy, not human-labeled physical entry", "Fullframe detector context may use below-ROI pixels and history; model-input mask does not remove this detector-context advantage", "Feature normalization uses completed-clip medians; no live streaming causality claim", "No real webcam, classifier, training, or production changes"],
        rows=auditrows, seconds=time.perf_counter()-started,
        inputs={str(U/name):sha(U/name) for name in ("PLAN.json", "CAPTURE_RESULT.json", "COMPARISON_CONTEXT.json", "COMPARISON.json", "verify_pair.py")})
    with outpath.open("x", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: final[k] for k in ("status", "conditions", "condition_frames", "raw_hands_replayed", "hidden_source_points_mutated")}), flush=True)


if __name__ == "__main__":
    main()
