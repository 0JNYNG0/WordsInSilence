"""Independent audit of the 20-clip resting-wrist ROI pair (no model/detector run).

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
PREVIOUS = U.with_name("word100_fullframe_roi_pair_20261016")
sys.path[:0] = [str(LAB / "src"), str(PREVIOUS)]
from wis_lab.core.observation import Observation
from roi_adapter import apply_roi, build_roi_channels
# Independent formulas below are retained from the prior reviewer, not producer
# comparison code. Producer adapter is called only for outside-value probes.


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


def boundary_reference(native, original_size, old_box):
    """Calculate the proposed boundary using only the fixed first five frames."""
    assert native.n_frames >= 5
    required = np.array([11, 12, 33, 54])
    assert native.usable[:5, required].all(), "initial calibration point missing"
    first = native.xy_px[:5]
    wrist_medians = np.median(first[:, [33, 54], 1], axis=0)
    shoulder_width = float(np.median(np.linalg.norm(first[:, 11] - first[:, 12], axis=1)))
    margin = .02 * shoulder_width
    bottom = int(np.floor(float(min(wrist_medians)) - margin))
    width, height = original_size
    assert old_box[:3] == [0, 0, width]
    assert old_box[3] < bottom < height
    return dict(box=[0, 0, width, bottom], first_frames=list(range(5)),
                wrist_median_y=wrist_medians.tolist(), shoulder_width_px=shoulder_width,
                margin_px=margin)


def endpoint_reference(ob):
    """Report endpoints and five-frame windows without assuming they are empty."""
    result = {}
    for name, indices in (("first_frame", [0]), ("first5", np.arange(5)),
                          ("last5", np.arange(ob.n_frames - 5, ob.n_frames)),
                          ("last_frame", [ob.n_frames - 1])):
        pointmask = ob.usable[indices, 33:75].reshape(-1, 2, 21)
        result[name] = dict(usable_hand_points=int(pointmask.sum()),
                            any_hand_frame_sides=int(pointmask.any(axis=2).sum()),
                            palm5_frame_sides=int(pointmask[:, :, [0, 5, 9, 13, 17]].all(axis=2).sum()),
                            all21_frame_sides=int(pointmask.all(axis=2).sum()))
    return result


def before_after_reference(before, after):
    result = {}
    for side, name in enumerate(("left", "right")):
        old, new = flags(before, side), flags(after, side)
        result[name] = {}
        for metric in old:
            result[name][metric] = dict(
                both=int(np.count_nonzero(old[metric] & new[metric])),
                old_only=int(np.count_nonzero(old[metric] & ~new[metric])),
                new_only=int(np.count_nonzero(~old[metric] & new[metric])),
                neither=int(np.count_nonzero(~old[metric] & ~new[metric])))
    return result


def extra_metrics_reference(ob):
    out = metrics(ob)
    t = ob.timestamp_us
    selections = {"first5": list(range(5)), "last5": list(range(ob.n_frames - 5, ob.n_frames)),
                  "first_half_second": [i for i in range(ob.n_frames) if int(t[i] - t[0]) <= 500000],
                  "last_half_second": [i for i in range(ob.n_frames) if int(t[-1] - t[i]) <= 500000]}
    out["end_windows"] = {}
    for name, frames in selections.items():
        hands = {}
        for side, label in enumerate(("left", "right")):
            point_counts = [int(ob.usable[f, 33 + side * 21:54 + side * 21].sum()) for f in frames]
            hands[label] = dict(points=sum(point_counts), any_frames=sum(x > 0 for x in point_counts),
                                all21_frames=sum(x == 21 for x in point_counts))
        out["end_windows"][name] = dict(frames=frames, hands=hands)
    return out


def diff_reference(before, after):
    equal(before.frame_id, after.frame_id)
    equal(before.timestamp_us, after.timestamp_us)
    out = {}
    for side, label in enumerate(("left", "right")):
        old, new = flags(before, side), flags(after, side)
        row = {}
        for key in old:
            old_set = set(np.flatnonzero(old[key]).tolist())
            new_set = set(np.flatnonzero(new[key]).tolist())
            gain_frames, loss_frames = sorted(new_set - old_set), sorted(old_set - new_set)
            row[key] = dict(old=len(old_set), new=len(new_set), gain=len(gain_frames), loss=len(loss_frames),
                            gain_frames=gain_frames, loss_frames=loss_frames)
        sl = slice(33 + side * 21, 54 + side * 21)
        old_points = set(zip(*np.where(before.usable[:, sl])))
        new_points = set(zip(*np.where(after.usable[:, sl])))
        row["points"] = dict(old=len(old_points), new=len(new_points),
                             gain=len(new_points - old_points), loss=len(old_points - new_points))
        out[label] = row
    return out


def main():
    started = time.perf_counter()
    outpath = U / "INDEPENDENT_VERIFICATION.json"
    assert not outpath.exists(), "Reuse the completed independent report"
    plan = read(U / "PLAN.json")
    capture = read(U / "CAPTURE_RESULT.json")
    result = read(U / "COMPARISON.json")
    previous = read(PREVIOUS / "PLAN.json")
    pincheck(plan["pins"])
    assert plan["conditions"] == ["old_full", "old_sensor", "wrist_full", "wrist_sensor"]
    assert len(plan["items"]) == len(capture["rows"]) == len(result["rows"]) == 20
    assert [it["clip_id"] for it in plan["items"]] == [it["clip_id"] for it in previous["items"]]
    assert len(set(r["detector_instance"] for r in capture["rows"])) == 20
    assert capture["conditions"] == plan["new_detector_conditions"] == 20
    captures = {r["clip_id"]: r for r in capture["rows"]}
    assert len(captures) == 20
    for row in capture["rows"]:
        pincheck(row["outputs"])
        assert row["plan_sha256"] == sha(U / "PLAN.json")
    spec = importlib.util.spec_from_file_location("independent_wrist_fixed_joint_v3", J / "assignment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = read(LAB / "config/contract.json")
    conditions = frames = raw_hands = hidden_points = 0
    auditrows, recomputed = [], []
    pairs = {"full_boundary": ("old_full", "wrist_full"),
             "sensor_boundary": ("old_sensor", "wrist_sensor"),
             "wrist_source": ("wrist_full", "wrist_sensor")}
    for item, old_item, saved in zip(plan["items"], previous["items"], result["rows"]):
        cid = item["clip_id"]
        assert cid == saved["clip_id"]
        assert item["source_role"] == "train_candidate" and item["signer_id"] in (1, 2, 3)
        for key in ("clip_id", "label", "signer_id", "source_role", "video_path", "source_sha256", "original_size", "box", "bottom_box", "cached"):
            assert item[key] == old_item[key], (cid, key)
        assert item["old_models"] == {
            "old_full": str(PREVIOUS / "outputs" / cid / "full_then_bottom_mask" / "model_input.npz"),
            "old_sensor": str(PREVIOUS / "outputs" / cid / "bottom_crop_then_mask" / "model_input.npz")}
        native_cache = item["cached"]["native"]
        native = Observation.load(native_cache["observation"])
        boundary = boundary_reference(native, item["original_size"], item["bottom_box"])
        assert boundary["box"] == item["wrist_box"]
        assert boundary["first_frames"] == item["calibration_frames"]
        assert boundary["wrist_median_y"] == item["wrist_median_y"]
        assert boundary["shoulder_width_px"] == item["shoulder_width_px"]
        assert np.ptp(native.xy_px[:5, [33, 54], 1], axis=0).tolist() == item["calibration_wrist_y_range"]
        assert captures[cid]["sensor_box"] == item["wrist_box"]
        assert captures[cid]["source_sha256"] == item["source_sha256"]
        old_dir = PREVIOUS / "sensor_crop" / cid
        sensor_dir = U / "sensor_crop" / cid
        assert item["old_sensor_raw"] == str(old_dir / "raw.npz")
        old_sensor = Observation.load(str(old_dir / "candidate.npz"))
        new_sensor = Observation.load(str(sensor_dir / "candidate.npz"))
        for raw_dir, candidate, assignment, size in (
            (native_cache["raw_dir"], native_cache["observation"], native_cache["assignment"], item["original_size"]),
            (old_dir, old_dir / "candidate.npz", old_dir / "assignment.npz", item["bottom_box"][2:]),
            (sensor_dir, sensor_dir / "candidate.npz", sensor_dir / "assignment.npz", item["wrist_box"][2:]),
        ):
            nf, nr = replay_check(raw_dir, candidate, assignment, module.HandAssigner, contract, size)
            assert nf == native.n_frames
            raw_hands += nr
        assert new_sensor.sidecar["source_bbox_px"] == item["wrist_box"]
        assert new_sensor.sidecar["detector_instance"] == captures[cid]["detector_instance"]
        for source in (old_sensor, new_sensor):
            equal(source.frame_id, native.frame_id)
            equal(source.timestamp_us, native.timestamp_us)
        pincheck(saved["outputs"])
        definitions = {
            "old_full": (native, item["bottom_box"], Path(item["old_models"]["old_full"]).parent),
            "old_sensor": (old_sensor, item["bottom_box"], Path(item["old_models"]["old_sensor"]).parent),
            "wrist_full": (native, item["wrist_box"], U / "outputs" / cid / "wrist_full"),
            "wrist_sensor": (new_sensor, item["wrist_box"], U / "outputs" / cid / "wrist_sensor"),
        }
        obs, checked_metrics, endpoint_metrics = {}, {}, {}
        for name, (source, box, directory) in definitions.items():
            ob = Observation.load(str(directory / "model_input.npz"))
            expected = mask_reference(source, box, (0, 0))
            for attr, reference in zip(("xy_px", "observed", "usable", "conf", "hand_score"), expected[:5]):
                equal(getattr(ob, attr), reference)
            equal(ob.frame_id, native.frame_id)
            equal(ob.timestamp_us, native.timestamp_us)
            ch = npz(directory / "channels.npz")
            x, body_scale, hand_scales = independent_channels(ob)
            assert x.shape == (native.n_frames, 280) and np.isfinite(x).all()
            equal(x, ch["x"])
            equal(ch["frame_id"], native.frame_id)
            equal(ch["timestamp_us"], native.timestamp_us)
            checked_metrics[name] = extra_metrics_reference(ob)
            assert checked_metrics[name] == saved["metrics"][name], (cid, name, "metrics")
            endpoint_metrics[name] = endpoint_reference(ob)
            changed = copy.deepcopy(source)
            hidden = ~expected[5] & source.observed
            changed.xy_px[hidden] = np.array([-100000, -200000], np.float32)
            changed.conf[hidden] = .123
            poison = build_roi_channels(apply_roi(changed, box))
            equal(poison["x"], x)
            assert poison["body_scale"] == body_scale
            np.testing.assert_array_equal(np.array(list(poison["hand_scales"].values())), np.array(hand_scales))
            hidden_points += int(hidden.sum())
            frames += native.n_frames
            conditions += 1
            obs[name] = ob
        assert not (obs["old_full"].usable & ~obs["wrist_full"].usable).any()
        assert not (obs["old_full"].observed & ~obs["wrist_full"].observed).any()
        checked_pairs = {name: diff_reference(obs[a], obs[b]) for name, (a, b) in pairs.items()}
        assert checked_pairs == saved["pairs"], (cid, "paired counts")
        recomputed.append(dict(clip_id=cid, label=item["label"], metrics=checked_metrics, pairs=checked_pairs))
        auditrows.append(dict(clip_id=cid, label=item["label"], frames=native.n_frames, conditions=4,
                              boundary=boundary, endpoint_metrics=endpoint_metrics, passed=True))
        print(json.dumps(dict(verified_clip=cid, conditions=conditions), ensure_ascii=False), flush=True)
    aggregate = {p: {m: {c: 0 for c in ("old", "new", "gain", "loss")} for m in ("any", "palm5", "all21", "points")} for p in pairs}
    for row in recomputed:
        for p in pairs:
            for m in aggregate[p]:
                for c in aggregate[p][m]:
                    aggregate[p][m][c] += row["pairs"][p]["left"][m][c] + row["pairs"][p]["right"][m][c]
    assert aggregate == result["aggregate"]
    endpoints = {}
    for mode in plan["conditions"]:
        endpoints[mode] = {}
        for window in ("first5", "last5", "first_half_second", "last_half_second"):
            rows = []
            for clip in recomputed:
                for side in ("left", "right"):
                    values = clip["metrics"][mode]["end_windows"][window]["hands"][side]
                    if values["points"]:
                        rows.append(dict(clip_id=clip["clip_id"], label=clip["label"], side=side, **values))
            endpoints[mode][window] = dict(total_hand_point_frames=sum(r["points"] for r in rows), nonempty_clip_sides=len(rows), rows=rows)
    assert endpoints == result["end_window_audit"]
    assert conditions == result["conditions"] == 80
    assert result["new_model_views"] == 40
    assert sum(r["frames"] for r in auditrows) == result["unique_video_frames"] == capture["frames"] == 2536
    assert result["training"] == result["classifier_inference"] == result["production_changes"] == capture["training"] == 0
    pincheck(plan["pins"])
    final = dict(status="PASS", reviewer="wrist_pair_verify", clips=20, conditions=conditions, unique_video_frames=2536,
                 condition_frames=frames, independently_recomputed_280channel_values=frames * 280,
                 source_detector_conditions_replayed=60, raw_hands_replayed=raw_hands,
                 protected_pins_checked=len(plan["pins"]), hidden_source_points_mutated=hidden_points,
                 boundary_range=[min(r["boundary"]["box"][3] for r in auditrows), max(r["boundary"]["box"][3] for r in auditrows)],
                 independent_of="producer boundary/mask/feature/metric formulae; frozen assignment replay intentionally uses same joint-v3 policy",
                 checks=["same fixed20 TRAIN clips and source/model/code/output pins unchanged", "all20 fixed first5 wrist boundaries independently calculated; top and width preserved",
                         "same joint-v3 replay exact for native20+oldbottom20+wristcrop20", "frame IDs and timestamps exact including empty frames",
                         "all75-point ROI mask exact and no old full-source observed/usable point lost", "all80 model-input tensors independently recalculated exactly",
                         "outside-source-coordinate perturbation cannot alter masked channels or normalization", "partial hands and missing wrists retain the existing channel contract",
                         "all paired gains/losses and frame lists independently recomputed", "all first/last5 and half-second window metrics independently recomputed",
                         "new crops use correct RGB sensor dimensions and20 distinct fresh detector instances"],
                 limitations=["PASS covers reproducible engineering and reporting, not candidate adoption or recognition improvement",
                              "Some startup/end windows retain hand points; see endpoint audit, no universal idle-hand exclusion claim",
                              "Initial five frames are fixed calibration samples, not a reliable automatic rest detector",
                              "Fullframe detector sees below-ROI context; wristcrop models do not; no actual webcam test",
                              "Feature normalization uses completed-clip medians, not causal streaming normalization",
                              "No hand ground truth, active interval labels, classifier inference, training, or production change"],
                 aggregate=aggregate, end_window_audit=endpoints, rows=auditrows, seconds=time.perf_counter() - started,
                 inputs={str(U / name): sha(U / name) for name in ("PLAN.json", "CAPTURE_RESULT.json", "COMPARISON.json", "run_wrist_pair.py", "verify_wrist_pair.py")})
    with outpath.open("x", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: final[k] for k in ("status", "conditions", "condition_frames", "raw_hands_replayed", "hidden_source_points_mutated", "protected_pins_checked")}), flush=True)


if __name__ == "__main__":
    main()

